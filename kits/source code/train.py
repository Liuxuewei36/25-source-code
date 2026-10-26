# =============================================================================
# train.py — KiTS19 2.5D Baseline 训练主循环
# =============================================================================
# 【文件职责】
#   - 定义 Loss（Dice Loss + CrossEntropy，直接写在本文件）
#   - train_one_epoch：单 epoch 训练
#   - val_one_epoch  ：单 epoch 验证，返回 kidney/tumor/mean Dice
#   - main           ：组装所有模块，循环训练，保存 checkpoint，写日志
#
# 【Loss 设计说明】
#   KiTS19 有严重的类别不平衡：
#     背景像素   >> 肾脏像素 >> 肿瘤像素
#   单用 CrossEntropy：模型容易偷懒全预测背景，tumor Dice 接近 0
#   单用 Dice Loss  ：早期梯度不稳定，容易震荡
#   组合使用：CE 保证基础优化稳定，Dice 直接优化目标指标
#
#   total_loss = CE_WEIGHT * ce_loss + DICE_WEIGHT * dice_loss
#
# 【logits / softmax / CE / Dice 的关系，务必看清楚】
#   模型输出：raw logits，shape [B, C, H, W]，未经任何激活
#   CE loss ：nn.CrossEntropyLoss 内部自动做 log_softmax
#             → 输入必须是 raw logits，❌ 不要提前 softmax
#   Dice loss：需要概率值 → 本文件里手动 softmax 后再计算
#             → ❌ 不要对同一个 logits 重复 softmax
#   推理时  ：logits.argmax(dim=1) 直接取类别，不需要 softmax
# =============================================================================

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from kits import config
from kits.dataset import get_dataloaders, compute_class_pixel_freq
from kits.model import UNet2D
from kits.utils import set_seed, ensure_dirs, save_checkpoint, CSVLogger
from torch.cuda.amp import autocast, GradScaler

# =============================================================================
# Loss 函数
# =============================================================================

class DiceLoss(nn.Module):
    """
    多类别 Soft Dice Loss —— 仅计算前景类（排除背景 class 0）。

    【与原版的区别】
    1. 只对前景类（kidney=1, tumor=2）计算 Dice，排除背景（class 0）
       原因：背景占 95%+ 像素，纳入平均会稀释前景类梯度，
             导致模型更擅长"前景 vs 背景"而不是"kidney vs tumor"
    2. smooth 从 1e-6 改为 config.DICE_SMOOTH（默认 1.0）
       原因：AMP 半精度下 1e-6 可能导致数值不稳定

    【计算流程】
    1. logits → softmax → probs [B, C, H, W]
    2. targets → one-hot [B, C, H, W]
    3. 对前景类 c=1,2 分别计算 dice_c
    4. 对前景类取平均 → loss = 1 - mean(dice_fg)
    """

    def __init__(self, smooth: float = config.DICE_SMOOTH,
                 num_classes: int = config.NUM_CLASSES,
                 ignore_index: int = 0):
        super().__init__()
        self.smooth       = smooth
        self.num_classes  = num_classes
        self.ignore_index = ignore_index   # 排除背景类

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        参数：
            logits : [B, C, H, W]  raw logits
            targets: [B, H, W]     int64, 值 ∈ {0, 1, ..., C-1}
        返回：
            scalar tensor, 前景类 dice loss 均值
        """
        B, C, H, W = logits.shape

        probs = F.softmax(logits, dim=1)

        targets_onehot = F.one_hot(targets, num_classes=C)    # [B, H, W, C]
        targets_onehot = targets_onehot.permute(0, 3, 1, 2)   # [B, C, H, W]
        targets_onehot = targets_onehot.float()

        probs_flat   = probs.view(B, C, -1)                    # [B, C, H*W]
        targets_flat = targets_onehot.view(B, C, -1)           # [B, C, H*W]

        intersection = (probs_flat * targets_flat).sum(dim=2)  # [B, C]
        union        = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)  # [B, C]

        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)
        # [B, C]

        # 只对前景类取平均（排除 class 0 = background）
        fg_dice = dice_per_class[:, self.ignore_index + 1:]    # [B, C-1]
        loss = 1.0 - fg_dice.mean()

        return loss


class CombinedLoss(nn.Module):
    """
    Dice Loss + 加权 CrossEntropy Loss 组合。

    total_loss = ce_weight * CE + dice_weight * Dice

    【与原版的区别】
    1. CE 支持传入 class_weight（降低背景权重，提高前景权重）
    2. DiceLoss 已改为只对前景类计算

    【权重来源】
    - 若 ce_class_weights 为 None：CE 不加权（退化为原版行为）
    - 若传入 Tensor   ：直接使用
    - 推荐做法：在 main() 里统计训练集像素频率，算出逆频率权重后传入
    """

    def __init__(self,
                 ce_weight  : float = config.CE_WEIGHT,
                 dice_weight: float = config.DICE_WEIGHT,
                 ce_class_weights: torch.Tensor | None = None):
        super().__init__()
        self.ce_weight   = ce_weight
        self.dice_weight = dice_weight

        # CrossEntropyLoss：内部自带 log_softmax，输入必须是 raw logits
        if ce_class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=ce_class_weights)
            print(f"[loss] CE class weights: {ce_class_weights.tolist()}")
        else:
            self.ce_loss = nn.CrossEntropyLoss()
            print(f"[loss] CE 无类别权重（原版行为）")

        self.dice_loss = DiceLoss()

    def forward(self,
                logits : torch.Tensor,   # [B, C, H, W] raw logits
                targets: torch.Tensor    # [B, H, W]    int64
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回：
            total_loss : scalar，用于 backward
            ce_loss    : scalar，仅用于日志记录
            dice_loss  : scalar，仅用于日志记录
        """
        ce   = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        total = self.ce_weight * ce + self.dice_weight * dice
        return total, ce, dice


# =============================================================================
# 验证指标：Hard Dice（基于 argmax 预测）
# =============================================================================




# =============================================================================
# 单 epoch 训练
# =============================================================================

def train_one_epoch(
    model     : nn.Module,
    loader    : DataLoader,
    optimizer : torch.optim.Optimizer,
    criterion : CombinedLoss,
    device    : str,
    epoch     : int,
    scaler    : GradScaler | None,   # [修正 1] AMP GradScaler，None 表示不用 AMP
) -> float:

    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Train]", leave=False)

    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        # [修正 1] AMP autocast：有 scaler 时开混合精度，否则普通 float32
        with autocast(enabled=(scaler is not None)):
            logits = model(x)                    # [B, C, H, W]
            loss, ce_val, dice_val = criterion(logits, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "ce"  : f"{ce_val.item():.4f}",
            "dice": f"{dice_val.item():.4f}",
        })

    # [修正 2] 防止空 loader 导致除零
    mean_loss = total_loss / max(len(loader), 1)
    return mean_loss


# =============================================================================
# 单 epoch 验证
# =============================================================================

def val_one_epoch(
    model    : nn.Module,
    loader   : DataLoader,
    criterion: CombinedLoss,
    device   : str,
    epoch    : int,
) -> tuple[float, float, float, float]:
    model.eval()
    total_loss         = 0.0
    smooth             = 1e-6
    total_intersection = torch.zeros(config.NUM_CLASSES)
    total_union        = torch.zeros(config.NUM_CLASSES)
    pred_pixel_count   = torch.zeros(config.NUM_CLASSES)   # 监控预测类别分布

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Val  ]", leave=False)

    with torch.no_grad():
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast(enabled=torch.cuda.is_available() and config.USE_AMP):
                logits = model(x)
                loss, _, _ = criterion(logits, y)

            # NaN 防护
            if torch.isnan(loss):
                print(f"\n[train] ⚠️  Epoch {epoch} val loss=NaN，跳过本 batch")
                continue

            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu()
            y_cpu = y.cpu()

            for c in range(config.NUM_CLASSES):
                pred_c   = (preds == c).float()
                target_c = (y_cpu == c).float()
                total_intersection[c] += (pred_c * target_c).sum()
                total_union[c]        += pred_c.sum() + target_c.sum()
                pred_pixel_count[c]   += pred_c.sum()

            k_now = (2 * total_intersection[1] + smooth) / (total_union[1] + smooth)
            t_now = (2 * total_intersection[2] + smooth) / (total_union[2] + smooth)
            pbar.set_postfix({
                "loss"  : f"{loss.item():.4f}",
                "kidney": f"{k_now:.4f}",
                "tumor" : f"{t_now:.4f}",
            })

    mean_loss   = total_loss / max(len(loader), 1)
    dice_kidney = ((2 * total_intersection[1] + smooth) / (total_union[1] + smooth)).item()
    dice_tumor  = ((2 * total_intersection[2] + smooth) / (total_union[2] + smooth)).item()
    dice_mean   = (dice_kidney + dice_tumor) / 2.0

    # 打印预测类别像素占比，用于监控是否塌缩成全背景
    total_pred_pixels = pred_pixel_count.sum().item()
    if total_pred_pixels > 0:
        bg_ratio  = (pred_pixel_count[0] / total_pred_pixels).item()
        kid_ratio = (pred_pixel_count[1] / total_pred_pixels).item()
        tum_ratio = (pred_pixel_count[2] / total_pred_pixels).item()
        print(f"         预测像素占比 → bg:{bg_ratio:.3f} kidney:{kid_ratio:.3f} tumor:{tum_ratio:.3f}")

    return mean_loss, dice_kidney, dice_tumor, dice_mean

# =============================================================================
# 主训练入口
# =============================================================================

def main():
    set_seed(config.SEED)
    ensure_dirs(config.CHECKPOINT_DIR, config.FIGURE_DIR, config.LOG_DIR)

    device = config.DEVICE
    print(f"\n[train] 使用设备: {device}")

    use_amp = config.USE_AMP and (device == "cuda")
    scaler  = GradScaler() if use_amp else None
    print(f"[train] AMP 混合精度: {'开启' if use_amp else '关闭'}")

    print("[train] 加载数据集...")
    train_loader, val_loader = get_dataloaders()

    # ---- 统计训练集像素频率，计算 CE 类别权重 ----
    if config.CE_CLASS_WEIGHTS is None:
        print("[train] 统计训练集像素频率（用于 CE 权重）...")
        freq = compute_class_pixel_freq(train_loader)
        print(f"[train] 像素频率: bg={freq[0]:.4f}, kidney={freq[1]:.4f}, tumor={freq[2]:.4f}")

        # 逆频率加权：频率越低 → 权重越高
        # 归一化使最小权重=1.0，避免 loss 数值量级过大
        ce_weights = 1.0 / (freq + 1e-8)
        ce_weights = ce_weights / ce_weights.min()
        ce_weights = ce_weights.float().to(device)
        print(f"[train] 自动 CE 权重: {ce_weights.tolist()}")
    else:
        ce_weights = torch.tensor(config.CE_CLASS_WEIGHTS, dtype=torch.float32).to(device)
        print(f"[train] 手动 CE 权重: {ce_weights.tolist()}")

    print("[train] 初始化模型...")
    model = UNet2D(
        in_ch       = config.IN_CHANNELS,
        num_classes = config.NUM_CLASSES,
        base_ch     = config.BASE_CHANNELS,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] 可训练参数量: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "max",               # 监控 val mean Dice，越大越好
        factor   = 0.5,                  # 触发时 LR 减半
        patience = config.LR_PATIENCE,   # 连续 N epoch 无提升才触发
        min_lr   = 1e-5,                 # LR 下限
    )

    criterion = CombinedLoss(
        ce_weight        = config.CE_WEIGHT,
        dice_weight      = config.DICE_WEIGHT,
        ce_class_weights = ce_weights,
    )

    # 用新文件名，避免和旧 run 的日志混在一起
    log_path       = os.path.join(config.LOG_DIR, "train_log_v6.csv")
    logger         = CSVLogger(log_path)
    best_dice      = 0.0
    best_ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    last_ckpt_path = os.path.join(config.CHECKPOINT_DIR, "last_model.pth")

    print(f"\n[train] 开始训练，共 {config.NUM_EPOCHS} epochs\n")
    # 改动三：表头加 lr 列
    print(f"{'Epoch':>6} | {'TrainLoss':>10} | {'ValLoss':>10} | "
          f"{'KidneyDice':>11} | {'TumorDice':>10} | {'MeanDice':>9} | {'LR':>8}")
    print("-" * 80)

    for epoch in range(1, config.NUM_EPOCHS + 1):

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, scaler
        )
        val_loss, dice_kidney, dice_tumor, dice_mean = val_one_epoch(
            model, val_loader, criterion, device, epoch
        )

        # 改动三：scheduler 调度 + LR 打印 + NaN 检测
        scheduler.step(dice_mean)
        current_lr = optimizer.param_groups[0]["lr"]

        nan_flag = "  ⚠️ NaN" if (
            torch.isnan(torch.tensor(train_loss)) or
            torch.isnan(torch.tensor(val_loss))
        ) else ""

        print(f"{epoch:>6} | {train_loss:>10.4f} | {val_loss:>10.4f} | "
              f"{dice_kidney:>11.4f} | {dice_tumor:>10.4f} | {dice_mean:>9.4f} | "
              f"{current_lr:>8.1e}{nan_flag}")

        logger.write(
            epoch           = epoch,
            train_loss      = train_loss,
            val_loss        = val_loss,
            val_dice_kidney = dice_kidney,
            val_dice_tumor  = dice_tumor,
            val_dice_mean   = dice_mean,
        )

        save_checkpoint(
            state={
                "epoch"    : epoch,
                "model"    : model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_dice" : dice_mean,
            },
            path=last_ckpt_path,
        )

        if dice_mean > best_dice:
            best_dice = dice_mean
            save_checkpoint(
                state={
                    "epoch"    : epoch,
                    "model"    : model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_dice" : dice_mean,
                },
                path=best_ckpt_path,
            )
            print(f"         ↑ best model 已更新 (mean_dice={best_dice:.4f})")

    print("\n" + "=" * 80)
    print(f"[train] 训练完成。Best Val Mean Dice: {best_dice:.4f}")
    print(f"[train] Best checkpoint : {best_ckpt_path}")
    print(f"[train] Last checkpoint : {last_ckpt_path}")
    print(f"[train] 训练日志        : {log_path}")
    print("=" * 80)


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    main()