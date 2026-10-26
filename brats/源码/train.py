# brats/train.py
# ─────────────────────────────────────────────────────────────
# 功能：BraTS3D U-Net 训练主循环
#
# 支持：
#   - GPU 训练（自动检测 CUDA）
#   - AMP 混合精度（torch.cuda.amp，可在 config 开关）
#   - train / val 循环，每 epoch 记录 loss 和 Dice
#   - 保存 best checkpoint（val dice 最高）和 last checkpoint
#   - 训练日志写入 CSV（方便后续画曲线）
#
# 当前 MVP 不包含（后续优化方向）：
#   - LR scheduler（待确认：可后续加 CosineAnnealingLR）
#   - Early stopping
#   - 滑窗推理（验证集当前用中心 patch 裁剪，见 dataset.py 说明）
#   - DDP 多卡训练
#
# ★ 关于空类别 Dice 的说明：
#   验证时计算 per-class Dice，某类在 patch 里不存在时，
#   该类 GT 全为 0，Dice 分母极小，Dice ≈ 0。
#   当前 MVP 策略：把空类 Dice 也纳入平均（偏保守，指标会偏低）。
#   标准做法是跳过空类再平均，后续可升级。
# ─────────────────────────────────────────────────────────────

import os
import sys
import csv
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# 把项目根目录加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brats.config  import cfg
from brats.dataset import BraTSDataset, get_case_list, split_dataset
from brats.model   import UNet3D
from brats.loss    import build_loss
from brats.evaluate import compute_brats_metrics_correct

# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════

def compute_dice_per_class(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-5,
) -> list:
    """
    计算每个类别的 Dice 系数（用于验证/评估，不用于 loss）。

    参数：
        pred       : [B, D, H, W]，argmax 后的预测标签，LongTensor
        target     : [B, D, H, W]，GT 标签，LongTensor
        num_classes: 类别数（4）
        smooth     : 防除零

    返回：
        list of float，长度 = num_classes，每个元素是该类的平均 Dice

    ★ 当前 MVP 策略：
        空类（GT 中该类不存在）→ Dice = 0，纳入平均
        这会使指标偏保守（偏低），但实现简单。
        标准做法：跳过空类，只对存在的类求平均。
    """
    dice_scores = []

    for c in range(num_classes):
        pred_c   = (pred == c).float()    # [B, D, H, W]，预测为类 c 的位置
        target_c = (target == c).float()  # [B, D, H, W]，GT 为类 c 的位置

        intersection = (pred_c * target_c).sum()
        pred_sum     = pred_c.sum()
        target_sum   = target_c.sum()

        dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
        dice_scores.append(dice.item())

    return dice_scores  # [dice_c0, dice_c1, dice_c2, dice_c3]





def save_checkpoint(state: dict, path: str):
    """保存 checkpoint 到指定路径。"""
    torch.save(state, path)
    print(f"  [Checkpoint] 已保存 → {path}")


def load_checkpoint(path: str, model: nn.Module, optimizer=None):
    """加载 checkpoint，返回起始 epoch 和最佳 val dice。"""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_dice   = ckpt.get("best_val_dice", 0.0)
    print(f"  [Checkpoint] 已加载 {path}，从 epoch {start_epoch} 继续")
    return start_epoch, best_dice


# ══════════════════════════════════════════════════════════════
# 单 Epoch 训练
# ══════════════════════════════════════════════════════════════

def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
    scaler,          # AMP GradScaler，若不用 AMP 则传 None
    epoch_idx: int,
    use_amp: bool,
) -> dict:
    """
    训练一个 epoch，返回平均 loss 统计。

    返回：
        dict {
            "loss_total": float,
            "loss_dice":  float,   # 若 loss_fn 是 DiceCELoss
            "loss_ce":    float,   # 若 loss_fn 是 DiceCELoss
        }
    """
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_ce   = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch_idx:03d} [Train]", leave=False)

    for batch_x, batch_y in pbar:
        # batch_x: [B, 4, D, H, W]，FloatTensor
        # batch_y: [B, D, H, W],    LongTensor
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()

        # ── 前向传播（支持 AMP）────────────────────────────────
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                # logits: [B, 4, D, H, W]，原始 logits，未经 softmax
                logits = model(batch_x)

                # loss_fn 接收 logits 和 target
                # DiceCELoss 返回 (total, dice, ce)
                # DiceLoss   返回 scalar
                loss_out = loss_fn(logits, batch_y)

            # 解包 loss
            if isinstance(loss_out, tuple):
                loss, loss_d, loss_c = loss_out
            else:
                loss, loss_d, loss_c = loss_out, loss_out, torch.tensor(0.0)

            # AMP 反向传播：scaler 防止 float16 下梯度下溢
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            # 普通 float32 训练
            logits   = model(batch_x)
            loss_out = loss_fn(logits, batch_y)

            if isinstance(loss_out, tuple):
                loss, loss_d, loss_c = loss_out
            else:
                loss, loss_d, loss_c = loss_out, loss_out, torch.tensor(0.0)

            loss.backward()
            optimizer.step()

        # 累计统计
        total_loss += loss.item()
        total_dice += loss_d.item()
        total_ce   += loss_c.item() if isinstance(loss_c, torch.Tensor) \
                      else loss_c
        n_batches  += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "dice_loss": f"{loss_d.item():.4f}",
        })

    return {
        "loss_total": total_loss / n_batches,
        "loss_dice":  total_dice / n_batches,
        "loss_ce":    total_ce   / n_batches,
    }


# ══════════════════════════════════════════════════════════════
# 单 Epoch 验证
# ══════════════════════════════════════════════════════════════

def val_one_epoch(
    model,
    loader,
    loss_fn,
    device,
    num_classes: int,
    use_amp: bool,
    epoch_idx: int,
) -> dict:
    """
    验证一个 epoch，返回 loss 和 Dice 统计。

    返回：
        dict {
            "loss_total": float,
            "dice_mean":  float,          # 4 类平均 Dice
            "dice_per_class": list,       # [d0, d1, d2, d3]
            "WT": float, "TC": float, "ET": float,
        }
    """
    model.eval()
    total_loss = 0.0
    accum_dice = [0.0] * num_classes
    accum_wt, accum_tc, accum_et = 0.0, 0.0, 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch_idx:03d} [Val]  ", leave=False)

    with torch.no_grad():
        for batch_x, batch_y in pbar:
            # batch_x: [B, 4, D, H, W]
            # batch_y: [B, D, H, W]
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            # ── 前向传播 ───────────────────────────────────────
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(batch_x)
                    loss_out = loss_fn(logits, batch_y)
            else:
                logits   = model(batch_x)
                loss_out = loss_fn(logits, batch_y)

            # 解包 loss
            if isinstance(loss_out, tuple):
                loss = loss_out[0]
            else:
                loss = loss_out

            # ── 计算预测标签 ────────────────────────────────────
            # logits: [B, 4, D, H, W]
            # argmax 在类别维度(dim=1)取最大值 → pred: [B, D, H, W]
            pred = torch.argmax(logits, dim=1)  # [B, D, H, W]，值 ∈ {0,1,2,3}

            # ── 计算 Dice ───────────────────────────────────────
            # 把 pred 和 target 移回 CPU 算 Dice（避免占用 GPU 显存）
            dice_list = compute_dice_per_class(
                pred.cpu(), batch_y.cpu(), num_classes
            )
            for c in range(num_classes):
                accum_dice[c] += dice_list[c]
            pred_cpu   = pred.squeeze(0).cpu()     # [D, H, W]
            target_cpu = batch_y.squeeze(0).cpu()  # [D, H, W]

            brats_metrics = compute_brats_metrics_correct(pred_cpu, target_cpu)
            accum_wt += brats_metrics["WT"]
            accum_tc += brats_metrics["TC"]
            accum_et += brats_metrics["ET"]

            total_loss += loss.item()
            n_batches  += 1

            pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

    mean_dice_per_class = [accum_dice[c] / n_batches for c in range(num_classes)]
    mean_dice = sum(mean_dice_per_class) / num_classes

    mean_wt = accum_wt / n_batches
    mean_tc = accum_tc / n_batches
    mean_et = accum_et / n_batches

    return {
        "loss_total":     total_loss / n_batches,
        "dice_mean":      mean_dice,
        "dice_per_class": mean_dice_per_class,
        "WT":             mean_wt,
        "TC":             mean_tc,
        "ET":             mean_et,
    }


# ══════════════════════════════════════════════════════════════
# 主训练函数
# ══════════════════════════════════════════════════════════════

def train():
    """
    主训练入口，整合所有模块：
    Dataset → Model → Loss → Optimizer → Train/Val Loop → Checkpoint → CSV
    """

    # ── 0. 基础设置 ────────────────────────────────────────────
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] 使用设备: {device}")
    if device.type == "cuda":
        print(f"        GPU: {torch.cuda.get_device_name(0)}")
        print(f"        显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # cudnn benchmark：patch size 固定时开启，自动选最快卷积算法
    if cfg.get("cudnn_benchmark", True):
        torch.backends.cudnn.benchmark = True
        print("[Train] cudnn.benchmark = True（patch size 固定时有加速效果）")

    use_amp = cfg.get("use_amp", True) and (device.type == "cuda")
    print(f"[Train] AMP 混合精度: {'开启' if use_amp else '关闭'}")

    # 输出路径
    ckpt_dir = os.path.join(cfg["output_dir"], "checkpoints")
    log_dir  = os.path.join(cfg["output_dir"], "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)
    log_path = os.path.join(log_dir, "train_log.csv")

    # ── 1. 数据 ────────────────────────────────────────────────
    print("\n[Train] 准备数据集...")
    all_cases = get_case_list(
        data_root=cfg["data_root"],
        debug_num_cases=cfg["debug_num_cases"],
    )
    train_cases, val_cases = split_dataset(all_cases, cfg["train_ratio"])

    train_ds = BraTSDataset(
        data_root=cfg["data_root"],
        case_list=train_cases,
        patch_size=cfg["patch_size"],
        is_train=True,
    )
    val_ds = BraTSDataset(
        data_root=cfg["data_root"],
        case_list=val_cases,
        patch_size=cfg["patch_size"],
        is_train=False,   # 验证集用中心裁剪（★ 待确认：后续可换滑窗推理）
    )

    n_train = len(train_ds)
    drop_last = n_train >= cfg["batch_size"] * 2
    print(f"[Train] drop_last={drop_last}（训练样本数={n_train}，batch_size={cfg['batch_size']}）")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,     # 验证时固定 batch=1，稳定
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    print(f"[Train] train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    # ── 2. 模型 ────────────────────────────────────────────────
    print("\n[Train] 构建模型...")
    model = UNet3D(
        in_channels=cfg["in_channels"],       # 4
        num_classes=cfg["num_classes"],       # 4
        base_channels=cfg["base_channels"],   # 32（4090 推荐）
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] 可训练参数量: {total_params:,}")

    # ── 3. Loss ────────────────────────────────────────────────
    # build_loss 根据 cfg["loss_type"] 返回 DiceLoss 或 DiceCELoss
    # ★ DiceLoss：内部对 logits 做 softmax
    # ★ CrossEntropyLoss：内部做 log_softmax，直接接收 logits，不要提前激活
    loss_fn = build_loss(cfg)

    # ── 4. 优化器 ──────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],     # 默认 1e-4
        weight_decay=cfg["weight_decay"],
    )
    # ★ 待确认：可后续加 CosineAnnealingLR 或 ReduceLROnPlateau，当前 MVP 不加

    # ── 5. AMP GradScaler ─────────────────────────────────────
    # GradScaler 在 float16 下动态缩放梯度，防止梯度下溢
    # 不用 AMP 时传 None，train_one_epoch 会跳过 scaler 逻辑
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # ── 6. 初始化日志 CSV ──────────────────────────────────────
    csv_fields = [
        "epoch", "train_loss", "train_loss_dice", "train_loss_ce",
        "val_loss", "val_dice_mean",
        "val_dice_c0", "val_dice_c1", "val_dice_c2", "val_dice_c3",
        "val_WT", "val_TC", "val_ET",
        "epoch_time_s",
    ]
    # 如果是全新训练，写 header；如果是续训，追加
    write_header = not os.path.exists(log_path)
    log_file = open(log_path, "a", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=csv_fields)
    if write_header:
        log_writer.writeheader()

    # ── 7. 训练循环 ────────────────────────────────────────────
    best_val_dice = 0.0
    num_epochs    = cfg["num_epochs"]

    print(f"\n[Train] 开始训练，共 {num_epochs} epochs")
    print(f"[Train] patch_size={cfg['patch_size']}, batch_size={cfg['batch_size']}")
    print(f"[Train] loss_type={cfg['loss_type']}, lr={cfg['learning_rate']}")
    print("=" * 60)

    for epoch in range(1, num_epochs + 1):
        t_start = time.time()

        # ── 训练 ───────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scaler=scaler,
            epoch_idx=epoch,
            use_amp=use_amp,
        )

        # ── 验证（按 val_interval 决定是否跑）─────────────────
        run_val = (epoch % cfg["val_interval"] == 0) or (epoch == num_epochs)

        if run_val:
            val_metrics = val_one_epoch(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                num_classes=cfg["num_classes"],
                use_amp=use_amp,
                epoch_idx=epoch,
            )
        else:
            # 未运行验证时用占位值（不写 CSV）
            val_metrics = None

        t_end     = time.time()
        epoch_time = t_end - t_start

        # ── 打印当前 epoch 结果 ────────────────────────────────
        print(
            f"Epoch [{epoch:03d}/{num_epochs}] "
            f"TrainLoss={train_metrics['loss_total']:.4f} "
            + (
                f"ValLoss={val_metrics['loss_total']:.4f}  "
                f"ValDice={val_metrics['dice_mean']:.4f}  "
                f"WT={val_metrics['WT']:.4f} TC={val_metrics['TC']:.4f} ET={val_metrics['ET']:.4f}  "
                f"[{epoch_time:.1f}s]"
                if val_metrics else f"[{epoch_time:.1f}s] (跳过验证)"
            )
        )

        # ── 写 CSV ─────────────────────────────────────────────
        if val_metrics is not None:
            dc = val_metrics["dice_per_class"]
            log_writer.writerow({
                "epoch":           epoch,
                "train_loss":      f"{train_metrics['loss_total']:.6f}",
                "train_loss_dice": f"{train_metrics['loss_dice']:.6f}",
                "train_loss_ce":   f"{train_metrics['loss_ce']:.6f}",
                "val_loss":        f"{val_metrics['loss_total']:.6f}",
                "val_dice_mean":   f"{val_metrics['dice_mean']:.6f}",
                "val_dice_c0":     f"{dc[0]:.6f}",
                "val_dice_c1":     f"{dc[1]:.6f}",
                "val_dice_c2":     f"{dc[2]:.6f}",
                "val_dice_c3":     f"{dc[3]:.6f}",
                "val_WT":          f"{val_metrics['WT']:.6f}",
                "val_TC":          f"{val_metrics['TC']:.6f}",
                "val_ET":          f"{val_metrics['ET']:.6f}",
                "epoch_time_s":    f"{epoch_time:.1f}",
            })
            log_file.flush()   # 实时写盘，防止崩溃丢日志

        # ── 保存 Best Checkpoint ───────────────────────────────
        if val_metrics is not None:
            monitor_score = (val_metrics["WT"] + val_metrics["TC"] + val_metrics["ET"]) / 3.0
            if monitor_score > best_val_dice:
                best_val_dice = monitor_score
                save_checkpoint(
                    state={
                        "epoch":          epoch,
                        "model_state":    model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_val_dice":  best_val_dice,
                        "cfg":            cfg,
                    },
                    path=os.path.join(ckpt_dir, "model_best.pth"),
                )
                print(f"  ★ 新最佳 monitor_score = {best_val_dice:.4f}")

        # ── 保存 Last Checkpoint（按 save_interval）────────────
        if epoch % cfg["save_interval"] == 0 or epoch == num_epochs:
            save_checkpoint(
                state={
                    "epoch":           epoch,
                    "model_state":     model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_dice":   best_val_dice,
                    "cfg":             cfg,
                },
                path=os.path.join(ckpt_dir, "model_last.pth"),
            )

    # ── 8. 训练结束 ────────────────────────────────────────────
    log_file.close()
    print("\n" + "=" * 60)
    print(f"[Train] 训练完成！")
    print(f"[Train] Best val_dice = {best_val_dice:.4f}")
    print(f"[Train] 日志已保存 → {log_path}")
    print(f"[Train] Best checkpoint → {os.path.join(ckpt_dir, 'model_best.pth')}")


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train()