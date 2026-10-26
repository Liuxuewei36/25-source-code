# brats/loss.py
# ─────────────────────────────────────────────────────────────
# 实现两个 loss：
#   1. DiceLoss      —— 纯 Dice，适合类别不平衡
#   2. DiceCELoss    —— Dice + CrossEntropy 的加权组合，训练更稳
#
# ★ 关于 logits / softmax / CrossEntropyLoss 的关系（必读）：
#
#   模型输出：logits，shape [B, C, D, H, W]，原始未激活值，没有 softmax
#
#   CrossEntropyLoss（PyTorch 内置）：
#     - 内部自动做 log_softmax，所以【直接接收 logits】
#     - 如果你在模型里或 loss 里提前做了 softmax 再送进来，等于做了两次，loss 会偏小且梯度错误
#     - target 是 LongTensor，shape [B, D, H, W]，值 ∈ {0,1,2,3}
#
#   DiceLoss（本文件实现）：
#     - PyTorch 没有内置，需要手动实现
#     - 需要概率值（0~1）才能计算 overlap，所以【内部自己做 softmax】
#     - 同一份 logits，CE 和 Dice 各自独立处理，互不干扰
#
#   总结：
#     logits → CrossEntropyLoss（内部 log_softmax）
#     logits → softmax → DiceLoss
#     两个 loss 共用同一份 logits，分别处理
# ─────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════
# Dice Loss
# ══════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    """
    多类别 Soft Dice Loss。

    输入：
        logits : [B, C, D, H, W]  —— 模型原始输出，未经 softmax
        target : [B, D, H, W]     —— 整数标签，值 ∈ {0, 1, ..., C-1}

    输出：
        scalar，平均 Dice Loss（值越小越好，=1 - 平均 Dice）

    计算步骤：
        1. logits → softmax → pred_prob，shape [B, C, D, H, W]
        2. target → one-hot，shape [B, C, D, H, W]
        3. 对每个类别 c，计算 Dice = 2*|P∩G| / (|P|+|G|)
        4. Dice Loss = 1 - Dice
        5. 对所有类别取平均（包含背景类）

    关于空类别（某个 patch 里某类不存在）的处理策略：
        当某类的 GT 全为 0（该类在这个 patch 里不存在）时，
        分子 = 0，分母 ≈ smooth（极小值）
        → Dice ≈ 0 / smooth ≈ 0，Dice Loss ≈ 1
        这会造成梯度惩罚一个根本不存在的类。

        当前 MVP 处理策略：
        【在分母加 smooth=1e-5 防止除零，但不跳过空类别】
        → 简单稳定，代价是空类 loss 偏高，对总 loss 有轻微干扰

        ★ 标准做法（后续优化）：检测空类并跳过其 Dice 计算，
          或只对 GT 中存在的类计算 Dice 再平均，
          但 MVP 阶段当前未实现。
    """

    def __init__(self, smooth: float = 1e-5, include_background: bool = True):
        """
        参数：
            smooth             : 分母平滑项，防止除零
            include_background : 是否把背景类（class 0）纳入 Dice 计算
                                 True  = 所有 C 类都算，再平均
                                 False = 跳过 class 0，只算前景类
                                 ★ 待确认：BraTS 任务通常 include_background=True，
                                   因为背景占大多数，对 loss 有"稳定"作用；
                                   但也有实现选择 False，两种都合理。
                                   当前默认 True（更简单，方便 MVP 跑通）。
        """
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits : [B, C, D, H, W]
        target : [B, D, H, W]，LongTensor
        """
        B, C, D, H, W = logits.shape

        # ── Step 1: logits → 概率（softmax）──────────────────
        # dim=1 是类别维度
        pred_prob = F.softmax(logits, dim=1)
        # pred_prob: [B, C, D, H, W]，每个体素在 C 类上的概率之和=1

        # ── Step 2: target → one-hot ──────────────────────────
        # target: [B, D, H, W]，需要变成 [B, C, D, H, W]
        #
        # 操作步骤：
        #   a. target: [B, D, H, W] → unsqueeze(1) → [B, 1, D, H, W]
        #   b. 创建全 0 的 [B, C, D, H, W] tensor
        #   c. scatter_(dim=1, index, value=1)：在类别维度上"打点"

        target_onehot = torch.zeros_like(pred_prob)
        # target_onehot: [B, C, D, H, W]，全 0

        target_onehot.scatter_(
            1,                        # dim=1，在类别维度上操作
            target.unsqueeze(1),      # index: [B, 1, D, H, W]，值是类别编号
            1.0                       # 填 1
        )
        # target_onehot: [B, C, D, H, W]，每个体素只有对应类别位置=1，其余=0

        # ── Step 3: 决定参与计算的类别范围 ───────────────────
        start_class = 0 if self.include_background else 1
        # include_background=True  → start_class=0，计算全部 C 类
        # include_background=False → start_class=1，跳过背景类

        # ── Step 4: 逐类计算 Dice Loss ───────────────────────
        dice_loss_per_class = []

        for c in range(start_class, C):
            p = pred_prob[:, c, ...]      # [B, D, H, W]，第 c 类的预测概率
            g = target_onehot[:, c, ...]  # [B, D, H, W]，第 c 类的 GT one-hot

            # Soft Dice 分子：2 * Σ(p * g)
            # .flatten(1) 把 [B, D, H, W] 展平成 [B, D*H*W]，再 sum(dim=1) → [B]
            intersection = (p * g).flatten(1).sum(dim=1)   # [B]

            # Soft Dice 分母：Σ(p) + Σ(g)
            pred_sum = p.flatten(1).sum(dim=1)              # [B]
            gt_sum   = g.flatten(1).sum(dim=1)              # [B]

            # Dice 系数（带 smooth 防除零）
            dice_coeff = (2.0 * intersection + self.smooth) / \
                         (pred_sum + gt_sum + self.smooth)  # [B]

            # Dice Loss = 1 - Dice（batch 内取平均）
            dice_loss_per_class.append(1.0 - dice_coeff.mean())

        # ── Step 5: 所有类别的 Dice Loss 取平均 ──────────────
        total_dice_loss = torch.stack(dice_loss_per_class).mean()
        # scalar

        return total_dice_loss


# ══════════════════════════════════════════════════════════════
# Dice + CrossEntropy 组合 Loss
# ══════════════════════════════════════════════════════════════

class DiceCELoss(nn.Module):
    """
    DiceLoss + CrossEntropyLoss 的加权组合。

    为什么要组合两个 loss？
        - Dice Loss：直接优化 Dice 指标，对类别不平衡鲁棒，
                     但梯度稀疏，训练初期收敛慢
        - CrossEntropy Loss：逐体素分类 loss，梯度密集，
                              收敛快，但对类别不平衡敏感
        - 组合后：CE 负责"快速找到方向"，Dice 负责"精细优化边界"
                  两者互补，训练更稳定

    公式：
        total_loss = w_dice * DiceLoss + w_ce * CELoss

    输入：
        logits : [B, C, D, H, W]  —— 原始 logits，不经过任何激活
        target : [B, D, H, W]     —— LongTensor，值 ∈ {0,...,C-1}

    ★ 再次强调 logits 处理方式：
        - DiceLoss.forward()  内部做 softmax（见上方）
        - nn.CrossEntropyLoss 内部做 log_softmax
        - 两者接收的都是同一份原始 logits，各自独立处理
        - 绝对不要在调用 loss 之前提前做 softmax
    """

    def __init__(
        self,
        w_dice: float = 0.5,
        w_ce:   float = 0.5,
        smooth: float = 1e-5,
        include_background: bool = True,
    ):
        super().__init__()
        self.w_dice = w_dice
        self.w_ce   = w_ce

        self.dice_loss = DiceLoss(
            smooth=smooth,
            include_background=include_background,
        )

        # nn.CrossEntropyLoss：
        #   - 接收 logits [B, C, D, H, W] 和 target [B, D, H, W]
        #   - 内部做 log_softmax + NLL loss
        #   - weight 参数可以给不同类别设置不同权重（当前未设，默认等权）
        #   ★ 待确认：是否需要对 BraTS 4 类设置 class_weight 来缓解不平衡
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        """
        logits : [B, C, D, H, W]  原始 logits
        target : [B, D, H, W]     LongTensor

        返回：(total_loss, dice_loss_val, ce_loss_val) 三元组
              total_loss 用于 backward()
              后两个用于日志记录，方便监控两个分量的变化
        """
        # ── Dice Loss ─────────────────────────────────────────
        # 内部会对 logits 做 softmax
        loss_dice = self.dice_loss(logits, target)

        # ── CrossEntropy Loss ──────────────────────────────────
        # 直接接收原始 logits，内部做 log_softmax
        # target 必须是 LongTensor（int64），shape [B, D, H, W]
        loss_ce = self.ce_loss(logits, target)

        # ── 加权求和 ───────────────────────────────────────────
        total = self.w_dice * loss_dice + self.w_ce * loss_ce

        return total, loss_dice, loss_ce


# ══════════════════════════════════════════════════════════════
# Loss 工厂函数（给 train.py 用）
# ══════════════════════════════════════════════════════════════

def build_loss(cfg: dict) -> nn.Module:
    """
    根据 config 里的 loss_type 返回对应的 loss 函数。
    train.py 只需调用这一个函数，不用关心具体实现。

    支持：
        cfg["loss_type"] = "dice"    → DiceLoss
        cfg["loss_type"] = "dice_ce" → DiceCELoss（默认推荐）
    """
    loss_type = cfg.get("loss_type", "dice_ce")

    if loss_type == "dice":
        print("[Loss] 使用 DiceLoss")
        return DiceLoss(smooth=1e-5, include_background=True)

    elif loss_type == "dice_ce":
        print(f"[Loss] 使用 DiceCELoss  "
              f"(w_dice={cfg['w_dice']}, w_ce={cfg['w_ce']})")
        return DiceCELoss(
            w_dice=cfg["w_dice"],
            w_ce=cfg["w_ce"],
            smooth=1e-5,
            include_background=True,
        )

    else:
        raise ValueError(f"[Loss] 不支持的 loss_type: '{loss_type}'，"
                         f"请设置为 'dice' 或 'dice_ce'")


# ══════════════════════════════════════════════════════════════
# 快速自测
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from brats.config import cfg as global_cfg

    print("=" * 50)
    print("Loss Sanity Check")
    print("=" * 50)

    B, C  = 2, 4
    D, H, W = 16, 32, 32     # 小 patch，测试用

    # 构造假 logits 和假 target
    torch.manual_seed(42)
    dummy_logits = torch.randn(B, C, D, H, W)   # [B, C, D, H, W]
    dummy_target = torch.randint(0, C, (B, D, H, W))  # [B, D, H, W]，值 ∈ {0,1,2,3}

    print(f"\n输入 logits shape : {dummy_logits.shape}")
    print(f"输入 target shape : {dummy_target.shape}")
    print(f"target 唯一值     : {dummy_target.unique().tolist()}")

    # ── 测试 DiceLoss ─────────────────────────────────────────
    dice_fn = DiceLoss(smooth=1e-5)
    loss_d  = dice_fn(dummy_logits, dummy_target)
    print(f"\n[DiceLoss]    loss = {loss_d.item():.4f}  （期望在 0~1 之间）")

    # ── 测试 DiceCELoss ───────────────────────────────────────
    dicece_fn = DiceCELoss(w_dice=0.5, w_ce=0.5)
    total, ld, lce = dicece_fn(dummy_logits, dummy_target)
    print(f"[DiceCELoss]  total={total.item():.4f}  "
          f"dice={ld.item():.4f}  ce={lce.item():.4f}")

    # ── 测试 backward ─────────────────────────────────────────
    dummy_logits_grad = dummy_logits.clone().requires_grad_(True)
    total2, _, _ = DiceCELoss()(dummy_logits_grad, dummy_target)
    total2.backward()
    print(f"\n[Backward]    grad norm = {dummy_logits_grad.grad.norm().item():.4f}")
    print("（grad norm > 0 说明梯度正常回传）")

    # ── 测试 build_loss ───────────────────────────────────────
    loss_fn = build_loss(global_cfg)
    print(f"\n[build_loss]  返回类型: {type(loss_fn).__name__}")

    print("\n✅ Loss sanity check 通过！可以继续写 train.py")