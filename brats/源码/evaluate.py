# brats/evaluate.py
# ─────────────────────────────────────────────────────────────
# 功能：加载 best checkpoint，在验证集上做推理，计算指标，
#       导出可视化所需的中间结果（供 visualize.py 使用）。
#
# 输出：
#   1. 终端打印 4 类基础 Dice + WT / TC / ET Dice
#   2. 导出一个 case 的推理结果到 outputs_brats/figures/eval_data.pt
#      包含：
#        - 输入图像 image
#        - GT mask
#        - Pred mask
#        - softmax 概率图
#        - best_slice_idx（推荐展示的 axial slice）
#
# ★ BraTS 指标计算说明（重要）：
#   正确做法：先把预测/GT 的相关类别合并成二值 mask，再重新计算 Dice
#   WT: pred/gt 中值属于 {1,2,3} 的体素 → 合并成二值 mask → 算 Dice
#   TC: pred/gt 中值属于 {1,3}   的体素 → 合并成二值 mask → 算 Dice
#   ET: pred/gt 中值等于 3       的体素 → 算 Dice
#
# ★ 关于概率热力图 vs Grad-CAM：
#   本文件导出的是模型 softmax 输出的概率图，不是 Grad-CAM。
#   概率热力图直接来自模型最终输出，含义更直接：
#     “模型认为这个体素属于某一类的概率是多少”
# ─────────────────────────────────────────────────────────────

import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brats.config import cfg
from brats.dataset import BraTSDataset, get_case_list, split_dataset
from brats.model import UNet3D


# ══════════════════════════════════════════════════════════════
# 核心指标计算函数（train.py 也可 import 这里的版本）
# ══════════════════════════════════════════════════════════════

def dice_binary(pred_mask: torch.Tensor,
                gt_mask: torch.Tensor,
                smooth: float = 1e-5) -> float:
    """
    计算两个二值 mask 的 Dice。

    参数：
        pred_mask : BoolTensor 或 0/1 Tensor，任意 shape
        gt_mask   : 同 shape
        smooth    : 防除零平滑项

    返回：
        float，Dice ∈ [0, 1]
    """
    pred_mask = pred_mask.float().reshape(-1)
    gt_mask = gt_mask.float().reshape(-1)

    intersection = (pred_mask * gt_mask).sum()
    denom = pred_mask.sum() + gt_mask.sum()

    return ((2.0 * intersection + smooth) / (denom + smooth)).item()


def compute_brats_metrics_correct(pred: torch.Tensor,
                                  target: torch.Tensor) -> dict:
    """
    正确计算 BraTS 三个标准区域的 Dice。

    参数：
        pred   : [D,H,W] 或 [B,D,H,W]，argmax 后预测标签
        target : 同 shape，GT 标签

    返回：
        {
            "WT": float,
            "TC": float,
            "ET": float,
        }

    区域定义：
        WT = {1,2,3}
        TC = {1,3}
        ET = {3}
    """
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    # WT
    pred_wt = (pred >= 1)
    gt_wt = (target >= 1)
    wt = dice_binary(pred_wt, gt_wt)

    # TC
    pred_tc = (pred == 1) | (pred == 3)
    gt_tc = (target == 1) | (target == 3)
    tc = dice_binary(pred_tc, gt_tc)

    # ET
    pred_et = (pred == 3)
    gt_et = (target == 3)
    et = dice_binary(pred_et, gt_et)

    return {"WT": wt, "TC": tc, "ET": et}


def compute_dice_per_class(pred: torch.Tensor,
                           target: torch.Tensor,
                           num_classes: int,
                           smooth: float = 1e-5) -> list:
    """
    计算每个类别的 Dice（硬标签版，用于评估，不用于 loss）。

    参数：
        pred   : [D,H,W] 或 [B,D,H,W]
        target : 同 shape
        num_classes : 类别数

    返回：
        list[float]，长度 = num_classes
    """
    pred = pred.reshape(-1).long()
    target = target.reshape(-1).long()

    dice_scores = []
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        denom = pred_c.sum() + target_c.sum()

        dice = ((2.0 * intersection + smooth) / (denom + smooth)).item()
        dice_scores.append(dice)

    return dice_scores


# ══════════════════════════════════════════════════════════════
# 概率图与推荐 slice
# ══════════════════════════════════════════════════════════════

def compute_probability_maps(logits: torch.Tensor) -> dict:
    """
    从模型 logits 计算 softmax 概率图。

    参数：
        logits : [1, C, D, H, W]

    返回：
        {
            "prob_c0": [D,H,W],
            "prob_c1": [D,H,W],
            "prob_c2": [D,H,W],
            "prob_c3": [D,H,W],
            "prob_ET": [D,H,W],
            "prob_WT": [D,H,W],
        }

    说明：
        这不是 Grad-CAM，而是 softmax 输出概率图。
        prob_WT = p1 + p2 + p3 = 1 - p0
        在 softmax 下值域严格位于 [0,1]，不会超过 1。
    """
    prob = F.softmax(logits, dim=1)   # [1, 4, D, H, W]
    prob = prob.squeeze(0)            # [4, D, H, W]

    prob_c0 = prob[0].cpu()
    prob_c1 = prob[1].cpu()
    prob_c2 = prob[2].cpu()
    prob_c3 = prob[3].cpu()

    prob_ET = prob_c3
    prob_WT = prob_c1 + prob_c2 + prob_c3   # = 1 - prob_c0

    return {
        "prob_c0": prob_c0,
        "prob_c1": prob_c1,
        "prob_c2": prob_c2,
        "prob_c3": prob_c3,
        "prob_ET": prob_ET,
        "prob_WT": prob_WT,
    }


def find_best_axial_slice(gt: torch.Tensor, pred: torch.Tensor) -> int:
    """
    找肿瘤面积最大的 axial slice index（沿 D 维度）。

    优先策略：
        1. 用 GT 的 WT 区域（标签属于 {1,2,3}）找面积最大 slice
        2. 如果 GT 全空，再用 pred 的 WT
        3. 如果 pred 也全空，返回中间 slice

    参数：
        gt   : [D, H, W]
        pred : [D, H, W]

    返回：
        int，推荐展示的 axial slice index
    """
    gt_wt = (gt >= 1).float()              # [D, H, W]
    areas = gt_wt.sum(dim=(1, 2))          # [D]

    if areas.max() > 0:
        return int(areas.argmax().item())

    pred_wt = (pred >= 1).float()
    areas_p = pred_wt.sum(dim=(1, 2))
    if areas_p.max() > 0:
        return int(areas_p.argmax().item())

    return gt.shape[0] // 2


# ══════════════════════════════════════════════════════════════
# 主评估函数
# ══════════════════════════════════════════════════════════════

def evaluate(export_case_idx: int = 0):
    """
    主评估入口。

    步骤：
        1. 加载 best checkpoint
        2. 在验证集上做推理
        3. 计算 4 类 Dice + WT/TC/ET Dice
        4. 导出一个 case 的推理结果供 visualize.py 使用
    """
    # ── 0. 设备 ─────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.get("use_amp", True) and (device.type == "cuda")
    print(f"[Evaluate] device = {device}, use_amp = {use_amp}")

    # ── 1. 数据集 ───────────────────────────────────────────
    all_cases = get_case_list(
        data_root=cfg["data_root"],
        debug_num_cases=cfg.get("debug_num_cases", None),
    )
    _, val_cases = split_dataset(
        all_cases,
        train_ratio=cfg["train_ratio"],
    )

    if len(val_cases) == 0:
        raise RuntimeError(
            "[Evaluate] 验证集为空。\n"
            "可能是 debug_num_cases 太小，比如 1 或 2。\n"
            "建议先把 debug_num_cases 设为 >=4。"
        )

    val_ds = BraTSDataset(
        data_root=cfg["data_root"],
        case_list=val_cases,
        patch_size=cfg["patch_size"],
        is_train=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print(f"[Evaluate] 验证集大小: {len(val_ds)} cases")

    # ── 2. 模型 & checkpoint ───────────────────────────────
    model = UNet3D(
        in_channels=cfg["in_channels"],
        num_classes=cfg["num_classes"],
        base_channels=cfg["base_channels"],
    ).to(device)

    ckpt_path = os.path.join(cfg["output_dir"], "checkpoints", "model_best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"[Evaluate] 找不到 checkpoint: {ckpt_path}\n"
            "请先运行 python brats/train.py"
        )

    ckpt = torch.load(ckpt_path, map_location=device)

    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        raise KeyError(
            "[Evaluate] checkpoint 中既没有 'model_state'，也没有 'model_state_dict'"
        )

    saved_epoch = ckpt.get("epoch", "?")
    saved_score = ckpt.get("monitor_score", ckpt.get("best_val_dice", "?"))
    print(f"[Evaluate] 已加载 checkpoint: epoch={saved_epoch}, saved_score={saved_score}")

    model.eval()

    # ── 3. 推理 + 指标累积 ─────────────────────────────────
    num_classes = cfg["num_classes"]

    accum_dice_per_class = [0.0] * num_classes
    accum_wt, accum_tc, accum_et = 0.0, 0.0, 0.0

    export_data = None
    n_cases = 0

    with torch.no_grad():
        for case_idx, (batch_x, batch_y) in enumerate(val_loader):
            batch_x = batch_x.to(device, non_blocking=True)   # [1, 4, D, H, W]
            batch_y = batch_y.to(device, non_blocking=True)   # [1, D, H, W]

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(batch_x)                   # [1, 4, D, H, W]
            else:
                logits = model(batch_x)

            pred = torch.argmax(logits, dim=1)               # [1, D, H, W]

            pred_cpu = pred.squeeze(0).cpu()                 # [D, H, W]
            target_cpu = batch_y.squeeze(0).cpu()            # [D, H, W]

            # per-class Dice
            dice_list = compute_dice_per_class(pred_cpu, target_cpu, num_classes)
            for c in range(num_classes):
                accum_dice_per_class[c] += dice_list[c]

            # WT / TC / ET
            brats_d = compute_brats_metrics_correct(pred_cpu, target_cpu)
            accum_wt += brats_d["WT"]
            accum_tc += brats_d["TC"]
            accum_et += brats_d["ET"]

            n_cases += 1

            # 导出一个 case 给 visualize.py
            if case_idx == export_case_idx:
                prob_maps = compute_probability_maps(logits.cpu())
                best_slice = find_best_axial_slice(target_cpu, pred_cpu)

                export_data = {
                    "image": batch_x.squeeze(0).cpu(),     # [4, D, H, W]
                    "gt": target_cpu,                      # [D, H, W]
                    "pred": pred_cpu,                      # [D, H, W]

                    "prob_c0": prob_maps["prob_c0"],
                    "prob_c1": prob_maps["prob_c1"],
                    "prob_c2": prob_maps["prob_c2"],
                    "prob_c3": prob_maps["prob_c3"],

                    "prob_ET": prob_maps["prob_ET"],
                    "prob_WT": prob_maps["prob_WT"],

                    "case_id": val_cases[case_idx],
                    "case_idx": case_idx,
                    "best_slice_idx": best_slice,
                }

                print(f"[Evaluate] 已选中 case {case_idx} 用于导出可视化数据")
                print(f"           case_id = {os.path.basename(val_cases[case_idx])}")
                print(f"           best_slice_idx = {best_slice}")

            print(
                f"  case {case_idx:03d}: "
                f"dice=[{dice_list[0]:.3f},{dice_list[1]:.3f},"
                f"{dice_list[2]:.3f},{dice_list[3]:.3f}]  "
                f"WT={brats_d['WT']:.3f} TC={brats_d['TC']:.3f} ET={brats_d['ET']:.3f}"
            )

    # ── 4. 汇总结果 ────────────────────────────────────────
    if n_cases == 0:
        raise RuntimeError("[Evaluate] 没有处理任何验证 case，请检查数据集。")

    mean_dice_per_class = [accum_dice_per_class[c] / n_cases for c in range(num_classes)]
    mean_wt = accum_wt / n_cases
    mean_tc = accum_tc / n_cases
    mean_et = accum_et / n_cases

    fg_dice_mean = sum(mean_dice_per_class[1:]) / (num_classes - 1)
    monitor_score = (mean_wt + mean_tc + mean_et) / 3.0

    print("\n" + "=" * 60)
    print("[Evaluate] 验证集评估结果汇总")
    print("=" * 60)
    print("  Per-class Dice（含背景）：")
    print(f"    class 0 (背景):  {mean_dice_per_class[0]:.4f}")
    print(f"    class 1 (坏死):  {mean_dice_per_class[1]:.4f}")
    print(f"    class 2 (水肿):  {mean_dice_per_class[2]:.4f}")
    print(f"    class 3 (ET):    {mean_dice_per_class[3]:.4f}")
    print(f"  前景 Dice 均值（c1+c2+c3）: {fg_dice_mean:.4f}")
    print("\n  BraTS 标准指标（合并 mask 后重算）：")
    print(f"    WT Dice: {mean_wt:.4f}   （Whole Tumor = 1∪2∪3）")
    print(f"    TC Dice: {mean_tc:.4f}   （Tumor Core  = 1∪3）")
    print(f"    ET Dice: {mean_et:.4f}   （Enhancing   = 3）")
    print(f"  Monitor score mean(WT,TC,ET): {monitor_score:.4f}")
    print("=" * 60)

    # ── 5. 导出可视化数据 ─────────────────────────────────
    fig_dir = os.path.join(cfg["output_dir"], "figures")
    os.makedirs(fig_dir, exist_ok=True)

    if export_data is not None:
        export_path = os.path.join(fig_dir, "eval_data.pt")
        torch.save(export_data, export_path)
        print(f"\n[Evaluate] 可视化数据已导出 → {export_path}")
        print(f"           包含字段: {list(export_data.keys())}")
        _print_export_shapes(export_data)
    else:
        print(
            f"\n[Evaluate] 警告：未找到 export_case_idx={export_case_idx} 的数据，"
            f"验证集共 {n_cases} 个 case，请检查 export_case_idx 范围。"
        )

    # ── 6. 返回结果 ────────────────────────────────────────
    return {
        "dice_per_class": mean_dice_per_class,
        "fg_dice_mean": fg_dice_mean,
        "WT": mean_wt,
        "TC": mean_tc,
        "ET": mean_et,
        "monitor_score": monitor_score,
    }


def _print_export_shapes(export_data: dict):
    """打印导出数据的 shape，方便 debug。"""
    print("\n  [导出数据 shape 检查]")
    shape_fields = [
        "image", "gt", "pred",
        "prob_ET", "prob_WT",
        "prob_c0", "prob_c1", "prob_c2", "prob_c3"
    ]
    for k in shape_fields:
        if k in export_data:
            v = export_data[k]
            shape = v.shape if hasattr(v, "shape") else str(v)
            print(f"    {k:12s}: {shape}")


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 用法：
    #   python brats/evaluate.py
    #   python brats/evaluate.py 3
    import sys as _sys
    export_idx = int(_sys.argv[1]) if len(_sys.argv) > 1 else 0
    evaluate(export_case_idx=export_idx)