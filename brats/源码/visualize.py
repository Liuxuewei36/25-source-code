# brats/visualize.py
# ─────────────────────────────────────────────────────────────
# 功能：读取 evaluate.py 导出的 eval_data.pt，生成所有可展示图片。
#
# ★ 不在这里重跑模型，不在这里做推理，只做可视化。
# ★ 不使用 Grad-CAM。热力图 = softmax 概率图，直接来自模型输出。
# ★ 不依赖 seaborn / scikit-image，只用 matplotlib + numpy + torch。
#
# 生成的图片（全部保存到 outputs_brats/figures/）：
#   (1) four_modalities.png
#   (2) seg_visualization.png
#   (3) probability_heatmap_ET.png
#   (4) probability_heatmap_WT.png
#   (5) training_curves.png
#
# 输入数据 shape 约定（来自 eval_data.pt）：
#   image          : [4, D, H, W]  float32，通道顺序 T1/T1ce/T2/FLAIR
#   gt             : [D, H, W]     int64，值 ∈ {0,1,2,3}
#   pred           : [D, H, W]     int64，值 ∈ {0,1,2,3}
#   prob_ET        : [D, H, W]     float32，ET softmax 概率，值 ∈ [0,1]
#   prob_WT        : [D, H, W]     float32，WT 概率 = 1-p0，值 ∈ [0,1]
#   best_slice_idx : int           推荐展示的 axial slice
# ─────────────────────────────────────────────────────────────

import os
import sys
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")  # 无显示器环境（服务器/AutoDL）必须设置
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brats.config import cfg


# ══════════════════════════════════════════════════════════════
# 全局颜色配置
# ══════════════════════════════════════════════════════════════

# BraTS 4 类的颜色（用于 mask 可视化）
# 0=背景(黑), 1=坏死(红), 2=水肿(绿), 3=ET增强(蓝)
CLASS_COLORS = np.array(
    [
        [0, 0, 0],        # class 0: background
        [255, 0, 0],      # class 1: NCR/NET
        [0, 255, 0],      # class 2: ED
        [0, 0, 255],      # class 3: ET
    ],
    dtype=np.uint8,
)

FIG_DPI = 150  # 图片分辨率，适合 PPT 使用


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════

def load_eval_data(eval_data_path: str) -> dict:
    """
    加载 evaluate.py 导出的 eval_data.pt。
    返回字典，所有 tensor 保持原始 dtype。
    """
    if not os.path.exists(eval_data_path):
        raise FileNotFoundError(
            f"[Visualize] eval_data.pt not found: {eval_data_path}\n"
            "Please run: python brats/evaluate.py first"
        )
    data = torch.load(eval_data_path, map_location="cpu")
    print(f"[Visualize] Loaded eval_data.pt with keys: {list(data.keys())}")
    return data


def get_best_slice(data: dict) -> int:
    """
    获取推荐展示的 axial slice index。
    优先用 eval_data.pt 里保存的 best_slice_idx；
    如果没有，自己计算（GT WT 面积最大的 slice）。
    """
    if "best_slice_idx" in data:
        idx = int(data["best_slice_idx"])
        print(f"[Visualize] Using exported best_slice_idx: {idx}")
        return idx

    print("[Visualize] best_slice_idx not found, computing automatically...")
    gt = data["gt"]  # [D, H, W]
    gt_wt = (gt >= 1).float()
    areas = gt_wt.sum(dim=(1, 2))  # [D]

    if areas.max() > 0:
        idx = int(areas.argmax().item())
    else:
        pred = data["pred"]
        pred_wt = (pred >= 1).float()
        areas_p = pred_wt.sum(dim=(1, 2))
        if areas_p.max() > 0:
            idx = int(areas_p.argmax().item())
        else:
            idx = gt.shape[0] // 2

    print(f"[Visualize] Computed best slice index: {idx}")
    return idx


def normalize_for_display(arr: np.ndarray) -> np.ndarray:
    """
    把任意范围的 float array 归一化到 [0,1]，用于 imshow 显示。
    如果 arr 全为常数，返回全零。
    """
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """
    把整数 mask（[H, W]，值 ∈ {0,1,2,3}）转成 RGB 图像（[H, W, 3]）。
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(len(CLASS_COLORS)):
        rgb[mask == c] = CLASS_COLORS[c]
    return rgb


def make_overlay(
    image_gray: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    把彩色 mask 半透明叠加到灰度图上。
    """
    #我先把离散标签 mask 映射成 RGB 颜色，再按一定透明度叠加到灰度 MRI 上。这样既能看到解剖背景，又能看到预测区域的位置和范围，比单独显示 mask 更直观。
    bg = np.stack([image_gray] * 3, axis=-1)  # [H, W, 3]
    mask_rgb = mask_to_rgb(mask).astype(np.float32) / 255.0

    fg_mask = (mask > 0)[:, :, np.newaxis]
    overlay = bg.copy()
    overlay[fg_mask[:, :, 0]] = (
        (1 - alpha) * bg[fg_mask[:, :, 0]]
        + alpha * mask_rgb[fg_mask[:, :, 0]]
    )
    return overlay.astype(np.float32)


def make_heatmap_overlay(
    image_gray: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    cmap: str = "jet",
) -> np.ndarray:
    """
    把概率热力图半透明叠加到灰度图上。
    """
    cmap_fn = plt.get_cmap(cmap)
    heatmap_rgb = cmap_fn(heatmap)[:, :, :3]  # 去掉 alpha 通道
    bg = np.stack([image_gray] * 3, axis=-1)
    overlay = (1 - alpha) * bg + alpha * heatmap_rgb
    return overlay.clip(0, 1).astype(np.float32)


# ══════════════════════════════════════════════════════════════
# 图1：四模态并排图
# ══════════════════════════════════════════════════════════════

def plot_four_modalities(data: dict, slice_idx: int, save_path: str):
    """
    生成同一 axial slice 的 T1/T1ce/T2/FLAIR 四图并排图。
    """
    image = data["image"]  # [4, D, H, W]
    modality_names = ["T1", "T1ce", "T2", "FLAIR"]

    D = image.shape[1]
    slice_idx = max(0, min(slice_idx, D - 1))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=FIG_DPI)
    fig.suptitle(
        f"BraTS2019 - Four Modalities (Axial Slice {slice_idx}/{D})",
        fontsize=13,
        y=1.02,
    )

    for i, (ax, name) in enumerate(zip(axes, modality_names)):
        sl = image[i, slice_idx, :, :].numpy()
        sl = normalize_for_display(sl)

        ax.imshow(sl, cmap="gray", interpolation="nearest")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=FIG_DPI)
    plt.close()
    print(f"[Visualize] Saved: {save_path}")


# ══════════════════════════════════════════════════════════════
# 图2：分割可视化（原图/GT/Pred/Overlay）
# ══════════════════════════════════════════════════════════════
#这里选择 FLAIR 作为底图，是因为它对水肿和病灶范围通常更敏感，视觉上更容易观察肿瘤区域。
def plot_seg_visualization(data: dict, slice_idx: int, save_path: str):
    """
    生成 4 格分割对比图：
        1. FLAIR input
        2. Ground truth
        3. Prediction
        4. Overlay
    """
    image = data["image"]   # [4, D, H, W]
    gt = data["gt"]         # [D, H, W]
    pred = data["pred"]     # [D, H, W]

    D = image.shape[1]
    slice_idx = max(0, min(slice_idx, D - 1))

    flair_sl = image[3, slice_idx, :, :].numpy()
    flair_sl = normalize_for_display(flair_sl)

    gt_sl = gt[slice_idx, :, :].numpy().astype(np.int32)
    pred_sl = pred[slice_idx, :, :].numpy().astype(np.int32)

    gt_rgb = mask_to_rgb(gt_sl)
    pred_rgb = mask_to_rgb(pred_sl)
    overlay = make_overlay(flair_sl, pred_sl, alpha=0.4)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=FIG_DPI)

    case_id = os.path.basename(str(data.get("case_id", "unknown")))
    fig.suptitle(
        f"Segmentation Result - {case_id} (Axial Slice {slice_idx})",
        fontsize=12,
        y=1.02,
    )

    axes[0].imshow(flair_sl, cmap="gray", interpolation="nearest")
    axes[0].set_title("FLAIR Input", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(gt_rgb, interpolation="nearest")
    axes[1].set_title("Ground Truth", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(pred_rgb, interpolation="nearest")
    axes[2].set_title("Prediction", fontsize=11)
    axes[2].axis("off")

    axes[3].imshow(overlay, interpolation="nearest")
    axes[3].set_title("Overlay (Pred on FLAIR)", fontsize=11)
    axes[3].axis("off")

    legend_patches = [
        mpatches.Patch(color=np.array(CLASS_COLORS[1]) / 255.0, label="NCR/NET"),
        mpatches.Patch(color=np.array(CLASS_COLORS[2]) / 255.0, label="ED"),
        mpatches.Patch(color=np.array(CLASS_COLORS[3]) / 255.0, label="ET"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.05),
    )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=FIG_DPI)
    plt.close()
    print(f"[Visualize] Saved: {save_path}")


# ══════════════════════════════════════════════════════════════
# 图3/4：概率热力图（ET 和 WT）
# ══════════════════════════════════════════════════════════════

def plot_probability_heatmap(
    data: dict,
    prob_key: str,   # "prob_ET" 或 "prob_WT"
    label: str,      # "ET (Enhancing Tumor)" / "WT (Whole Tumor)"
    slice_idx: int,
    save_path: str,
):
    """
    生成 3 格概率热力图：
        1. FLAIR input
        2. Probability map
        3. Overlay
    """
    #这个热力图表示模型对某个体素属于 ET 或 WT 的概率置信度。它直接来自 softmax 输出，不是 Grad-CAM，因此更接近模型最终预测本身。
    image = data["image"]       # [4, D, H, W]
    prob_map = data[prob_key]   # [D, H, W]

    D = image.shape[1]
    slice_idx = max(0, min(slice_idx, D - 1))

    flair_sl = image[3, slice_idx, :, :].numpy()
    flair_sl = normalize_for_display(flair_sl)

    prob_sl = prob_map[slice_idx, :, :].numpy()

    max_prob = prob_sl.max()
    if max_prob < 1e-4:
        print(
            f"[Visualize WARNING] {prob_key} is almost zero on slice {slice_idx} "
            f"(max={max_prob:.6f}). Heatmap may look very dark."
        )

    heatmap_overlay = make_heatmap_overlay(flair_sl, prob_sl, alpha=0.5, cmap="jet")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=FIG_DPI)

    case_id = os.path.basename(str(data.get("case_id", "unknown")))
    fig.suptitle(
        f"{label} Probability Heatmap - {case_id} (Slice {slice_idx})\n"
        f"[Softmax Probability Map, not Grad-CAM]",
        fontsize=11,
        y=1.04,
    )

    axes[0].imshow(flair_sl, cmap="gray", interpolation="nearest")
    axes[0].set_title("FLAIR Input", fontsize=11)
    axes[0].axis("off")

    im = axes[1].imshow(
        prob_sl,
        cmap="jet",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[1].set_title(
        f"{label} Probability Map\n(Softmax Output, not Grad-CAM)",
        fontsize=10,
    )
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(heatmap_overlay, interpolation="nearest")
    axes[2].set_title(f"Overlay ({label} on FLAIR)", fontsize=11)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=FIG_DPI)
    plt.close()
    print(f"[Visualize] Saved: {save_path}")


# ══════════════════════════════════════════════════════════════
# 图5：训练曲线
# ══════════════════════════════════════════════════════════════

def plot_training_curves(csv_path: str, save_path: str):
    """
    从 train_log.csv 读取训练日志，绘制：
        - Train Loss
        - Val Loss
        - mean(WT,TC,ET)
        - WT / TC / ET curves
    """
    if not os.path.exists(csv_path):
        print(f"[Visualize WARNING] Training log not found: {csv_path}. Skip.")
        return

    df = pd.read_csv(csv_path)

    if len(df) == 0:
        print("[Visualize WARNING] train_log.csv is empty. Skip.")
        return

    if all(c in df.columns for c in ["val_WT", "val_TC", "val_ET"]):
        df["monitor_score"] = (df["val_WT"] + df["val_TC"] + df["val_ET"]) / 3.0
    else:
        df["monitor_score"] = None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=FIG_DPI)
    fig.suptitle("Training Curves of 3D U-Net on BraTS2019", fontsize=15)

    epochs = df["epoch"]

    ax1 = axes[0]
    if "train_loss" in df.columns:
        ax1.plot(epochs, df["train_loss"], color="steelblue", linewidth=1.5, label="Train Loss")
    if "val_loss" in df.columns:
        ax1.plot(epochs, df["val_loss"], color="darkorange", linewidth=1.5, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss Curves")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    plotted = False

    if "monitor_score" in df.columns and df["monitor_score"].notna().any():
        ax2.plot(
            epochs,
            df["monitor_score"],
            color="seagreen",
            linewidth=1.5,
            linestyle="--",
            label="mean(WT,TC,ET)",
        )
        plotted = True

    if "val_WT" in df.columns:
        ax2.plot(epochs, df["val_WT"], color="tomato", linewidth=1.0, linestyle=":", label="WT Dice")
        plotted = True
    if "val_TC" in df.columns:
        ax2.plot(epochs, df["val_TC"], color="mediumpurple", linewidth=1.0, linestyle=":", label="TC Dice")
        plotted = True
    if "val_ET" in df.columns:
        ax2.plot(epochs, df["val_ET"], color="goldenrod", linewidth=1.0, linestyle=":", label="ET Dice")
        plotted = True

    if not plotted and "val_dice_mean" in df.columns:
        ax2.plot(epochs, df["val_dice_mean"], color="seagreen", linewidth=1.5, label="Val Dice Mean")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice")
    ax2.set_title("Validation Dice Curves (WT / TC / ET)")
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=FIG_DPI)
    plt.close()
    print(f"[Visualize] Saved: {save_path}")


# ══════════════════════════════════════════════════════════════
# 主函数：生成所有图片
# ══════════════════════════════════════════════════════════════

def run_all():
    """
    一键生成所有可展示图片。
    """
    fig_dir = os.path.join(cfg["output_dir"], "figures")
    log_dir = os.path.join(cfg["output_dir"], "logs")
    os.makedirs(fig_dir, exist_ok=True)

    eval_data_path = os.path.join(fig_dir, "eval_data.pt")
    csv_path = os.path.join(log_dir, "train_log.csv")

    print("=" * 60)
    print("[Visualize] Generating all figures...")
    print(f"[Visualize] eval_data path: {eval_data_path}")
    print(f"[Visualize] output dir: {fig_dir}")
    print("=" * 60)

    data = load_eval_data(eval_data_path)

    slice_idx = get_best_slice(data)
    print(f"[Visualize] Axial slice index: {slice_idx}")

    plot_four_modalities(
        data=data,
        slice_idx=slice_idx,
        save_path=os.path.join(fig_dir, "four_modalities.png"),
    )

    plot_seg_visualization(
        data=data,
        slice_idx=slice_idx,
        save_path=os.path.join(fig_dir, "seg_visualization.png"),
    )

    if "prob_ET" in data:
        plot_probability_heatmap(
            data=data,
            prob_key="prob_ET",
            label="ET (Enhancing Tumor)",
            slice_idx=slice_idx,
            save_path=os.path.join(fig_dir, "probability_heatmap_ET.png"),
        )
    else:
        print("[Visualize WARNING] prob_ET not found in eval_data.pt. Skip ET heatmap.")

    if "prob_WT" in data:
        plot_probability_heatmap(
            data=data,
            prob_key="prob_WT",
            label="WT (Whole Tumor)",
            slice_idx=slice_idx,
            save_path=os.path.join(fig_dir, "probability_heatmap_WT.png"),
        )
    else:
        print("[Visualize WARNING] prob_WT not found in eval_data.pt. Skip WT heatmap.")

    plot_training_curves(
        csv_path=csv_path,
        save_path=os.path.join(fig_dir, "training_curves.png"),
    )

    print("\n" + "=" * 60)
    print("[Visualize] Done. Generated figures:")
    for fname in [
        "four_modalities.png",
        "seg_visualization.png",
        "probability_heatmap_ET.png",
        "probability_heatmap_WT.png",
        "training_curves.png",
    ]:
        fpath = os.path.join(fig_dir, fname)
        status = "OK" if os.path.exists(fpath) else "MISSING"
        print(f"  [{status}] {fpath}")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_all()