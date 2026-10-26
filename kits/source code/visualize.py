# =============================================================================
# visualize.py — KiTS19 2.5D Baseline 可视化
# =============================================================================
# 读取已有文件，生成所有可展示图片，不重跑模型推理。
#
# 读取：
#   - outputs_kits/logs/train_log_clean.csv（若存在，优先）
#   - outputs_kits/logs/train_log.csv
#   - eval_results.pt（支持多个候选路径）
#   - config.DATA_ROOT（少量切片）→ 2.5D 输入示意图
#
# 输出（全部保存到 outputs_kits/figures/）：
#   - training_curve.png
#   - input_2_5d_demo.png
#   - seg_result_gallery.png
#   - hard_case_gallery.png
# =============================================================================

import os
import sys
import csv
import math
import random

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from kits import config
from kits.utils import ensure_dirs


# -----------------------------------------------------------------------------
# 全局绘图风格（PPT / 汇报友好）
# -----------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "lines.linewidth": 2.0,
})


# -----------------------------------------------------------------------------
# 分割类别颜色
# -----------------------------------------------------------------------------

LABEL_COLORS = np.array([
    [0,   0,   0  ],   # 0: background
    [30,  144, 255],   # 1: kidney
    [255, 69,  0  ],   # 2: tumor
], dtype=np.uint8)


def label_to_rgb(mask: np.ndarray) -> np.ndarray:
    """int mask [H, W] -> RGB [H, W, 3]"""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c, color in enumerate(LABEL_COLORS):
        rgb[mask == c] = color
    return rgb


# =============================================================================
# 路径辅助函数
# =============================================================================

def resolve_train_log_path() -> str:
    """
    优先使用干净版日志 train_log_clean.csv；
    若不存在，则退回 train_log.csv。
    """
    clean_path = os.path.join(config.LOG_DIR, "train_log_clean.csv")
    raw_path   = os.path.join(config.LOG_DIR, "train_log.csv")

    if os.path.exists(clean_path):
        print(f"[visualize] 使用 clean log: {clean_path}")
        return clean_path
    print(f"[visualize] 使用 raw log: {raw_path}")
    return raw_path


def resolve_eval_results_path() -> str:
    """
    兼容多种 eval_results.pt 保存位置。
    """
    candidates = []

    if hasattr(config, "EVAL_RESULTS_PATH"):
        candidates.append(config.EVAL_RESULTS_PATH)

    if hasattr(config, "LOG_DIR"):
        candidates.append(os.path.join(config.LOG_DIR, "eval_results.pt"))

    if hasattr(config, "OUTPUT_ROOT"):
        candidates.append(os.path.join(config.OUTPUT_ROOT, "eval_results.pt"))

    # 去重并返回第一个存在的
    seen = set()
    dedup = []
    for p in candidates:
        if p not in seen:
            dedup.append(p)
            seen.add(p)

    for p in dedup:
        if os.path.exists(p):
            return p

    # 如果都不存在，返回第一个候选，便于报错信息更明确
    return dedup[0] if len(dedup) > 0 else "eval_results.pt"


# =============================================================================
# 训练曲线
# =============================================================================

def load_and_clean_training_log(csv_path: str) -> dict:
    """
    读取训练日志，并自动处理：
      1. 重复 epoch
      2. NaN 行
      3. 混入旧实验的情况

    清洗策略：
      - 按 epoch 分组
      - 优先保留 val_dice_mean 非 NaN 的记录
      - 若某个 epoch 有多条记录，保留 val_dice_mean 最大的一条
      - 若全是 NaN，则保留最后一条

    返回：
      {
        "epoch": [...],
        "train_loss": [...],
        "val_loss": [...],
        "val_dice_kidney": [...],
        "val_dice_tumor": [...],
        "val_dice_mean": [...],
      }
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"training log not found: {csv_path}")

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if len(rows) == 0:
        raise RuntimeError("training log is empty")

    # 按 epoch 分组
    grouped = {}
    for r in rows:
        ep = int(r["epoch"])
        grouped.setdefault(ep, []).append(r)

    cleaned_rows = []
    for ep in sorted(grouped.keys()):
        candidates = grouped[ep]

        valid = []
        for r in candidates:
            try:
                v = float(r["val_dice_mean"])
                if not math.isnan(v):
                    valid.append((v, r))
            except Exception:
                pass

        if len(valid) > 0:
            chosen = max(valid, key=lambda x: x[0])[1]
        else:
            chosen = candidates[-1]

        cleaned_rows.append(chosen)

    data = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_dice_kidney": [],
        "val_dice_tumor": [],
        "val_dice_mean": [],
    }

    for r in cleaned_rows:
        data["epoch"].append(int(r["epoch"]))
        data["train_loss"].append(float(r["train_loss"]))
        data["val_loss"].append(float(r["val_loss"]) if r["val_loss"].lower() != "nan" else np.nan)
        data["val_dice_kidney"].append(float(r["val_dice_kidney"]))
        data["val_dice_tumor"].append(float(r["val_dice_tumor"]))
        data["val_dice_mean"].append(float(r["val_dice_mean"]))

    print(f"[visualize] 训练日志清洗完成: {len(rows)} -> {len(cleaned_rows)} rows")
    return data


def plot_training_curve(csv_path: str, save_path: str):
    """
    绘制 1x2 训练曲线图：
      左图：Training and Validation Loss Curves
      右图：Validation Dice Curves (Kidney / Tumor / Mean)
    """
    if not os.path.exists(csv_path):
        print(f"[visualize] ⚠️ 找不到训练日志: {csv_path}，跳过训练曲线。")
        return

    try:
        data = load_and_clean_training_log(csv_path)
    except Exception as e:
        print(f"[visualize] ⚠️ 读取训练日志失败: {e}")
        return

    epochs       = np.array(data["epoch"])
    train_loss   = np.array(data["train_loss"], dtype=float)
    val_loss     = np.array(data["val_loss"], dtype=float)
    kidney_dice  = np.array(data["val_dice_kidney"], dtype=float)
    tumor_dice   = np.array(data["val_dice_tumor"], dtype=float)
    mean_dice    = np.array(data["val_dice_mean"], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # -----------------------------------------------------------------
    # 左图：Loss
    # -----------------------------------------------------------------
    ax1 = axes[0]
    ax1.plot(epochs, train_loss, color="#2878B5", linestyle="-",  label="Train Loss")
    ax1.plot(epochs, val_loss,   color="#FF7F0E", linestyle="--", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss Curves")
    ax1.set_ylim(bottom=0)
    ax1.legend(loc="upper right", framealpha=0.85)

    # -----------------------------------------------------------------
    # 右图：Validation Dice
    # -----------------------------------------------------------------
    ax2 = axes[1]
    ax2.plot(epochs, kidney_dice, color="#4C72B0", linestyle="--", label="Kidney Dice")
    ax2.plot(epochs, tumor_dice,  color="#DD8452", linestyle=":",  label="Tumor Dice")
    ax2.plot(epochs, mean_dice,   color="#2CA02C", linestyle="-.", label="Mean Dice")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice")
    ax2.set_ylim(0, 1)
    ax2.set_title("Validation Dice Curves (Kidney / Tumor / Mean)")
    ax2.legend(loc="lower right", framealpha=0.85)

    # 标注最佳 mean dice
    best_idx   = int(np.nanargmax(mean_dice))
    best_epoch = int(epochs[best_idx])
    best_dice  = float(mean_dice[best_idx])

    ax2.axvline(x=best_epoch, color="#2CA02C", linestyle=":", alpha=0.6)
    ax2.annotate(
        f"Best Mean Dice = {best_dice:.3f}\nEpoch {best_epoch}",
        xy=(best_epoch, best_dice),
        xytext=(best_epoch + max(len(epochs) // 10, 1), max(best_dice - 0.12, 0.05)),
        fontsize=9,
        color="#2CA02C",
        arrowprops=dict(arrowstyle="->", color="#2CA02C", lw=1.2),
    )

    fig.suptitle(
        "Training Curves of 2.5D U-Net on KiTS19",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] 训练曲线已保存: {os.path.basename(save_path)}")


# =============================================================================
# 2.5D 输入示意图
# =============================================================================

def plot_2_5d_demo(save_path: str, data_root: str = config.DATA_ROOT, img_size: int = config.IMG_SIZE):
    """
    随机选一个含前景的真实 case/slice，展示：
      [s-1] [s] [s+1] [center GT]
    """
    def load_gray(path: str) -> np.ndarray:
        img = Image.open(path).convert("L")
        img = img.resize((img_size, img_size), resample=Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0

    def load_mask(path: str) -> np.ndarray:
        from kits.dataset import normalize_mask
        mask = Image.open(path).convert("L")
        mask = mask.resize((img_size, img_size), resample=Image.NEAREST)
        arr = np.array(mask, dtype=np.uint8)
        try:
            return normalize_mask(arr)
        except ValueError:
            return (arr / 85).astype(np.int32).clip(0, 2)

    found = False
    if os.path.exists(data_root):
        cases = sorted([
            d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d))
        ])
        random.shuffle(cases)

        for case_name in cases[:20]:
            case_dir = os.path.join(data_root, case_name)
            img_dir  = os.path.join(case_dir, "imaging")
            seg_dir  = os.path.join(case_dir, "segmentation")
            if not (os.path.isdir(img_dir) and os.path.isdir(seg_dir)):
                continue

            img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".png")])
            seg_files = sorted([f for f in os.listdir(seg_dir) if f.lower().endswith(".png")])

            if len(img_files) < 3 or len(img_files) != len(seg_files):
                continue

            for s in range(1, len(img_files) - 1):
                seg_arr = np.array(Image.open(os.path.join(seg_dir, seg_files[s])).convert("L"))
                if seg_arr.max() > 0:
                    s_prev = load_gray(os.path.join(img_dir, img_files[s - 1]))
                    s_curr = load_gray(os.path.join(img_dir, img_files[s]))
                    s_next = load_gray(os.path.join(img_dir, img_files[s + 1]))
                    gt     = load_mask(os.path.join(seg_dir, seg_files[s]))
                    demo_label = f"{case_name} | slice {s}"
                    found = True
                    break
            if found:
                break

    if not found:
        print("[visualize] ⚠️ 未找到真实数据，使用合成 demo 图。")
        h = img_size
        s_prev = np.random.rand(h, h).astype(np.float32) * 0.4 + 0.1
        s_curr = np.random.rand(h, h).astype(np.float32) * 0.4 + 0.2
        s_next = np.random.rand(h, h).astype(np.float32) * 0.4 + 0.1
        gt = np.zeros((h, h), dtype=np.int32)
        gt[h // 4:h * 3 // 4, h // 4:h * 3 // 4] = 1
        gt[h // 3:h * 2 // 3, h // 3:h * 2 // 3] = 2
        demo_label = "Synthetic Demo"

    gt_rgb = label_to_rgb(gt)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    border_ctx = "#64B5F6"
    border_center = "#FF8F00"

    panels = [
        (s_prev, "Slice s-1\n(context)",  border_ctx,    "gray"),
        (s_curr, "Slice s\n(center)",     border_center, "gray"),
        (s_next, "Slice s+1\n(context)",  border_ctx,    "gray"),
        (gt_rgb, "Center GT\n(label)",    "#66BB6A",     None),
    ]

    for ax, (img, title, border_color, cmap) in zip(axes, panels):
        if cmap is not None:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=11, pad=6)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3.5)

    fig.text(0.245, 0.5, "→", fontsize=22, ha="center", va="center", color="#999999")
    fig.text(0.48,  0.5, "→", fontsize=22, ha="center", va="center", color="#999999")

    legend_patches = [
        mpatches.Patch(color=LABEL_COLORS[1] / 255., label="Kidney (class 1)"),
        mpatches.Patch(color=LABEL_COLORS[2] / 255., label="Tumor (class 2)"),
    ]
    axes[3].legend(handles=legend_patches, loc="lower right", fontsize=9, framealpha=0.8)

    fig.suptitle(
        f"2.5D Input Construction — [s-1, s, s+1] → Predict the Center Slice\n"
        f"({demo_label})",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] 2.5D demo 已保存: {os.path.basename(save_path)}")


# =============================================================================
# Gallery 内部绘图工具
# =============================================================================

def _draw_quad(ax_img, ax_gt, ax_pred, ax_overlay, image, gt, pred, title=""):
    """
    在一行的 4 个子图上画：
      image / gt / pred / overlay
    """
    gt_rgb = label_to_rgb(gt)
    pred_rgb = label_to_rgb(pred)

    img_rgb = np.stack([image * 255] * 3, axis=-1).astype(np.uint8)
    overlay = img_rgb.copy()
    fg = pred > 0
    if fg.any():
        alpha = 0.5
        overlay[fg] = (
            (1 - alpha) * img_rgb[fg] + alpha * pred_rgb[fg]
        ).astype(np.uint8)

    ax_img.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax_gt.imshow(gt_rgb)
    ax_pred.imshow(pred_rgb)
    ax_overlay.imshow(overlay)

    for ax in [ax_img, ax_gt, ax_pred, ax_overlay]:
        ax.axis("off")

    if title:
        ax_img.set_ylabel(title, fontsize=8, labelpad=10, rotation=0, ha="right", va="center")


# =============================================================================
# 正常样例 gallery
# =============================================================================

def plot_seg_gallery(vis_records: list[dict], save_path: str, n_show: int = 4,
                     title: str = "Segmentation Results — 2.5D U-Net on KiTS19"):
    fg_records = [r for r in vis_records if r["gt"].max() > 0]
    if len(fg_records) == 0:
        fg_records = vis_records

    samples = random.sample(fg_records, min(n_show, len(fg_records)))
    n = len(samples)

    fig, axes = plt.subplots(n, 4, figsize=(15, 3.8 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, rec in enumerate(samples):
        slice_title = f"{rec['label']} | Tumor Dice={rec['tumor_dice']:.3f}"
        _draw_quad(
            axes[row, 0], axes[row, 1], axes[row, 2], axes[row, 3],
            image=rec["image"],
            gt=rec["gt"],
            pred=rec["pred"],
            title=slice_title,
        )

    for ax, col_title in zip(axes[0], ["CT Image", "Ground Truth", "Prediction", "Overlay"]):
        ax.set_title(col_title, fontsize=10, fontweight="bold", pad=5)

    legend_patches = [
        mpatches.Patch(color=LABEL_COLORS[1] / 255., label="Kidney"),
        mpatches.Patch(color=LABEL_COLORS[2] / 255., label="Tumor"),
    ]
    axes[-1, -1].legend(handles=legend_patches, loc="lower right", fontsize=9, framealpha=0.85)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.005)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] 结果 gallery 已保存: {os.path.basename(save_path)}")


# =============================================================================
# 困难样例 gallery
# =============================================================================

def plot_hard_case_gallery(vis_records: list[dict], save_path: str, n_show: int = 3):
    tumor_records = [r for r in vis_records if (r["gt"] == 2).any()]
    if len(tumor_records) == 0:
        print("[visualize] ⚠️ 没有含肿瘤的切片，跳过困难样例 gallery。")
        return

    hard_samples = sorted(tumor_records, key=lambda r: r["tumor_dice"])[:n_show]
    n = len(hard_samples)

    fig, axes = plt.subplots(n, 4, figsize=(15, 3.8 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, rec in enumerate(hard_samples):
        slice_title = f"[Hard] {rec['label']} | Tumor Dice={rec['tumor_dice']:.3f}"
        _draw_quad(
            axes[row, 0], axes[row, 1], axes[row, 2], axes[row, 3],
            image=rec["image"],
            gt=rec["gt"],
            pred=rec["pred"],
            title=slice_title,
        )

    for ax, col_title in zip(axes[0], ["CT Image", "Ground Truth", "Prediction", "Overlay"]):
        ax.set_title(col_title, fontsize=10, fontweight="bold", pad=5)

    legend_patches = [
        mpatches.Patch(color=LABEL_COLORS[1] / 255., label="Kidney"),
        mpatches.Patch(color=LABEL_COLORS[2] / 255., label="Tumor"),
    ]
    axes[-1, -1].legend(handles=legend_patches, loc="lower right", fontsize=9, framealpha=0.85)

    fig.suptitle(
        "Hard Cases — Low Tumor Dice (Boundary Ambiguity / Class Imbalance)",
        fontsize=12, fontweight="bold", y=1.005
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] 困难样例 gallery 已保存: {os.path.basename(save_path)}")


# =============================================================================
# 主入口
# =============================================================================

def main():
    random.seed(config.SEED)
    np.random.seed(config.SEED)

    ensure_dirs(config.FIGURE_DIR)

    # -------------------------------------------------------------------------
    # 1. 训练曲线
    # -------------------------------------------------------------------------
    train_log_path = resolve_train_log_path()
    plot_training_curve(
        csv_path=train_log_path,
        save_path=os.path.join(config.FIGURE_DIR, "training_curve.png"),
    )

    # -------------------------------------------------------------------------
    # 2. 2.5D 输入示意图
    # -------------------------------------------------------------------------
    plot_2_5d_demo(
        save_path=os.path.join(config.FIGURE_DIR, "input_2_5d_demo.png"),
    )

    # -------------------------------------------------------------------------
    # 3. 读取 eval_results.pt
    # -------------------------------------------------------------------------
    results_path = resolve_eval_results_path()
    if not os.path.exists(results_path):
        print(f"[visualize] ⚠️ 找不到 {results_path}，请先运行 evaluate.py。跳过结果图。")
        return

    print(f"[visualize] 读取评估结果: {results_path}")
    results = torch.load(results_path, map_location="cpu", weights_only=False)

    print(f"\n  Kidney Dice : {results['dice_kidney']:.4f}")
    print(f"  Tumor  Dice : {results['dice_tumor']:.4f}")
    print(f"  Mean   Dice : {results['dice_mean']:.4f}\n")

    vis_records = results.get("vis_records", [])
    if len(vis_records) == 0:
        print("[visualize] ⚠️ eval_results.pt 中没有 vis_records，跳过 gallery。")
        return

    # -------------------------------------------------------------------------
    # 4. 正常样例 gallery
    # -------------------------------------------------------------------------
    plot_seg_gallery(
        vis_records=vis_records,
        save_path=os.path.join(config.FIGURE_DIR, "seg_result_gallery.png"),
        n_show=4,
        title=(
            f"Segmentation Results — 2.5D U-Net on KiTS19\n"
            f"Kidney Dice={results['dice_kidney']:.3f}   "
            f"Tumor Dice={results['dice_tumor']:.3f}   "
            f"Mean Dice={results['dice_mean']:.3f}"
        ),
    )

    # -------------------------------------------------------------------------
    # 5. 困难样例 gallery
    # -------------------------------------------------------------------------
    plot_hard_case_gallery(
        vis_records=vis_records,
        save_path=os.path.join(config.FIGURE_DIR, "hard_case_gallery.png"),
        n_show=3,
    )

    # -------------------------------------------------------------------------
    # 6. 汇总
    # -------------------------------------------------------------------------
    print("\n[visualize] 全部输出文件：")
    figs = [
        "training_curve.png",
        "input_2_5d_demo.png",
        "seg_result_gallery.png",
        "hard_case_gallery.png",
    ]
    for f in figs:
        full = os.path.join(config.FIGURE_DIR, f)
        status = "✅" if os.path.exists(full) else "❌ 未生成"
        print(f"  {status}  {full}")

    print("\n[visualize] 完成。")


if __name__ == "__main__":
    main()