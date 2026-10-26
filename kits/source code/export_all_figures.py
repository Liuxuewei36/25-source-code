# =============================================================================
# export_all_figures.py — KiTS19 全量推理 + 每切片结果图导出（展示专用）
# =============================================================================
# 放置位置：建议与原有 kits/evaluate.py 同级
# 用途：
#   1) 用已经训练好的 checkpoint，对指定 split 做完整推理
#   2) 为每个切片导出 1 张四联图（CT / GT / Pred / Overlay）
#   3) 额外导出 slice_metrics.csv，方便按 Dice 排序挑图
#
# 推荐：
#   - 做 PPT 展示时，优先导出 all + only_fg（只保留含前景切片）
#   - 然后按 top_mean.csv / best_per_case.csv 选图
# =============================================================================

import os
import sys
import csv
import argparse
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from kits import config
from kits.dataset import KiTS25DDataset, get_case_list
from kits.model import UNet2D
from kits.utils import ensure_dirs, load_checkpoint


LABEL_COLORS = np.array([
    [0,   0,   0  ],
    [30,  144, 255],
    [255, 69,  0  ],
], dtype=np.uint8)


def label_to_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c, color in enumerate(LABEL_COLORS):
        rgb[mask == c] = color
    return rgb


def dice_of_class(pred: np.ndarray, target: np.ndarray, class_id: int, smooth: float = 1e-6) -> float:
    pred_c = (pred == class_id)
    target_c = (target == class_id)
    inter = np.logical_and(pred_c, target_c).sum()
    union = pred_c.sum() + target_c.sum()
    return float((2.0 * inter + smooth) / (union + smooth))



def save_result_figure(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pr_mask: np.ndarray,
    save_path: str,
    title: str = "",
) -> None:
    gt_rgb = label_to_rgb(gt_mask)
    pred_rgb = label_to_rgb(pr_mask)

    h, w = pr_mask.shape
    overlay_rgba = np.zeros((h, w, 4), dtype=np.float32)
    overlay_rgba[..., :3] = pred_rgb.astype(np.float32) / 255.0
    overlay_rgba[..., 3] = (pr_mask > 0).astype(np.float32) * 0.45

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0))
    fig.suptitle(title, fontsize=11)

    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("CT Image")

    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth")

    axes[2].imshow(pred_rgb)
    axes[2].set_title("Prediction")

    axes[3].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[3].imshow(overlay_rgba)
    if (gt_mask == 1).any():
        axes[3].contour((gt_mask == 1).astype(np.uint8), levels=[0.5], colors=["deepskyblue"], linewidths=0.9)
    if (gt_mask == 2).any():
        axes[3].contour((gt_mask == 2).astype(np.uint8), levels=[0.5], colors=["gold"], linewidths=0.9)
    axes[3].set_title("Overlay (pred fill + GT contour)")

    legend_items = [
        mpatches.Patch(color=LABEL_COLORS[1] / 255.0, label="Kidney"),
        mpatches.Patch(color=LABEL_COLORS[2] / 255.0, label="Tumor"),
    ]
    axes[3].legend(handles=legend_items, loc="lower right", fontsize=8, framealpha=0.85)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)



def resolve_case_list(split: str) -> list[str]:
    all_cases = get_case_list(config.DATA_ROOT)
    if split == "all":
        return all_cases
    if split == "val":
        _, val_cases = train_test_split(
            all_cases,
            test_size=config.VAL_RATIO,
            random_state=config.SEED,
            shuffle=True,
        )
        return val_cases
    raise ValueError(f"unknown split: {split}")



def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def main(args):
    ensure_dirs(config.CHECKPOINT_DIR, config.FIGURE_DIR, config.LOG_DIR)

    device = config.DEVICE
    case_list = resolve_case_list(args.split)
    dataset = KiTS25DDataset(case_list, bg_keep_ratio=1.0, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"[export] split={args.split} | cases={len(case_list)} | slices={len(dataset)}")

    model = UNet2D(
        in_ch=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        base_ch=config.BASE_CHANNELS,
    ).to(device)

    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    elif not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, ckpt_path)
    load_checkpoint(ckpt_path, model, device=device)
    model.eval()

    save_root = os.path.join(config.FIGURE_DIR, args.save_dir)
    ensure_dirs(save_root)

    total_intersection = np.zeros(config.NUM_CLASSES, dtype=np.float64)
    total_union = np.zeros(config.NUM_CLASSES, dtype=np.float64)
    rows = []

    global_idx = 0
    saved_count = 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="[export] Inference"):
            x_dev = x.to(device, non_blocking=True)
            logits = model(x_dev)
            preds = logits.argmax(dim=1).cpu().numpy()
            ys = y.numpy()
            xs = x[:, 1].numpy()  # 中心切片

            for i in range(len(preds)):
                pred = preds[i].astype(np.int32)
                target = ys[i].astype(np.int32)
                image = xs[i].astype(np.float32)

                case_dir, s, _ = dataset.index_table[global_idx]
                case_name = os.path.basename(case_dir)
                label = f"{case_name}_s{s:03d}"

                kidney_dice = dice_of_class(pred, target, 1)
                tumor_dice = dice_of_class(pred, target, 2)
                mean_dice = 0.5 * (kidney_dice + tumor_dice)

                for c in range(config.NUM_CLASSES):
                    pred_c = (pred == c)
                    target_c = (target == c)
                    total_intersection[c] += np.logical_and(pred_c, target_c).sum()
                    total_union[c] += pred_c.sum() + target_c.sum()

                gt_has_fg = bool(target.max() > 0)
                gt_has_tumor = bool((target == 2).any())
                pred_has_fg = bool(pred.max() > 0)

                should_save = True
                if args.only_fg and not (gt_has_fg or pred_has_fg):
                    should_save = False

                fig_relpath = os.path.join(case_name, f"{label}.png")
                fig_abspath = os.path.join(save_root, fig_relpath)

                if should_save:
                    ensure_dirs(os.path.dirname(fig_abspath))
                    save_result_figure(
                        image=image,
                        gt_mask=target,
                        pr_mask=pred,
                        save_path=fig_abspath,
                        title=(
                            f"{label} | Kidney Dice={kidney_dice:.3f} | "
                            f"Tumor Dice={tumor_dice:.3f} | Mean={mean_dice:.3f}"
                        ),
                    )
                    saved_count += 1
                else:
                    fig_relpath = ""

                rows.append({
                    "case_name": case_name,
                    "slice_idx": s,
                    "label": label,
                    "kidney_dice": f"{kidney_dice:.6f}",
                    "tumor_dice": f"{tumor_dice:.6f}",
                    "mean_dice": f"{mean_dice:.6f}",
                    "gt_has_fg": int(gt_has_fg),
                    "gt_has_tumor": int(gt_has_tumor),
                    "pred_has_fg": int(pred_has_fg),
                    "fig_path": fig_relpath,
                })

                global_idx += 1

    dice = (2.0 * total_intersection + 1e-6) / (total_union + 1e-6)
    dice_bg = float(dice[0])
    dice_kidney = float(dice[1])
    dice_tumor = float(dice[2])
    dice_mean = 0.5 * (dice_kidney + dice_tumor)

    rows_sorted = sorted(rows, key=lambda r: float(r[args.rank_by]), reverse=True)
    best_per_case = []
    seen = set()
    for r in rows_sorted:
        if r["case_name"] in seen:
            continue
        seen.add(r["case_name"])
        best_per_case.append(r)

    all_csv = os.path.join(save_root, "slice_metrics.csv")
    top_csv = os.path.join(save_root, f"top_{args.topk}_by_{args.rank_by}.csv")
    best_case_csv = os.path.join(save_root, "best_per_case.csv")
    summary_txt = os.path.join(save_root, "summary.txt")

    fieldnames = [
        "case_name", "slice_idx", "label",
        "kidney_dice", "tumor_dice", "mean_dice",
        "gt_has_fg", "gt_has_tumor", "pred_has_fg", "fig_path",
    ]
    write_csv(all_csv, rows_sorted, fieldnames)
    write_csv(top_csv, rows_sorted[:args.topk], fieldnames)
    write_csv(best_case_csv, best_per_case, fieldnames)

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"split={args.split}\n")
        f.write(f"num_cases={len(case_list)}\n")
        f.write(f"num_slices={len(dataset)}\n")
        f.write(f"saved_figures={saved_count}\n")
        f.write(f"dice_bg={dice_bg:.6f}\n")
        f.write(f"dice_kidney={dice_kidney:.6f}\n")
        f.write(f"dice_tumor={dice_tumor:.6f}\n")
        f.write(f"dice_mean={dice_mean:.6f}\n")
        f.write(f"rank_by={args.rank_by}\n")
        f.write(f"only_fg={int(args.only_fg)}\n")

    print("\n" + "=" * 72)
    print("[export] 完成")
    print(f"[export] save_root      : {save_root}")
    print(f"[export] saved_figures  : {saved_count}")
    print(f"[export] slice_metrics  : {all_csv}")
    print(f"[export] top list       : {top_csv}")
    print(f"[export] best per case  : {best_case_csv}")
    print(f"[export] Kidney Dice    : {dice_kidney:.4f}")
    print(f"[export] Tumor  Dice    : {dice_tumor:.4f}")
    print(f"[export] Mean   Dice    : {dice_mean:.4f}")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint path or checkpoint filename under CHECKPOINT_DIR")
    parser.add_argument("--split", type=str, default="all", choices=["all", "val"], help="run on all labeled cases or only val split")
    parser.add_argument("--save_dir", type=str, default="export_all_fg", help="subdir under FIGURE_DIR")
    parser.add_argument("--rank_by", type=str, default="mean_dice", choices=["mean_dice", "tumor_dice", "kidney_dice"], help="which metric to use for ranking CSV")
    parser.add_argument("--topk", type=int, default=100, help="how many rows to keep in top_k csv")
    parser.add_argument("--batch_size", type=int, default=None, help="override config.BATCH_SIZE at export time")
    parser.add_argument("--include_bg", action="store_true", help="save background-only slices too")
    args = parser.parse_args()
    args.only_fg = not args.include_bg
    main(args)
