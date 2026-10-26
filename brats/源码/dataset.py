# brats/dataset.py
# ─────────────────────────────────────────────────────────────
# 作用：读取 BraTS2019 NIfTI 文件，完成预处理，随机裁 3D patch，
#       返回可以直接喂给模型的 tensor。
#
# ★ 全项目轴顺序约定（必须统一，否则 shape 对不上）：
#   nibabel 读出来：numpy array，shape = (H, W, D) = (240, 240, 155)
#   经过 np.transpose 后统一变成：(D, H, W) = (155, 240, 240)
#   最终返回 tensor：
#       x : [4, D, H, W]  —— 通道顺序：0=T1 / 1=T1ce / 2=T2 / 3=FLAIR
#       y : [D, H, W]     —— 分割标签，long 型，值 ∈ {0,1,2,3}
#
# ★ 标签映射：
#   原始 BraTS seg 值：0(背景) / 1(坏死+非增强) / 2(水肿) / 4(增强肿瘤)
#   映射后：         0        / 1                / 2       / 3
#   → 只需把 4 → 3，其余不变
# ─────────────────────────────────────────────────────────────

import os
import glob
import random

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader

# 从同级 config 导入（如果作为独立脚本运行，需要先把项目根目录加入 sys.path）
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brats.config import cfg


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════

def load_nii(path: str) -> np.ndarray:
    """
    用 nibabel 加载单个 .nii / .nii.gz 文件。
    返回 float32 numpy array，shape = (H, W, D)。
    注意：nibabel 默认返回的轴顺序就是 (H, W, D)，不是 (D, H, W)。
    统一在后续 transpose 成 (D, H, W)。
    """
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)   # 直接读成 float32，省一步转换
    return data                              # shape: (H, W, D)


def zscore_normalize(volume: np.ndarray) -> np.ndarray:
    """
    只在非零脑区（前景 mask）做 z-score 归一化。
    背景（值为 0 的体素）保持为 0，不参与均值/方差计算。

    原因：BraTS MRI 背景是 0，如果把背景也纳入统计，
          均值会被大量 0 拉低，方差被撑大，影响归一化质量。

    参数：
        volume: shape (D, H, W)，单模态 MRI 数据（已 transpose）
    返回：
        同 shape，归一化后的 float32 array
    """
    # 前景 mask：非零体素
    mask = volume > 0                        # bool array，shape (D, H, W)

    # 防御：如果整个 volume 都是 0（异常 case），直接返回原值
    if mask.sum() == 0:
        return volume

    mean = volume[mask].mean()
    std  = volume[mask].std()

    # 防御：std 极小时避免除零
    if std < 1e-8:
        std = 1e-8

    # 归一化：前景区域做 z-score，背景保持 0
    normalized = volume.copy()
    normalized[mask] = (volume[mask] - mean) / std
    return normalized


def remap_label(seg: np.ndarray) -> np.ndarray:
    """
    把 BraTS 原始标签 {0,1,2,4} 映射到 {0,1,2,3}。
    只需把 4 改成 3，其余值不变。

    参数：seg: shape (D, H, W)，int 类型
    返回：同 shape，值 ∈ {0,1,2,3}
    """
    seg = seg.copy()
    seg[seg == 4] = 3
    return seg


def random_crop_3d(
    image: np.ndarray,
    label: np.ndarray,
    patch_size: tuple,
) -> tuple:
    """
    在 image 和 label 上裁同一个随机 3D patch。

    参数：
        image      : shape (4, D, H, W)  —— 4 模态已 stack
        label      : shape (D, H, W)
        patch_size : (pd, ph, pw)         —— 目标 patch 大小，遵循 (D,H,W) 顺序

    返回：
        image_patch: shape (4, pd, ph, pw)
        label_patch: shape (pd, ph, pw)

    逻辑：
        在每个维度上，随机选一个起点 start，
        使得 [start : start+patch_size] 不越界。
        如果 volume 某维度小于 patch_size，先 pad 再裁。
    """
    pd, ph, pw = patch_size
    _, D, H, W = image.shape   # image: (4, D, H, W)

    # ── Step 1: 如果 volume 小于 patch_size，先 pad ──────────
    # pad 量 = max(0, patch_size - volume_size)
    pad_d = max(0, pd - D)
    pad_h = max(0, ph - H)
    pad_w = max(0, pw - W)

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        # image: (4, D, H, W) → pad 后三个空间维度
        image = np.pad(
            image,
            ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
            mode='constant', constant_values=0
        )
        label = np.pad(
            label,
            ((0, pad_d), (0, pad_h), (0, pad_w)),
            mode='constant', constant_values=0
        )
        D, H, W = image.shape[1], image.shape[2], image.shape[3]

    # ── Step 2: 随机选起点 ───────────────────────────────────
    # randint(a, b) → [a, b)，所以上界是 size - patch_size + 1（保证终点不越界）
    start_d = random.randint(0, D - pd)
    start_h = random.randint(0, H - ph)
    start_w = random.randint(0, W - pw)

    # ── Step 3: 裁剪 ─────────────────────────────────────────
    image_patch = image[
        :,
        start_d : start_d + pd,
        start_h : start_h + ph,
        start_w : start_w + pw,
    ]  # shape: (4, pd, ph, pw)

    label_patch = label[
        start_d : start_d + pd,
        start_h : start_h + ph,
        start_w : start_w + pw,
    ]  # shape: (pd, ph, pw)

    return image_patch, label_patch


# ══════════════════════════════════════════════════════════════
# Dataset 主类
# ══════════════════════════════════════════════════════════════

class BraTSDataset(Dataset):
    """
    BraTS2019 3D 分割 Dataset。

    目录结构预期（每个 case 一个文件夹）：
        data_root/
          BraTS19_xxx/
            BraTS19_xxx_flair.nii.gz
            BraTS19_xxx_t1.nii.gz
            BraTS19_xxx_t1ce.nii.gz
            BraTS19_xxx_t2.nii.gz
            BraTS19_xxx_seg.nii.gz

    ★ 注意：BraTS2019 的文件扩展名有时是 .nii，有时是 .nii.gz。
      本代码用 glob 同时匹配两种格式（见 _find_file）。

    返回（__getitem__）：
        x : torch.FloatTensor, shape [4, D, H, W]
        y : torch.LongTensor,  shape [D, H, W]，值 ∈ {0,1,2,3}
    """

    def __init__(
        self,
        data_root:  str,
        case_list:  list,        # 病例文件夹路径的列表（由外部 split 函数生成）
        patch_size: tuple,       # (D, H, W)
        is_train:   bool = True, # 训练集做随机裁剪，验证集裁中心 patch（★ 待确认：验证策略）（否则无法验证一个模型是否比另一个好）
    ):
        super().__init__()
        self.data_root  = data_root
        self.case_list  = case_list
        self.patch_size = patch_size
        self.is_train   = is_train

        print(f"[Dataset] {'Train' if is_train else 'Val'} set: {len(case_list)} cases")

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx: int):
        case_dir = self.case_list[idx]
        case_id  = os.path.basename(case_dir)

        # ── Step 1: 找到每个模态的文件路径 ───────────────────
        flair_path = self._find_file(case_dir, "flair")
        t1_path    = self._find_file(case_dir, "t1")     # 注意：t1 和 t1ce 要分开匹配
        t1ce_path  = self._find_file(case_dir, "t1ce")
        t2_path    = self._find_file(case_dir, "t2")
        seg_path   = self._find_file(case_dir, "seg")

        # ── Step 2: 用 nibabel 加载，得到 (H, W, D) ──────────
        flair = load_nii(flair_path)   # (H, W, D)
        t1    = load_nii(t1_path)
        t1ce  = load_nii(t1ce_path)
        t2    = load_nii(t2_path)
        seg   = load_nii(seg_path)     # (H, W, D)，float32（后面转 int）

        # ── Step 3: 转置到 (D, H, W) —— 全项目统一轴顺序 ────
        # nibabel: (H=240, W=240, D=155) → 转置 → (D=155, H=240, W=240)
        flair = np.transpose(flair, (2, 0, 1))  # (D, H, W)
        t1    = np.transpose(t1,    (2, 0, 1))
        t1ce  = np.transpose(t1ce,  (2, 0, 1))
        t2    = np.transpose(t2,    (2, 0, 1))
        seg   = np.transpose(seg,   (2, 0, 1))  # (D, H, W)

        # ── Step 4: z-score 归一化（只在非零区域）──────────────
        flair = zscore_normalize(flair)
        t1    = zscore_normalize(t1)
        t1ce  = zscore_normalize(t1ce)
        t2    = zscore_normalize(t2)
        # seg 不做归一化

        # ── Step 5: 标签处理 ──────────────────────────────────
        seg = seg.astype(np.int32)     # 从 float32 转成整数
        seg = remap_label(seg)         # {0,1,2,4} → {0,1,2,3}

        # 快速校验（debug 用，不影响速度）
        unique_vals = np.unique(seg)
        assert not np.any(unique_vals > 3), \
            f"[{case_id}] 标签重映射后仍有非法值: {unique_vals}"

        # ── Step 6: Stack 四模态 → (4, D, H, W) ──────────────
        image = np.stack([t1, t1ce, t2, flair], axis=0)  # channel 0=T1, 1=T1ce, 2=T2, 3=FLAIR

        # ── Step 7: 裁 3D patch ───────────────────────────────
        if self.is_train:
            image, seg = random_crop_3d(image, seg, self.patch_size)
        else:
            # 验证集：裁中心 patch（简单策略，★ 待确认：是否改为滑窗推理）
            image, seg = center_crop_3d(image, seg, self.patch_size)

        # ── Step 8: 转成 PyTorch tensor ───────────────────────
        # x: [4, D, H, W] float32
        # y: [D, H, W]    int64 (long)
        x = torch.from_numpy(image).float()         # [4, D, H, W]
        y = torch.from_numpy(seg.astype(np.int64))  # [D, H, W]

        return x, y

    # ── 私有辅助函数 ──────────────────────────────────────────

    def _find_file(self, case_dir: str, modality: str) -> str:
        """
        在 case_dir 下找包含 modality 字符串的 .nii 或 .nii.gz 文件。

        特殊处理：
            - 搜索 "t1" 时需要排除 "t1ce"，避免错误匹配
            - 搜索 "t1ce" 时正常匹配即可

        返回文件的完整路径；找不到则报错。
        """
        # 搜索两种后缀
        candidates = (
            glob.glob(os.path.join(case_dir, f"*{modality}*.nii.gz")) +
            glob.glob(os.path.join(case_dir, f"*{modality}*.nii"))
        )

        # 针对 t1 的特殊处理：排除文件名中含 t1ce 的结果
        if modality == "t1":
            candidates = [p for p in candidates if "t1ce" not in os.path.basename(p)]

        if len(candidates) == 0:
            raise FileNotFoundError(
                f"[Dataset] 在 {case_dir} 中找不到模态 '{modality}' 的文件。\n"
                f"请检查文件名是否包含 '{modality}' 字符串。"
            )
        if len(candidates) > 1:
            # 多个匹配时取第一个，并发出警告
            print(f"[Dataset WARNING] 找到多个 '{modality}' 文件，使用第一个: {candidates[0]}")

        return candidates[0]


# ══════════════════════════════════════════════════════════════
# 中心裁剪（验证集用）
# ══════════════════════════════════════════════════════════════

def center_crop_3d(
    image: np.ndarray,
    label: np.ndarray,
    patch_size: tuple,
) -> tuple:
    """
    从 volume 中心裁 patch_size 大小的 patch。
    shape 约定同 random_crop_3d。

    ★ 待确认：这是最简单的验证策略。
      标准做法是滑窗推理（sliding window inference），
      但 MVP 阶段用中心裁剪已足够验证 pipeline 正确性。
    """
    pd, ph, pw = patch_size
    _, D, H, W = image.shape

    # 先 pad（同 random_crop_3d）
    pad_d = max(0, pd - D)
    pad_h = max(0, ph - H)
    pad_w = max(0, pw - W)

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0,0),(0,pad_d),(0,pad_h),(0,pad_w)),
                       mode='constant', constant_values=0)
        label = np.pad(label, ((0,pad_d),(0,pad_h),(0,pad_w)),
                       mode='constant', constant_values=0)
        D, H, W = image.shape[1], image.shape[2], image.shape[3]

    # 中心起点
    start_d = (D - pd) // 2
    start_h = (H - ph) // 2
    start_w = (W - pw) // 2

    image_patch = image[:, start_d:start_d+pd, start_h:start_h+ph, start_w:start_w+pw]
    label_patch = label[   start_d:start_d+pd, start_h:start_h+ph, start_w:start_w+pw]

    return image_patch, label_patch


# ══════════════════════════════════════════════════════════════
# 数据集划分工具
# ══════════════════════════════════════════════════════════════

def get_case_list(data_root: str, debug_num_cases=None) -> list:
    """
    扫描 data_root 下所有病例文件夹，返回路径列表。

    当前目录结构：
        data_root/
          BraTS19_001/
          BraTS19_002/
          ...
    """
    case_dirs = sorted(
        [
            p for p in glob.glob(os.path.join(data_root, "BraTS19_*"))
            if os.path.isdir(p)
        ]
    )

    if len(case_dirs) == 0:
        raise RuntimeError(
            f"[get_case_list] 在 {data_root} 下找不到任何 BraTS19_* 目录。\n"
            "请检查 cfg['data_root'] 路径是否正确。"
        )

    print(f"[get_case_list] 共找到 {len(case_dirs)} 个 case")

    if debug_num_cases is not None:
        case_dirs = case_dirs[:debug_num_cases]
        print(f"[get_case_list] DEBUG 模式：只使用前 {debug_num_cases} 个 case")

    return case_dirs


def split_dataset(case_list: list, train_ratio: float = 0.8, seed: int = 42):
    """
    按比例划分训练集 / 验证集，固定随机种子保证可复现。

    返回：
        train_cases: list of str
        val_cases  : list of str
    """
    random.seed(seed)
    shuffled = case_list.copy()
    random.shuffle(shuffled)

    n_train = int(len(shuffled) * train_ratio)
    train_cases = shuffled[:n_train]
    val_cases   = shuffled[n_train:]

    print(f"[split_dataset] train: {len(train_cases)}, val: {len(val_cases)}")
    return train_cases, val_cases


# ══════════════════════════════════════════════════════════════
# 快速自测（直接运行这个文件时执行）
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    用法：cd project根目录 && python brats/dataset.py
    作用：加载 1 个 batch，打印 shape，确认 pipeline 无误。
    这是 MVP 最重要的一步：先把数据管道跑通，再写模型。
    """
    print("=" * 50)
    print("BraTS Dataset Sanity Check")
    print("=" * 50)

    # ── 1. 获取 case 列表（debug 模式只用 2 个）──────────────
    all_cases = get_case_list(
        data_root=cfg["data_root"],
        debug_num_cases=cfg["debug_num_cases"],
    )

    train_cases, val_cases = split_dataset(
        all_cases,
        train_ratio=cfg["train_ratio"],
    )

    # ── 2. 构建 Dataset ───────────────────────────────────────
    train_ds = BraTSDataset(
        data_root=cfg["data_root"],
        case_list=train_cases,
        patch_size=cfg["patch_size"],
        is_train=True,
    )

    # ── 3. 取一个样本，打印 shape ─────────────────────────────
    x, y = train_ds[0]

    print(f"\n[Sanity Check] 取第 0 个样本：")
    print(f"  x.shape  = {x.shape}   ← 期望 [4, D, H, W]")
    print(f"  y.shape  = {y.shape}   ← 期望 [D, H, W]")
    print(f"  x.dtype  = {x.dtype}")
    print(f"  y.dtype  = {y.dtype}")
    print(f"  x 范围   = [{x.min():.3f}, {x.max():.3f}]")
    print(f"  y 唯一值 = {torch.unique(y).tolist()}  ← 期望 ⊆ {{0,1,2,3}}")

    # ── 4. 用 DataLoader 取一个 batch ─────────────────────────
    loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,   # sanity check 时用 0，避免多进程 debug 困难
    )

    batch_x, batch_y = next(iter(loader))
    print(f"\n[Sanity Check] DataLoader 一个 batch：")
    print(f"  batch_x.shape = {batch_x.shape}  ← 期望 [B, 4, D, H, W]")
    print(f"  batch_y.shape = {batch_y.shape}  ← 期望 [B, D, H, W]")

    print("\n✅ Dataset sanity check 通过！可以继续写 model.py")
