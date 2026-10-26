# =============================================================================
# dataset.py — KiTS19 2.5D Dataset
# =============================================================================
# 【当前真实实现说明】
#   数据来源：PNG 切片，不是原始 KiTS19 NIfTI/HU 数据
#   这意味着：
#     - HU clip（如 -200~300）已无法执行，HU 值已被固化进 PNG 的 0-255
#     - 窗宽窗位调整（腹部窗、肾脏窗）已无法重新执行
#     - 原始 spacing、方向信息已丢失
#   当前可做的预处理：
#     - 读取 PNG → 灰度 → resize → float32 → 归一化到 [0, 1]
#     - mask 像素值映射：兼容 {0,1,2} 和 {0,128,255} 两种存储格式
#   标准做法（后续改进方向）：
#     - 从 NIfTI 读取原始 HU 值
#     - clip HU 到 [-200, 300]，归一化到 [0, 1]
#     - 按 case 做 spacing 对齐（resampling）
#
# 【2.5D 输入设计】
#   以中心切片 s 为预测目标，输入为 [s-1, s, s+1] 三张切片叠成 3 通道
#   x shape: [3, H, W]   （3 通道，每通道一张灰度切片）
#   y shape: [H, W]      （中心切片 s 的分割 mask，值为 0/1/2）
#   边界处理：首尾切片用自身复制 padding，不用零填充
#   折中理由：比纯 2D 多了层间上下文，比 3D U-Net 实现更简单，显存更小
#
# 【数据划分】
#   必须按 case 划分 train/val，不能按 slice 随机划分
#   原因：同一个 case 的相邻切片高度相似，随机划分会导致数据泄漏
# =============================================================================

import os
import sys

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from kits import config


# =============================================================================
# 工具函数：扫描 case 列表
# =============================================================================

def get_case_list(data_root: str) -> list[str]:
    """
    扫描 data_root，返回所有合法 case 的路径列表，按名称排序。

    合法 case：是目录，且同时含有 imaging/ 和 segmentation/ 子目录。

    返回：
        List[str]，每个元素是 case 的完整路径
    """
    # [修正 5] 提前检查 data_root 是否存在，给出明确报错
    if not os.path.exists(data_root):
        raise FileNotFoundError(
            f"[dataset] ❌ DATA_ROOT 不存在: {data_root}\n"
            f"           请检查 config.py 中的 DATA_ROOT 路径是否正确。"
        )
    if not os.path.isdir(data_root):
        raise NotADirectoryError(
            f"[dataset] ❌ DATA_ROOT 不是一个目录: {data_root}"
        )

    cases = []
    for name in sorted(os.listdir(data_root)):         # sorted 保证顺序稳定
        case_dir = os.path.join(data_root, name)
        if not os.path.isdir(case_dir):
            continue
        img_dir = os.path.join(case_dir, "imaging")
        seg_dir = os.path.join(case_dir, "segmentation")
        if os.path.isdir(img_dir) and os.path.isdir(seg_dir):
            cases.append(case_dir)

    if len(cases) == 0:
        raise RuntimeError(
            f"[dataset] ❌ 在 {data_root} 下未找到任何合法 case。\n"
            f"           每个 case 需包含 imaging/ 和 segmentation/ 子目录。"
        )

    return cases


# =============================================================================
# 工具函数：mask 像素值映射
# =============================================================================

def normalize_mask(mask_np: np.ndarray) -> np.ndarray:
    """
    将 mask 的像素值统一映射到 {0, 1, 2}。

    支持常见 PNG 存储格式：
      - {0, 1, 2}
      - {0, 128, 255}
      - {0, 127, 254}
      - 它们的任意子集

    约定：
      0           -> 背景
      127 / 128   -> 肾脏 (class 1)
      254 / 255   -> 肿瘤 (class 2)
    """
    unique_vals = set(np.unique(mask_np).tolist())

    # 情况 A：已经是类别索引
    if unique_vals.issubset({0, 1, 2}):
        return mask_np.astype(np.int64)

    # 情况 B：PNG 拉伸后的标签值
    allowed_vals = {0, 127, 128, 254, 255}
    if unique_vals.issubset(allowed_vals):
        mapped = np.zeros_like(mask_np, dtype=np.int64)
        mapped[(mask_np == 127) | (mask_np == 128)] = 1
        mapped[(mask_np == 254) | (mask_np == 255)] = 2
        return mapped

    raise ValueError(
        f"[dataset] ❌ mask 包含未知像素值: {sorted(unique_vals)}。\n"
        f"当前仅支持 {{0,1,2}} 或 {{0,127/128,254/255}}。"
    )


# =============================================================================
# Dataset 类
# =============================================================================

class KiTS25DDataset(Dataset):
    """
    KiTS19 2.5D 切片级 Dataset。

    【输入输出】
    x: torch.Tensor, shape [3, H, W], float32, 值域 [0, 1]
        通道 0 = slice s-1（或首切片 padding）
        通道 1 = slice s  （中心切片，预测目标对应的切片）
        通道 2 = slice s+1（或末切片 padding）

    y: torch.Tensor, shape [H, W], int64, 值 ∈ {0=背景, 1=肾脏, 2=肿瘤}

    【边界 padding 规则（自身复制，非零填充）】
    s=0   → idx_prev = 0    （通道 0 与通道 1 相同）
    s=N-1 → idx_next = N-1  （通道 2 与通道 1 相同）

    【resize】
    所有图像和 mask 在读取后统一 resize 到 config.IMG_SIZE × config.IMG_SIZE：
    - image：PIL.Image.BILINEAR（连续值，双线性插值合理）
    - mask ：PIL.Image.NEAREST （类别标签，禁止插值混色）

    【当前限制（PNG 版 baseline）】
    - 无法恢复原始 HU 值
    - 无法做窗宽窗位调整
    - 只做简化归一化：PNG 0-255 → float32 [0, 1]

    【transform 说明】
    [修正 3] 当前版本移除 transform 参数。
    原因：如果增强只作用于 x 而不同步作用于 y，会导致图像与 mask 错位。
    后续改进方向：引入支持 (x, y) 同步变换的增强方案，
    例如 albumentations 或手动用相同随机参数分别变换 x 和 y。
    """

    def __init__(self, case_list: list[str], bg_keep_ratio: float = 1.0, augment: bool = False):
        """
        参数：
            case_list    : List[str]，已在外部按 case 划分好的路径列表
            bg_keep_ratio: 纯背景切片保留比例（0.0~1.0）
                           1.0 = 全部保留（用于 val 集）
                           0.2 = 只保留 20% 纯背景（用于 train 集减少背景主导）
            augment      : 是否做数据增强（仅 train 集开启）
        """
        self.img_size    = config.IMG_SIZE       # resize 目标尺寸
        self.augment     = augment
        self.index_table = []                    # [(case_dir, s, n), ...]

        # [修正 1] 在 __init__ 里预先缓存每个 case 的文件名列表
        self._file_cache: dict[str, dict[str, list[str]]] = {}

        n_fg = 0
        n_bg_kept = 0
        n_bg_dropped = 0

        for case_dir in case_list:
            img_dir = os.path.join(case_dir, "imaging")
            seg_dir = os.path.join(case_dir, "segmentation")

            img_files = sorted([
                f for f in os.listdir(img_dir) if f.lower().endswith(".png")
            ])
            seg_files = sorted([
                f for f in os.listdir(seg_dir) if f.lower().endswith(".png")
            ])

            # 断言：同一 case 下 imaging 和 segmentation 切片数必须相等
            assert len(img_files) == len(seg_files), (
                f"[dataset] ❌ case {case_dir} 的 imaging({len(img_files)}) "
                f"和 segmentation({len(seg_files)}) 切片数不一致！"
            )
            assert len(img_files) > 0, (
                f"[dataset] ❌ case {case_dir} 的 imaging/ 下没有 PNG 文件！"
            )

            # 写入缓存
            self._file_cache[case_dir] = {
                "img": img_files,
                "seg": seg_files,
            }

            # 写入切片索引表（可选：过滤纯背景切片）
            n = len(img_files)
            for s in range(n):
                if bg_keep_ratio < 1.0:
                    # 快速检查 mask 是否含前景（读原始 PNG，不做 resize，只看 max）
                    seg_path = os.path.join(seg_dir, seg_files[s])
                    seg_arr = np.array(Image.open(seg_path).convert("L"))
                    has_fg = seg_arr.max() > 0

                    if has_fg:
                        self.index_table.append((case_dir, s, n))
                        n_fg += 1
                    else:
                        # 纯背景切片按概率保留
                        if np.random.rand() < bg_keep_ratio:
                            self.index_table.append((case_dir, s, n))
                            n_bg_kept += 1
                        else:
                            n_bg_dropped += 1
                else:
                    # bg_keep_ratio=1.0 → 全部保留（val 集走这条路径）
                    self.index_table.append((case_dir, s, n))

        if bg_keep_ratio < 1.0:
            print(f"[dataset] 加载完成: {len(case_list)} cases, "
                  f"前景切片={n_fg}, 保留背景={n_bg_kept}, 丢弃背景={n_bg_dropped}, "
                  f"总样本={len(self.index_table)}, "
                  f"resize → {self.img_size}×{self.img_size}")
        else:
            print(f"[dataset] 加载完成: {len(case_list)} cases, "
                  f"{len(self.index_table)} slices, "
                  f"resize → {self.img_size}×{self.img_size}")

    def __len__(self) -> int:
        return len(self.index_table)

    def _load_image(self, case_dir: str, file_idx: int) -> np.ndarray:
        """
        读取一张 imaging PNG，resize，归一化到 [0, 1]。

        返回：np.ndarray, shape [H, W], float32, 值域 [0, 1]

        【当前实现】PNG → 灰度 → resize(BILINEAR) → /255
        【标准做法（后续改进）】NIfTI → HU clip → 窗位归一化 → resize
        """
        img_dir  = os.path.join(case_dir, "imaging")
        filename = self._file_cache[case_dir]["img"][file_idx]
        path     = os.path.join(img_dir, filename)

        img = Image.open(path).convert("L")              # 灰度单通道
        # [修正 2] resize：image 用 BILINEAR
        img = img.resize(
            (self.img_size, self.img_size),
            resample=Image.BILINEAR
        )
        arr = np.array(img, dtype=np.float32) / 255.0   # [H, W], [0,1]
        return arr

    def _load_mask(self, case_dir: str, file_idx: int) -> np.ndarray:
        """
        读取一张 segmentation PNG，resize，映射到 {0,1,2}。

        返回：np.ndarray, shape [H, W], int64, 值 ∈ {0,1,2}

        注意：mask resize 必须用 NEAREST，禁止插值，
              否则会产生 0/1/2 以外的中间值导致类别混淆。
        """
        seg_dir  = os.path.join(case_dir, "segmentation")
        filename = self._file_cache[case_dir]["seg"][file_idx]
        path     = os.path.join(seg_dir, filename)

        mask = Image.open(path).convert("L")
        # [修正 2] resize：mask 用 NEAREST
        mask = mask.resize(
            (self.img_size, self.img_size),
            resample=Image.NEAREST
        )
        mask_np = np.array(mask, dtype=np.uint8)         # [H, W]
        return normalize_mask(mask_np)                   # [H, W], int64

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        返回第 idx 个样本。

        返回：
            x: torch.Tensor, shape [3, H, W], float32
            y: torch.Tensor, shape [H, W],    int64
        """
        case_dir, s, n = self.index_table[idx]

        # 2.5D 边界 padding：确定 prev / curr / next 的实际文件索引
        idx_prev = max(s - 1, 0)        # s=0 时复制自身
        idx_curr = s
        idx_next = min(s + 1, n - 1)    # s=N-1 时复制自身

        # [修正 1] 直接查缓存，不再 os.listdir
        slice_prev = self._load_image(case_dir, idx_prev)   # [H, W]
        slice_curr = self._load_image(case_dir, idx_curr)   # [H, W]
        slice_next = self._load_image(case_dir, idx_next)   # [H, W]

        # 堆叠为 3 通道
        # x shape: [3, H, W], float32, 值域 [0, 1]
        x_np = np.stack([slice_prev, slice_curr, slice_next], axis=0)

        # 读取中心切片的 mask
        # y shape: [H, W], int64, 值 ∈ {0, 1, 2}
        y_np = self._load_mask(case_dir, idx_curr)

        # 数据增强：随机水平翻转（仅 train 集）
        # 对肾脏分割安全（左右肾对称），image 和 mask 同步翻转不会错位
        if self.augment and np.random.rand() > 0.5:
            x_np = x_np[:, :, ::-1].copy()    # 翻转所有 3 个通道的 W 维度
            y_np = y_np[:, ::-1].copy()        # 翻转 mask 的 W 维度

        x = torch.from_numpy(x_np)          # [3, H, W], float32
        y = torch.from_numpy(y_np).long()   # [H, W],    int64

        return x, y


# =============================================================================
# 数据划分函数（按 case 划分，不按 slice）
# =============================================================================

def compute_class_pixel_freq(loader: DataLoader, num_classes: int = config.NUM_CLASSES) -> torch.Tensor:
    """
    遍历一个 DataLoader，统计每个类别的像素频率。

    返回：
        torch.Tensor, shape [num_classes], 每个类别的像素占比，总和为 1.0

    用途：用于计算 CE loss 的类别权重
    """
    pixel_counts = torch.zeros(num_classes, dtype=torch.float64)
    for _, y in loader:
        for c in range(num_classes):
            pixel_counts[c] += (y == c).sum().item()

    freq = pixel_counts / pixel_counts.sum()
    return freq


def get_dataloaders(
    data_root  : str   = config.DATA_ROOT,
    val_ratio  : float = config.VAL_RATIO,
    batch_size : int   = config.BATCH_SIZE,
    num_workers: int   = config.NUM_WORKERS,
    seed       : int   = config.SEED,
    bg_keep_ratio: float = config.BG_KEEP_RATIO,
) -> tuple[DataLoader, DataLoader]:
    """
    按 case 划分 train/val，返回两个 DataLoader。

    改动：
      - train 集使用 bg_keep_ratio 过滤纯背景切片
      - val 集始终保留所有切片（bg_keep_ratio=1.0），保证评估一致性

    返回：
        train_loader, val_loader
    """
    all_cases = get_case_list(data_root)

    train_cases, val_cases = train_test_split(
        all_cases,
        test_size    = val_ratio,
        random_state = seed,
        shuffle      = True,
    )

    print(f"[dataset] Case 划分: "
          f"train={len(train_cases)} cases, val={len(val_cases)} cases")

    # train 集：过滤纯背景切片 + 数据增强
    train_dataset = KiTS25DDataset(train_cases, bg_keep_ratio=bg_keep_ratio, augment=True)
    # val 集：保留所有切片，不做增强，保证评估口径一致
    val_dataset   = KiTS25DDataset(val_cases, bg_keep_ratio=1.0, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,            # 训练时打乱 slice 顺序
        num_workers = num_workers,
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )

    print(f"[dataset] Train batches: {len(train_loader)}, "
          f"Val batches: {len(val_loader)}")

    return train_loader, val_loader


# =============================================================================
# Sanity Check
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("dataset.py sanity check")
    print("=" * 60)

    # --- 1. 扫描 case 列表 ---
    all_cases = get_case_list(config.DATA_ROOT)
    print(f"\n总 case 数: {len(all_cases)}")
    print(f"第一个 case: {os.path.basename(all_cases[0])}")
    print(f"最后一个 case: {os.path.basename(all_cases[-1])}")

    # --- 2. 按 case 划分 ---
    train_cases, val_cases = train_test_split(
        all_cases,
        test_size    = config.VAL_RATIO,
        random_state = config.SEED,
        shuffle      = True,
    )
    print(f"\nTrain cases: {len(train_cases)}, Val cases: {len(val_cases)}")

    # --- 3. 构建 Dataset ---
    dataset = KiTS25DDataset(train_cases)
    print(f"总 slice 数（train）: {len(dataset)}")

    # --- 4. 取第 0 个样本，检查 shape / dtype / 值域 ---
    x, y = dataset[0]
    print(f"\n--- 第 0 个样本 ---")
    print(f"x shape  : {x.shape}")                              # 期望 [3, 512, 512]
    print(f"x dtype  : {x.dtype}")                              # 期望 float32
    print(f"x 值域   : [{x.min():.4f}, {x.max():.4f}]")        # 期望 [0, 1]
    print(f"y shape  : {y.shape}")                              # 期望 [512, 512]
    print(f"y dtype  : {y.dtype}")                              # 期望 int64
    print(f"y unique : {y.unique().tolist()}")                  # 期望子集 of [0,1,2]

    # --- 5. DataLoader batch 测试 ---
    print("\n--- DataLoader batch 测试 ---")
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    x_batch, y_batch = next(iter(loader))
    print(f"x_batch shape : {x_batch.shape}")    # 期望 [4, 3, 512, 512]
    print(f"y_batch shape : {y_batch.shape}")    # 期望 [4, 512, 512]
    print(f"y_batch unique: {y_batch.unique().tolist()}")

    # --- 6. 边界 padding 验证 ---
    print("\n--- 边界 padding 验证 ---")
    case0_dir = train_cases[0]
    n_slices  = len(dataset._file_cache[case0_dir]["img"])
    print(f"Case 0 总切片数: {n_slices}")

    # 找到 case0 的第一个切片（s=0）
    first_idx = next(
        i for i, (cd, s, _) in enumerate(dataset.index_table)
        if cd == case0_dir and s == 0
    )
    x_first, _ = dataset[first_idx]
    # 首切片：通道 0（prev padding）应等于通道 1（curr）
    print(f"首切片 通道0==通道1: {torch.allclose(x_first[0], x_first[1])}")  # True

    # 找到 case0 的最后一张切片（s=N-1）
    last_idx = next(
        i for i, (cd, s, _) in enumerate(dataset.index_table)
        if cd == case0_dir and s == n_slices - 1
    )
    x_last, _ = dataset[last_idx]
    # 末切片：通道 2（next padding）应等于通道 1（curr）
    print(f"末切片 通道2==通道1: {torch.allclose(x_last[2], x_last[1])}")   # True

    print("\n✅ sanity check 全部通过")
    print("=" * 60)