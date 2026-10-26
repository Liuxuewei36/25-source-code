# =============================================================================
# config.py — KiTS19 2.5D Baseline 项目配置
# =============================================================================
# 【项目定位】
#   这是 BraTS2019 主项目的辅助对比项目。
#   BraTS：多模态 MRI，3D U-Net，脑肿瘤分割
#   KiTS ：单模态 CT（PNG 版），2.5D baseline，肾脏及肿瘤分割
#
# 【当前真实实现 vs 标准做法】
#   当前：数据已经是 PNG 切片，HU 值已映射到 0-255，无法恢复窗宽窗位
#   标准：应在原始 NIfTI 上做 HU clip（-200~300）+ 窗位归一化，再转切片
#   因此当前预处理只做：读 PNG → 归一化到 [0,1] → mask 像素值映射检查
#
# 【2.5D 设计说明】
#   输入：相邻 3 张切片 [s-1, s, s+1] 叠成 3 通道，shape: [3, H, W]
#   输出：中心切片 s 的分割 mask，shape: [H, W]，值为 0/1/2
#   模型：仍然是 2D U-Net，输入通道改为 3
#   折中理由：比纯 2D 多了层间上下文，比 3D 少了显存压力和工程复杂度
# =============================================================================

import os

# -----------------------------------------------------------------------------
# 路径配置
# -----------------------------------------------------------------------------

# 数据根目录：每个子文件夹是一个 case，内含 imaging/ 和 segmentation/
DATA_ROOT = "/root/autodl-tmp/project/kits19_png"       # ← 修改成你的实际路径

# 输出根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs_kits")

# 子目录（自动创建，见 utils.py）
CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT, "checkpoints")
FIGURE_DIR     = os.path.join(OUTPUT_ROOT, "figures")
LOG_DIR        = os.path.join(OUTPUT_ROOT, "logs")

# -----------------------------------------------------------------------------
# 数据配置
# -----------------------------------------------------------------------------

# 2.5D 滑窗宽度：当前固定为 3，即取 [s-1, s, s+1]
# 后续改进方向：可改为 5（[s-2,...,s+2]），做消融实验对比
SLICE_WINDOW = 3                        # 必须是奇数

# KiTS19 分割类别
#   0: 背景 (background)
#   1: 肾脏 (kidney)
#   2: 肿瘤 (tumor)
NUM_CLASSES = 3

# 输入通道数 = SLICE_WINDOW（每张切片一个灰度通道）
IN_CHANNELS = SLICE_WINDOW              # = 3

BASE_CHANNELS = 32   # U-Net 第一层输出通道数，后续逐层翻倍
                     # 改为 32 比默认 64 更省显存，MVP 阶段足够
USE_AMP = True       # 是否启用混合精度训练（AMP），有 CUDA 时建议开启
# 图像尺寸（推理/训练时统一 resize 到此大小）
# KiTS PNG 原始尺寸通常是 512×512，可以直接用，也可以改成 256×256 加速
IMG_SIZE = 512                          # (H, W) = (512, 512)

# 按 case 划分 train/val（不按 slice 随机划分，避免数据泄漏）
# 划分方式：取全部 case 列表，前 VAL_RATIO 比例作为 val，其余为 train
# 后续改进方向：可改为 k-fold cross validation
VAL_RATIO = 0.2                         # 20% case 作为验证集

# 随机种子（固定可复现）
SEED = 42

# -----------------------------------------------------------------------------
# 训练配置
# -----------------------------------------------------------------------------

BATCH_SIZE  = 6                         # 2.5D 切片级训练，显存压力小
NUM_EPOCHS  = 25                        # 配合数据增强，不再容易过拟合
LR          = 1e-3                      # Adam 初始学习率
NUM_WORKERS = 4                         # DataLoader 子进程数，Windows 建议设 0

# Loss 权重（Dice Loss + CrossEntropy Loss 组合）
# 原因：CE 保证基础优化稳定，Dice 直接优化前景类指标
DICE_WEIGHT = 0.5
CE_WEIGHT   = 0.5

# CE 类别权重：None = 自动逆频率（不做均衡化）
#
# v2/v3/v4 三轮对比结果：
#   v2 [1, ~32, ~63] auto     → kidney 0.325, tumor 0.698, mean 0.511 ← 最优
#   v3 [0.3, 1.0, 1.2] manual → kidney 0.191, 太弱
#   v4 [1, ~63, ~63] equalized → kidney ~0, 太激进
# 结论：v2 的自动权重是当前最优配置，不需要修改
CE_CLASS_WEIGHTS = None

# 训练集纯背景切片保留比例（仅影响 train，不影响 val）
# 0.2 = 保留 20% 的纯背景切片，丢弃 80%
# 1.0 = 全部保留（等价于原版行为）
BG_KEEP_RATIO = 0.2

# Dice Loss smooth 参数（防止分母为零）
# 1.0 比 1e-6 在 AMP 半精度下更安全
DICE_SMOOTH = 1.0

# ReduceLROnPlateau 的 patience（连续多少 epoch 无提升才降 LR）
LR_PATIENCE = 5

# 每隔多少 epoch 保存一次 checkpoint（best model 始终单独保存）
SAVE_EVERY  = 5

# -----------------------------------------------------------------------------
# 设备配置
# -----------------------------------------------------------------------------

# eval_results.pt 保存路径（放根输出目录，visualize.py 直接读）
EVAL_RESULTS_PATH = os.path.join(OUTPUT_ROOT, "eval_results.pt")

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# 快速自检：直接运行此文件可以验证路径是否存在
# =============================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("KiTS19 2.5D Baseline — Config Check")
    print("=" * 50)
    print(f"DATA_ROOT    : {DATA_ROOT}")
    print(f"  exists?    : {os.path.exists(DATA_ROOT)}")
    print(f"OUTPUT_ROOT  : {OUTPUT_ROOT}")
    print(f"DEVICE       : {DEVICE}")
    print(f"SLICE_WINDOW : {SLICE_WINDOW}  (2.5D 输入通道数)")
    print(f"NUM_CLASSES  : {NUM_CLASSES}  (0=背景, 1=肾脏, 2=肿瘤)")
    print(f"IMG_SIZE     : {IMG_SIZE}x{IMG_SIZE}")
    print(f"BATCH_SIZE   : {BATCH_SIZE}")
    print(f"NUM_EPOCHS   : {NUM_EPOCHS}")
    print(f"LR           : {LR}")
    print("=" * 50)

    # 检查 case 文件夹是否存在
    if os.path.exists(DATA_ROOT):
        cases = sorted([
            d for d in os.listdir(DATA_ROOT)
            if os.path.isdir(os.path.join(DATA_ROOT, d))
        ])
        print(f"找到 case 数量 : {len(cases)}")
        if cases:
            print(f"第一个 case   : {cases[0]}")
            print(f"最后一个 case : {cases[-1]}")
    else:
        print("⚠️  DATA_ROOT 不存在，请先修改路径再运行。")