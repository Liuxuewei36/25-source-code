# brats/config.py
# ─────────────────────────────────────────────────────────────
# 作用：集中管理所有超参数和路径。
# 好处：改一个地方，全项目生效；复试时可以说"所有超参数统一在 config 管理"。
# ─────────────────────────────────────────────────────────────

import os

cfg = {
    # ── 数据路径 ──────────────────────────────────────────────
    # 指向 BraTS2019 的根目录，里面每个子文件夹是一个病例
    # 例如：/data/BraTS2019/MICCAI_BraTS_2019_Data_Training/HGG/BraTS19_2013_0_1/
    "data_root": "/root/autodl-tmp/project/BraTS2019",          # ← 改成你自己的路径

    # 输出根目录（checkpoints / figures / logs 都放这里）
    "output_dir": "outputs_brats",

    # ── 数据划分 ──────────────────────────────────────────────
    # 训练集占比；其余做验证（当前未做独立测试集，复试时可说明这是 demo 设定）
    "train_ratio": 0.8,

    # ★ Sanity check / overfit 开关
    # 设为正整数 N：只用前 N 个 case（用来快速验证 pipeline）
    # 设为 None：使用全部 case
    "debug_num_cases": 10,                    # 先用 10 个 case 跑通，再改 None

    # ── Patch 设置 ────────────────────────────────────────────
    # 轴顺序约定（全项目统一）：
    #   nibabel 读出来是 (H, W, D) = (240, 240, 155)
    #   我们 transpose 成 (D, H, W) = (155, 240, 240) 再处理
    #   最终 tensor: x → [4, D, H, W], y → [D, H, W]
    #
    # patch_size 也遵循 (D, H, W) 顺序
    #
    # ★ 待确认项：patch_size 需根据实际显存调整
    #   - 8GB  显存建议：(64, 128, 128)
    #   - 16GB 显存建议：(128, 128, 128)
    "patch_size": (96, 128, 128),            # (D, H, W)，待确认

    # ── 训练超参数 ────────────────────────────────────────────
    "num_epochs":    1,                     # debug 时可先设 5
    "batch_size":    1,                      
    "num_workers":   0,                      # DataLoader 工作进程数
    "learning_rate": 1e-4,                   # ★ 待确认：常见范围 1e-4 ~ 3e-4
    "weight_decay":  1e-5,
    # 新增：AMP 混合精度开关
    # True = 用 float16 加速计算，显存减少约 30-40%，速度提升明显
    "use_amp": True,
    # 新增：cudnn benchmark
    # 输入 shape 固定时（patch 固定大小），开启后 cudnn 自动选最快卷积算法
    "cudnn_benchmark": True,
    
    # ── 模型 ──────────────────────────────────────────────────
    "in_channels":  4,                       # 通道顺序：T1 / T1ce / T2 / FLAIR
    "num_classes":  4,                       # 0背景 / 1坏死 / 2水肿 / 3增强肿瘤
    # ★ 待确认：base_channels 越大显存占用越高
    "base_channels": 16,                     # 建议从 16 开始，稳定后再试 32

    # ── Loss ──────────────────────────────────────────────────
    # 可选 "dice" 或 "dice_ce"
    # dice_ce = Dice Loss + CrossEntropy Loss 的简单加权和
    "loss_type": "dice_ce",

    # dice_ce 时两项的权重：total = w_dice * dice + w_ce * ce
    "w_dice": 0.5,
    "w_ce":   0.5,

    # ── 评估 & 保存 ───────────────────────────────────────────
    "val_interval":  1,                      # 每 N 个 epoch 做一次验证
    "save_interval": 5,                      # 每 N 个 epoch 保存一次 checkpoint
}

# ── 自动创建输出子目录（import config 时自动执行）─────────────
for _subdir in ["checkpoints", "figures", "logs"]:
    os.makedirs(os.path.join(cfg["output_dir"], _subdir), exist_ok=True)