# brats/model.py
# ─────────────────────────────────────────────────────────────
# 最小版 3D U-Net，纯 PyTorch 手写，无额外依赖。
#
# 设计原则：
#   - 只用最基础的组件：Conv3d / BatchNorm3d / ReLU / MaxPool3d / ConvTranspose3d
#   - 3 层下采样 + Bottleneck + 3 层上采样，对称结构
#   - Skip connection：把 Encoder 每层输出直接 concat 到对应 Decoder 层
#   - 不加 Residual / Attention / SE / Transformer / Deep Supervision
#
# 输入输出：
#   输入  x      : [B, 4,    D,    H,    W]   4 通道（T1/T1ce/T2/FLAIR）
#   输出  logits : [B, 4,    D,    H,    W]   4 类（背景/坏死/水肿/增强肿瘤）
#   注意：输出是原始 logits，不经过 softmax。
#         CrossEntropyLoss 接收 logits；DiceLoss 内部自己做 softmax。
#
# 默认 base_channels = 16
# 各层通道数：16 → 32 → 64 → 128（bottleneck）→ 64 → 32 → 16 → 4(输出)
# ─────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brats.config import cfg


# ══════════════════════════════════════════════════════════════
# 基础模块：双层卷积块
# ══════════════════════════════════════════════════════════════

class DoubleConv3d(nn.Module):
    """
    # 每层后接 InstanceNorm3d(affine=True) + ReLU
    # ★ 归一化类型属于待确认项；此处选 InstanceNorm 是因为默认 batch_size=1，
    #   BatchNorm 在 batch=1 时统计不稳定，InstanceNorm 对每个样本独立统计，更稳。
    这是 U-Net 每个阶段的基本单元。

    结构：
        Conv3d(in_ch, out_ch, 3, padding=1)
        → InstanceNorm3d(out_ch)
        → ReLU
        → Conv3d(out_ch, out_ch, 3, padding=1)
        → BatchNorm3d(out_ch)
        → ReLU

    padding=1 保证卷积前后空间尺寸不变（same padding）。
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            # 第一层卷积
            nn.Conv3d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            # bias=False：因为后面有 BN，BN 自带偏置，Conv 的 bias 是冗余的
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),

            # 第二层卷积
            nn.Conv3d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)
        # shape 不变：[B, in_ch, D, H, W] → [B, out_ch, D, H, W]


# ══════════════════════════════════════════════════════════════
# Encoder 模块（下采样）
# ══════════════════════════════════════════════════════════════

class EncoderBlock(nn.Module):
    """
    Encoder 的一个阶段：
        DoubleConv3d → 输出 skip feature
        MaxPool3d(2) → 空间尺寸减半，作为下一阶段输入

    返回两个值：
        skip : 送给对应 Decoder 层的 skip connection（未经 pool）
        down : MaxPool 之后的 feature，继续往下传
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv3d(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        # MaxPool3d(2)：在 D/H/W 三个维度各缩小 2 倍

    def forward(self, x):
        skip = self.conv(x)   # [B, out_ch, D,   H,   W  ]  ← 保存给 skip connection
        down = self.pool(skip) # [B, out_ch, D/2, H/2, W/2]  ← 继续往下
        return skip, down


# ══════════════════════════════════════════════════════════════
# Decoder 模块（上采样）
# ══════════════════════════════════════════════════════════════

class DecoderBlock(nn.Module):
    """
    Decoder 的一个阶段：
        ConvTranspose3d (stride=2) → 空间尺寸翻倍（上采样）
        concat skip feature         → 通道数翻倍
        DoubleConv3d                → 融合特征，通道数减半

    为什么用 ConvTranspose3d 而不是 Upsample + Conv？
        - ConvTranspose3d（转置卷积）是可学习的上采样，模型可以自己学习如何插值
        - Upsample（双线性/最近邻）是固定插值，不可学习
        - 两者效果接近，ConvTranspose3d 更"标准"，复试更好讲
        ★ 待确认：如果显存紧张可以换成 Upsample(mode='trilinear') + Conv3d

    关于 skip connection 的 concat：
        Encoder 第 k 层输出 skip_k: [B, C_k, D_k, H_k, W_k]
        上采样后的 feature:          [B, C_k, D_k, H_k, W_k]（理想情况）
        concat 后：                  [B, 2*C_k, D_k, H_k, W_k]
        DoubleConv3d(2*C_k → C_k) 把通道数压回去
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        参数说明：
            in_channels  : 上采样之后（concat 之前）的通道数
                           = 来自下方的通道数（ConvTranspose 的输出通道）
            out_channels : DoubleConv3d 的输出通道数

        concat 之后通道数 = in_channels（上采样）+ in_channels（skip）
                         = 2 * in_channels
        所以 DoubleConv3d 的 in_channels 是 2 * in_channels。
        """
        super().__init__()

        # 转置卷积：把空间尺寸翻倍，同时把通道数从 in_channels*2 降到 in_channels
        # kernel=2, stride=2 是最常用的设置，精确翻倍空间尺寸
        self.up = nn.ConvTranspose3d(
            in_channels * 2,   # 输入通道（来自 bottleneck 或上一个 decoder 的输出）
            in_channels,       # 输出通道（concat 前，和 skip 通道数相等）
            kernel_size=2,
            stride=2,
        )

        # concat 之后通道 = in_channels（up）+ in_channels（skip）= 2 * in_channels
        self.conv = DoubleConv3d(in_channels * 2, out_channels)

    def forward(self, x, skip):
        """
        参数：
            x    : 来自下方（更深层）的 feature，shape [B, in_ch*2, D, H, W]
            skip : 来自 Encoder 同层的 feature，shape [B, in_ch,   D*2, H*2, W*2]
        """
        # ── 上采样：空间尺寸翻倍 ─────────────────────────────
        x = self.up(x)
        # x: [B, in_ch, D*2, H*2, W*2]

        # ── 处理尺寸不匹配（padding 边界情况）──────────────
        # 当 D/H/W 是奇数时，MaxPool 向下取整，ConvTranspose 翻倍后
        # 可能比 skip 小 1。用 F.pad 补齐。
        # 例：skip=(B,C,65,65,65) 而 x=(B,C,64,64,64) → 在右边各 pad 1
        diff_d = skip.shape[2] - x.shape[2]
        diff_h = skip.shape[3] - x.shape[3]
        diff_w = skip.shape[4] - x.shape[4]

        if diff_d > 0 or diff_h > 0 or diff_w > 0:
            # F.pad 的顺序是从最后一个维度开始，两两一组：(W前,W后, H前,H后, D前,D后)
            x = F.pad(x, [
                0, diff_w,   # W 维度：右边 pad
                0, diff_h,   # H 维度：下边 pad
                0, diff_d,   # D 维度：后面 pad
            ])

        # ── Concat skip connection ───────────────────────────
        x = torch.cat([skip, x], dim=1)
        # shape: [B, in_ch + in_ch, D*2, H*2, W*2]
        #      = [B, in_ch * 2,     D*2, H*2, W*2]

        # ── 双层卷积融合特征 ──────────────────────────────────
        x = self.conv(x)
        # shape: [B, out_ch, D*2, H*2, W*2]

        return x


# ══════════════════════════════════════════════════════════════
# 完整 3D U-Net
# ══════════════════════════════════════════════════════════════

class UNet3D(nn.Module):
    """
    最小版 3D U-Net。

    结构示意（base_channels=16）：

    输入 [B, 4, D, H, W]
         │
    Enc1: DoubleConv → [B, 16,  D,   H,   W  ] ──────────────── skip1
         │ MaxPool↓
    Enc2: DoubleConv → [B, 32,  D/2, H/2, W/2] ─────────── skip2
         │ MaxPool↓
    Enc3: DoubleConv → [B, 64,  D/4, H/4, W/4] ──────── skip3
         │ MaxPool↓
    Bottleneck:        [B, 128, D/8, H/8, W/8]
         │ ConvTranspose↑
    Dec3: cat(skip3) → [B, 64,  D/4, H/4, W/4]
         │ ConvTranspose↑
    Dec2: cat(skip2) → [B, 32,  D/2, H/2, W/2]
         │ ConvTranspose↑
    Dec1: cat(skip1) → [B, 16,  D,   H,   W  ]
         │
    输出头 1×1×1 Conv → [B, 4, D, H, W]   (logits，未经 softmax)
    """

    def __init__(
        self,
        in_channels:  int = 4,   # 输入通道：T1/T1ce/T2/FLAIR
        num_classes:  int = 4,   # 输出类别：背景/坏死/水肿/增强肿瘤
        base_channels: int = 16, # 第一层通道数，后续每层翻倍
    ):
        super().__init__()

        b = base_channels   # 简写，b=16

        # ── Encoder ────────────────────────────────────────────
        # EncoderBlock(in, out): DoubleConv + MaxPool
        # 每经过一个 EncoderBlock，空间尺寸减半，通道数翻倍

        self.enc1 = EncoderBlock(in_channels, b)      # 4   → 16
        self.enc2 = EncoderBlock(b,           b * 2)  # 16  → 32
        self.enc3 = EncoderBlock(b * 2,       b * 4)  # 32  → 64

        # ── Bottleneck ─────────────────────────────────────────
        # 最深层，不做 MaxPool，只做双层卷积
        # 输入通道 = enc3 的输出经过 MaxPool 后的通道 = b*4 = 64
        # 输出通道 = b*8 = 128
        self.bottleneck = DoubleConv3d(b * 4, b * 8)  # 64 → 128

        # ── Decoder ────────────────────────────────────────────
        # DecoderBlock(in, out)：
        #   in  = 当前层（来自下方）的通道数的一半
        #         （因为 DecoderBlock 内部 ConvTranspose 输入是 in*2）
        #   out = 该层 DoubleConv 的输出通道数
        #
        # dec3 接收 bottleneck 输出（b*8=128），skip 来自 enc3（b*4=64）
        #   ConvTranspose: 128 → 64，concat skip(64) → 128，DoubleConv: 128 → 64
        self.dec3 = DecoderBlock(b * 4, b * 4)   # in=64, out=64

        # dec2 接收 dec3 输出（b*4=64），skip 来自 enc2（b*2=32）
        #   ConvTranspose: 64 → 32，concat skip(32) → 64，DoubleConv: 64 → 32
        self.dec2 = DecoderBlock(b * 2, b * 2)   # in=32, out=32

        # dec1 接收 dec2 输出（b*2=32），skip 来自 enc1（b=16）
        #   ConvTranspose: 32 → 16，concat skip(16) → 32，DoubleConv: 32 → 16
        self.dec1 = DecoderBlock(b,     b)        # in=16, out=16

        # ── 输出头 ─────────────────────────────────────────────
        # 1×1×1 卷积：把通道数从 b=16 映射到 num_classes=4
        # 不加激活函数，输出原始 logits
        self.out_conv = nn.Conv3d(b, num_classes, kernel_size=1)

    def forward(self, x):
        """
        前向传播，含逐层 shape 注释。
        假设：B=1, in_ch=4, patch=(64,128,128), base_channels=16

        输入 x: [B, 4, 64, 128, 128]
        """

        # ── Encoder ────────────────────────────────────────────
        skip1, x = self.enc1(x)
        # skip1: [B, 16,  64,  128, 128]  ← 保留给 dec1
        # x:     [B, 16,  32,  64,  64 ]  ← MaxPool 后，空间减半

        skip2, x = self.enc2(x)
        # skip2: [B, 32,  32,  64,  64 ]  ← 保留给 dec2
        # x:     [B, 32,  16,  32,  32 ]

        skip3, x = self.enc3(x)
        # skip3: [B, 64,  16,  32,  32 ]  ← 保留给 dec3
        # x:     [B, 64,  8,   16,  16 ]

        # ── Bottleneck ─────────────────────────────────────────
        x = self.bottleneck(x)
        # x: [B, 128, 8, 16, 16]

        # ── Decoder ────────────────────────────────────────────
        x = self.dec3(x, skip3)
        # 内部：ConvTranspose(128→64): [B,64,16,32,32]
        #        cat(skip3):           [B,128,16,32,32]
        #        DoubleConv(128→64):   [B,64,16,32,32]
        # x: [B, 64, 16, 32, 32]

        x = self.dec2(x, skip2)
        # 内部：ConvTranspose(64→32):  [B,32,32,64,64]
        #        cat(skip2):           [B,64,32,64,64]
        #        DoubleConv(64→32):    [B,32,32,64,64]
        # x: [B, 32, 32, 64, 64]

        x = self.dec1(x, skip1)
        # 内部：ConvTranspose(32→16):  [B,16,64,128,128]
        #        cat(skip1):           [B,32,64,128,128]
        #        DoubleConv(32→16):    [B,16,64,128,128]
        # x: [B, 16, 64, 128, 128]

        # ── 输出头 ─────────────────────────────────────────────
        logits = self.out_conv(x)
        # logits: [B, 4, 64, 128, 128]
        # 注意：这是原始 logits，不经过 softmax！
        # - CrossEntropyLoss 需要 logits（PyTorch 内部会做 log_softmax）
        # - DiceLoss 在 loss.py 内部自己做 softmax
        # - 推理时用 torch.argmax(logits, dim=1) 得到预测类别

        return logits


# ══════════════════════════════════════════════════════════════
# 快速自测
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("UNet3D Sanity Check")
    # ★ 归一化：InstanceNorm3d(affine=True)，适配 batch_size=1（待确认项）
    print("=" * 50)

    # 从 config 读取参数
    model = UNet3D(
        in_channels   = cfg["in_channels"],    # 4
        num_classes   = cfg["num_classes"],    # 4
        base_channels = cfg["base_channels"],  # 16
    )

    # 构造一个假输入（不需要真实数据）
    pd, ph, pw = cfg["patch_size"]             # (64, 128, 128)
    B = 1
    dummy_input = torch.zeros(B, cfg["in_channels"], pd, ph, pw)
    # dummy_input: [1, 4, 64, 128, 128]

    print(f"\n输入 shape:  {dummy_input.shape}")

    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"输出 shape:  {output.shape}")
    print(f"期望输出:    torch.Size([{B}, {cfg['num_classes']}, {pd}, {ph}, {pw}])")

    # 验证输出 shape 正确
    expected = torch.Size([B, cfg["num_classes"], pd, ph, pw])
    assert output.shape == expected, \
        f"shape 不匹配！期望 {expected}，得到 {output.shape}"

    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量:    {total_params:,}")
    print(f"可训练参数:  {trainable_params:,}")

    print("\n✅ UNet3D sanity check 通过！可以继续写 loss.py")