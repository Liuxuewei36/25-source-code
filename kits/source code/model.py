# =============================================================================
# model.py — 最小 2D U-Net（2.5D baseline 专用）
# =============================================================================
# 【设计说明】
#   这是一个"够用即可"的最小 U-Net，服务于复试展示，不追求 SOTA。
#   核心改动只有一处：输入通道从标准 1 改为 3，对应 2.5D 的三切片输入。
#   其余结构与教科书 U-Net 完全一致，方便口头讲解。
#
# 【2.5D 与模型的关系】
#   输入 x: [B, 3, H, W]  ← 3 个通道 = [s-1, s, s+1] 三张灰度切片
#   输出  : [B, 3, H, W]  ← 3 个类别的 raw logits（背景/肾脏/肿瘤）
#   模型本身仍然是标准 2D 卷积，"2.5D"体现在 Dataset 的输入构造上，
#   而不是模型结构上。这一点在复试中很值得主动说清楚。
#
# 【输出约定：raw logits，不含 softmax】
#   - CrossEntropy Loss（torch 内置）自带 log_softmax，输入必须是 raw logits
#   - Dice Loss 在 loss.py 里手动加 softmax
#   - 推理时在 evaluate.py 里手动 argmax
#   ❌ 不要在模型 forward 里加 softmax，否则 CE loss 会双重 softmax，导致训练出错
#
# 【文件结构】
#   ConvBlock    — 两次 Conv-BN-ReLU，U-Net 的基本单元
#   Encoder      — 4 次下采样，每次 MaxPool + ConvBlock
#   Decoder      — 4 次上采样，每次 ConvTranspose + skip concat + ConvBlock
#   UNet2D       — 组装 Encoder + Bottleneck + Decoder + 输出头
# =============================================================================

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from kits import config


# =============================================================================
# 基本模块：两次 Conv-BN-ReLU
# =============================================================================

class ConvBlock(nn.Module):
    """
    U-Net 的基本卷积单元：连续两次 (Conv2d → BatchNorm2d → ReLU)。

    输入 shape:  [B, in_ch,  H, W]
    输出 shape:  [B, out_ch, H, W]   （H、W 不变，padding=1 保持尺寸）

    【为什么用 BN】
    BatchNorm 加速收敛，减少对学习率的敏感度，MVP 阶段标配。
    后续改进方向：可换成 GroupNorm（batch_size 很小时 BN 不稳定）。

    【为什么 padding=1】
    3×3 卷积 + padding=1 → 输出尺寸 = 输入尺寸，便于 skip connection 拼接。
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            # 第一次卷积
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            # bias=False：后面跟 BN，BN 自带偏移参数，Conv bias 冗余
            nn.BatchNorm2d(out_ch),
            # 【当前 MVP 选择】BatchNorm2d
            # 适合 batch_size >= 4 的常规训练
            # 【后续改进方向 / 待确认】若 batch_size <= 2，BN 统计不稳定，
            # 可改为 nn.GroupNorm(num_groups=8, num_channels=out_ch)

            nn.ReLU(inplace=True),

            # 第二次卷积
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:      [B, in_ch,  H, W]
        # return: [B, out_ch, H, W]
        return self.block(x)


# =============================================================================
# 编码器（下采样路径）
# =============================================================================

class Encoder(nn.Module):
    """
    U-Net 编码器：4 个下采样阶段。

    每个阶段：ConvBlock → 保存 skip feature → MaxPool2d(2×2) 下采样

    输入:  [B, in_ch, H,    W   ]
    输出:  bottleneck feature + 4 个 skip connection feature

    通道数变化：in_ch → 64 → 128 → 256 → 512
    空间尺寸变化（以 H=512 为例）：
        after stage1 pool: 256
        after stage2 pool: 128
        after stage3 pool: 64
        after stage4 pool: 32   ← 送入 bottleneck
    """

    def __init__(self, in_ch: int, base_ch: int = 64):
        """
        参数：
            in_ch  : 输入通道数，2.5D 时为 3
            base_ch: 第一层输出通道数，后续每层翻倍，默认 64
        """
        super().__init__()

        self.enc1 = ConvBlock(in_ch,        base_ch)       # [B,  64, H,   W  ]
        self.enc2 = ConvBlock(base_ch,      base_ch * 2)   # [B, 128, H/2, W/2]
        self.enc3 = ConvBlock(base_ch * 2,  base_ch * 4)   # [B, 256, H/4, W/4]
        self.enc4 = ConvBlock(base_ch * 4,  base_ch * 8)   # [B, 512, H/8, W/8]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        """
        返回：
            skips: list of 4 个 feature map，用于 decoder skip connection
                   skips[0]: [B,  64, H,   W  ]
                   skips[1]: [B, 128, H/2, W/2]
                   skips[2]: [B, 256, H/4, W/4]
                   skips[3]: [B, 512, H/8, W/8]
            out  : [B, 512, H/16, W/16]  ← 送入 bottleneck
        """
        s1 = self.enc1(x)           # [B,  64, H,    W   ]
        s2 = self.enc2(self.pool(s1))  # [B, 128, H/2,  W/2 ]
        s3 = self.enc3(self.pool(s2))  # [B, 256, H/4,  W/4 ]
        s4 = self.enc4(self.pool(s3))  # [B, 512, H/8,  W/8 ]

        out = self.pool(s4)            # [B, 512, H/16, W/16]

        return [s1, s2, s3, s4], out


# =============================================================================
# 解码器（上采样路径）
# =============================================================================

class Decoder(nn.Module):
    """
    U-Net 解码器：4 个上采样阶段。

    每个阶段：
        1. ConvTranspose2d 上采样（×2）
        2. 与对应 skip feature 在通道维度拼接（concat）
        3. ConvBlock 融合特征

    通道数变化（逆序）：512 → 256 → 128 → 64 → 32（最终送入输出头）

    【为什么用 ConvTranspose 而不是 Upsample + Conv】
    ConvTranspose 是可学习的上采样，参数量稍多但更灵活。
    后续改进方向：可换成 nn.Upsample(mode='bilinear') + ConvBlock，
    防止 ConvTranspose 产生的棋盘格伪影（checkerboard artifact）。
    """

    def __init__(self, base_ch: int = 64):
        super().__init__()

        # 上采样层：ConvTranspose2d(in, out, kernel=2, stride=2) → 空间 ×2
        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8,
                                       kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(base_ch * 8,  base_ch * 4,
                                       kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(base_ch * 4,  base_ch * 2,
                                       kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2,  base_ch,
                                       kernel_size=2, stride=2)

        # 拼接后的卷积块：输入通道 = 上采样通道 + skip 通道（×2）
        self.dec4 = ConvBlock(base_ch * 16, base_ch * 8)   # 1024 → 512
        self.dec3 = ConvBlock(base_ch * 8,  base_ch * 4)   # 512  → 256
        self.dec2 = ConvBlock(base_ch * 4,  base_ch * 2)   # 256  → 128
        self.dec1 = ConvBlock(base_ch * 2,  base_ch)       # 128  → 64

    def forward(self,
                bottleneck: torch.Tensor,
                skips: list[torch.Tensor]) -> torch.Tensor:
        """
        参数：
            bottleneck: [B, 1024, H/16, W/16]  ← 来自 bottleneck ConvBlock
            skips      : [s1, s2, s3, s4]       ← 来自 Encoder

        返回：
            [B, 64, H, W]
        """
        # --- 第 4 级上采样 ---
        x = self.up4(bottleneck)              # [B, 512, H/8,  W/8 ]
        x = torch.cat([x, skips[3]], dim=1)  # [B,1024, H/8,  W/8 ]  concat skip s4
        x = self.dec4(x)                      # [B, 512, H/8,  W/8 ]

        # --- 第 3 级上采样 ---
        x = self.up3(x)                       # [B, 256, H/4,  W/4 ]
        x = torch.cat([x, skips[2]], dim=1)  # [B, 512, H/4,  W/4 ]  concat skip s3
        x = self.dec3(x)                      # [B, 256, H/4,  W/4 ]

        # --- 第 2 级上采样 ---
        x = self.up2(x)                       # [B, 128, H/2,  W/2 ]
        x = torch.cat([x, skips[1]], dim=1)  # [B, 256, H/2,  W/2 ]  concat skip s2
        x = self.dec2(x)                      # [B, 128, H/2,  W/2 ]

        # --- 第 1 级上采样 ---
        x = self.up1(x)                       # [B,  64, H,    W   ]
        x = torch.cat([x, skips[0]], dim=1)  # [B, 128, H,    W   ]  concat skip s1
        x = self.dec1(x)                      # [B,  64, H,    W   ]

        return x                              # [B, 64, H, W]


# =============================================================================
# 完整 U-Net
# =============================================================================

class UNet2D(nn.Module):
    """
    2D U-Net，专为 KiTS19 2.5D baseline 设计。

    【整体结构】
    Input [B, 3, H, W]
        ↓ Encoder（4× 下采样）
    Bottleneck [B, 1024, H/16, W/16]
        ↓ Decoder（4× 上采样 + skip concat）
    Output head（1×1 Conv）
    Output [B, num_classes, H, W]  ← raw logits，不含 softmax

    【参数规模估算（base_ch=64）】
    约 31M 参数（标准 U-Net），在单张 512×512 CT 切片上显存友好。
    后续改进方向：可将 base_ch 改为 32 进一步减小模型，加快训练。

    【复试讲述要点】
    "模型本身是标准 2D U-Net，2.5D 体现在输入端：
     把相邻 3 张切片叠成 3 通道，输入通道从 1 改为 3，
     其余结构不变。这样既引入了层间上下文，又保持了模型简洁。"
    """

    def __init__(self,
                 in_ch      : int = config.IN_CHANNELS,   # 默认 3
                 num_classes: int = config.NUM_CLASSES,   # 默认 3
                 base_ch    : int = config.BASE_CHANNELS):
        """
        参数：
            in_ch       : 输入通道数，2.5D 时 = SLICE_WINDOW = 3
            num_classes : 输出类别数，KiTS19 = 3（背景/肾脏/肿瘤）
            base_ch     : 第一层通道数，后续逐层翻倍
        """
        super().__init__()

        self.encoder    = Encoder(in_ch, base_ch)

        # Bottleneck：最深处的卷积块
        # 输入 [B, 512, H/16, W/16] → 输出 [B, 1024, H/16, W/16]
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 16)

        self.decoder    = Decoder(base_ch)

        # 输出头：1×1 卷积，将通道数映射到类别数
        # [B, 64, H, W] → [B, num_classes, H, W]
        self.out_conv   = nn.Conv2d(base_ch, num_classes, kernel_size=1)

        # 权重初始化（Kaiming，适合 ReLU 激活）
        self._init_weights()

    def _init_weights(self):
        """
        对所有 Conv2d 做 Kaiming 初始化，BN 权重初始化为 1。
        良好的初始化可以加速收敛，减少早期 loss 震荡。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数：
            x: [B, in_ch, H, W]  → 2.5D 时 in_ch=3

        返回：
            logits: [B, num_classes, H, W]  ← raw logits，不含 softmax/sigmoid
                    在 loss.py 里分别处理：
                      CE loss  → 直接接收 logits（内部含 log_softmax）
                      Dice loss → 先 softmax 再计算
        """
        # --- 编码 ---
        skips, enc_out = self.encoder(x)
        # enc_out: [B, 512,  H/16, W/16]
        # skips  : [s1:[B,64,H,W], s2:[B,128,H/2,W/2],
        #           s3:[B,256,H/4,W/4], s4:[B,512,H/8,W/8]]

        # --- Bottleneck ---
        bottleneck = self.bottleneck(enc_out)
        # bottleneck: [B, 1024, H/16, W/16]

        # --- 解码 ---
        dec_out = self.decoder(bottleneck, skips)
        # dec_out: [B, 64, H, W]

        # --- 输出头 ---
        logits = self.out_conv(dec_out)
        # logits: [B, num_classes, H, W]

        return logits


# =============================================================================
# Sanity Check
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("model.py sanity check")
    print("=" * 60)

    device = config.DEVICE
    model  = UNet2D(
        in_ch       = config.IN_CHANNELS,
        num_classes = config.NUM_CLASSES,
        base_ch     = 64,
    ).to(device)

    # --- 参数量统计 ---
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    print(f"\n总参数量    : {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # --- Forward pass ---
    B, C, H, W = 2, config.IN_CHANNELS, config.IMG_SIZE, config.IMG_SIZE
    dummy_input = torch.randn(B, C, H, W).to(device)
    print(f"\n输入 shape  : {list(dummy_input.shape)}")   # [2, 3, 512, 512]
# 期望: [B, {config.IN_CHANNELS}, {config.IMG_SIZE}, {config.IMG_SIZE}]
    with torch.no_grad():
        output = model(dummy_input)

    print(f"输出 shape  : {list(output.shape)}")          # [2, 3, 512, 512]
# 期望: [B, {config.NUM_CLASSES}, {config.IMG_SIZE}, {config.IMG_SIZE}]
    # --- 验证输出是 raw logits（不是概率）---
    # softmax 后各类别概率之和为 1，raw logits 没有这个约束
    import torch.nn.functional as F
    probs = F.softmax(output, dim=1)
    prob_sum = probs.sum(dim=1)          # [B, H, W]，每个像素的概率之和
    print(f"\n各像素概率之和（应全为 1.0）: "
          f"min={prob_sum.min():.6f}, max={prob_sum.max():.6f}")

    # --- argmax 预测类别 ---
    pred = output.argmax(dim=1)          # [B, H, W]
    print(f"argmax 预测 shape: {list(pred.shape)}")       # [2, 512, 512]
    print(f"预测类别范围     : [{pred.min().item()}, {pred.max().item()}]")

    # --- 显存占用（若有 GPU）---
    if device == "cuda":
        mem_mb = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"\nGPU 显存占用: {mem_mb:.1f} MB")

    print("\n✅ model.py sanity check 通过")
    print("=" * 60)
# 

# **结构一览，方便复试时画图讲解：**

# Input [B, 3, H, W]
#   │
#   ├─ enc1: ConvBlock(3→64)    → skip s1 [B, 64,  H,    W   ]
#   ├─ enc2: ConvBlock(64→128)  → skip s2 [B, 128, H/2,  W/2 ]
#   ├─ enc3: ConvBlock(128→256) → skip s3 [B, 256, H/4,  W/4 ]
#   ├─ enc4: ConvBlock(256→512) → skip s4 [B, 512, H/8,  W/8 ]
#   │   └─ pool
#   │
#   └─ bottleneck: ConvBlock(512→1024)    [B,1024, H/16, W/16]
#         │
#         ├─ up4 + cat(s4) → dec4(1024→512)  [B, 512, H/8,  W/8 ]
#         ├─ up3 + cat(s3) → dec3(512→256)   [B, 256, H/4,  W/4 ]
#         ├─ up2 + cat(s2) → dec2(256→128)   [B, 128, H/2,  W/2 ]
#         ├─ up1 + cat(s1) → dec1(128→64)    [B,  64, H,    W   ]
#         │
#         └─ out_conv: Conv1×1(64→3)
#                ↓
#         Output [B, 3, H, W]  ← raw logits