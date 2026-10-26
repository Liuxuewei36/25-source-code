# =============================================================================
# utils.py — 通用工具函数
# =============================================================================
# 负责：
#   1. 固定随机种子（保证可复现）
#   2. 创建输出目录
#   3. 保存 / 加载 checkpoint
#   4. 日志写入（append 模式，每个 epoch 追加一行）
# =============================================================================

import os
import random
import csv
from datetime import datetime

import numpy as np
import torch


# -----------------------------------------------------------------------------
# 1. 固定随机种子
# -----------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """
    固定所有随机源，保证实验可复现。
    包括：Python random、numpy、torch CPU、torch GPU（若可用）。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 关闭 cudnn 的随机优化，换取完全可复现
    # 后续改进方向：如果不在意绝对复现，可注释掉下面两行以提升速度
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    print(f"[utils] 随机种子已固定: seed={seed}")


# -----------------------------------------------------------------------------
# 2. 创建输出目录
# -----------------------------------------------------------------------------

def ensure_dirs(*dirs):
    """
    传入任意多个路径，若不存在则自动创建（含多级目录）。

    用法：
        ensure_dirs(CHECKPOINT_DIR, FIGURE_DIR, LOG_DIR)
    """
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"[utils] 输出目录已就绪: {[os.path.basename(d) for d in dirs]}")


# -----------------------------------------------------------------------------
# 3. 保存 / 加载 Checkpoint
# -----------------------------------------------------------------------------

def save_checkpoint(state: dict, path: str):
    """
    保存训练状态到磁盘。

    参数：
        state: dict，通常包含：
            {
                'epoch'     : int,
                'model'     : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'val_dice'  : float,
            }
        path: 保存路径，例如 checkpoints/best_model.pth
    """
    torch.save(state, path)
    print(f"[utils] Checkpoint 已保存: {path}  (epoch={state.get('epoch', '?')},"
          f" val_dice={state.get('val_dice', 0.0):.4f})")


def load_checkpoint(path: str, model: torch.nn.Module,
                    optimizer=None, device: str = "cpu"):
    """
    从磁盘加载 checkpoint，恢复模型权重（以及可选的 optimizer 状态）。

    参数：
        path      : checkpoint 文件路径
        model     : 已实例化的模型
        optimizer : （可选）若传入则同时恢复优化器状态
        device    : 加载到哪个设备

    返回：
        start_epoch : 上次训练到的 epoch（用于 resume 训练）
        best_dice   : 上次记录的最佳 val dice
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[utils] Checkpoint 不存在: {path}")

    ckpt = torch.load(path, map_location=device)

    state_dict = ckpt["model"] if "model" in ckpt else ckpt["model_state"]
    model.load_state_dict(state_dict)

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    start_epoch = ckpt.get("epoch", 0) + 1      # 下一个 epoch 从这里开始
    best_dice   = ckpt.get("val_dice", 0.0)

    print(f"[utils] Checkpoint 已加载: {path}")
    print(f"        恢复自 epoch={start_epoch - 1}, best_val_dice={best_dice:.4f}")

    return start_epoch, best_dice


# -----------------------------------------------------------------------------
# 4. CSV 日志（每个 epoch 追加一行）
# -----------------------------------------------------------------------------

class CSVLogger:
    """
    轻量训练日志，每个 epoch 追加写入一行 CSV。

    字段：
        timestamp, epoch, train_loss, val_loss,
        val_dice_kidney, val_dice_tumor, val_dice_mean, lr

    用法：
        logger = CSVLogger(LOG_DIR / "train_log.csv")
        logger.write(
            epoch=1,
            train_loss=0.5,
            val_loss=0.4,
            val_dice_kidney=0.82,
            val_dice_tumor=0.41,
            val_dice_mean=0.61,
            lr=1e-3,
        )
    """

    # CSV 表头，顺序固定
    FIELDS = [
        "timestamp",
        "epoch",
        "train_loss",
        "val_loss",
        "val_dice_kidney",
        "val_dice_tumor",
        "val_dice_mean",
        "lr",
    ]

    def __init__(self, path: str):
        self.path = path

        # 文件不存在：直接创建并写入表头
        if not os.path.exists(path):
            self._create_new_log_file()
            print(f"[utils] 日志文件已创建: {path}")
            return

        # 文件存在：检查表头是否匹配
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)

        if existing_header == self.FIELDS:
            print(f"[utils] 日志文件已存在，将继续追加: {path}")
        else:
            # 旧版日志（无 lr 列）或表头不一致：自动备份旧文件，新建新文件
            backup_path = path + ".bak_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            os.rename(path, backup_path)
            self._create_new_log_file()
            print(f"[utils] 发现旧版日志表头不匹配，已备份为: {backup_path}")
            print(f"[utils] 已创建新版日志文件: {path}")

    def _create_new_log_file(self):
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writeheader()

    def write(self,
              epoch: int,
              train_loss: float,
              val_loss: float,
              val_dice_kidney: float,
              val_dice_tumor: float,
              val_dice_mean: float,
              lr: float | None = None):
        """
        追加写入一行日志。

        参数：
            lr: 学习率；为 None 时写空字符串，兼容旧调用
        """
        row = {
            "timestamp"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epoch"          : epoch,
            "train_loss"     : f"{train_loss:.6f}",
            "val_loss"       : f"{val_loss:.6f}",
            "val_dice_kidney": f"{val_dice_kidney:.6f}",
            "val_dice_tumor" : f"{val_dice_tumor:.6f}",
            "val_dice_mean"  : f"{val_dice_mean:.6f}",
            "lr"             : "" if lr is None else f"{lr:.8e}",
        }

        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writerow(row)


# -----------------------------------------------------------------------------
# 5. 自检
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from kits import config

    print("=" * 50)
    print("utils.py 自检")
    print("=" * 50)

    # 固定种子
    set_seed(config.SEED)

    # 创建输出目录
    ensure_dirs(config.CHECKPOINT_DIR, config.FIGURE_DIR, config.LOG_DIR)

    # 测试 CSVLogger
    log_path = os.path.join(config.LOG_DIR, "train_log.csv")
    logger = CSVLogger(log_path)
    logger.write(
        epoch=0,
        train_loss=0.9876,
        val_loss=0.9543,
        val_dice_kidney=0.1234,
        val_dice_tumor=0.0512,
        val_dice_mean=0.0873,
        lr=1e-3,
    )
    print(f"\n写入测试行成功，查看: {log_path}")

    # 测试 checkpoint 保存 / 加载（用一个假模型）
    dummy_model = torch.nn.Linear(4, 3)
    dummy_optim = torch.optim.Adam(dummy_model.parameters())
    ckpt_path   = os.path.join(config.CHECKPOINT_DIR, "test_ckpt.pth")

    save_checkpoint(
        state={
            "epoch"    : 0,
            "model"    : dummy_model.state_dict(),
            "optimizer": dummy_optim.state_dict(),
            "val_dice" : 0.1234,
        },
        path=ckpt_path,
    )

    new_model = torch.nn.Linear(4, 3)
    start_ep, best_dice = load_checkpoint(ckpt_path, new_model, device="cpu")
    print(f"\ncheckpoint 加载成功: start_epoch={start_ep}, best_dice={best_dice:.4f}")
    print("=" * 50)