"""
全局随机种子设置（Phase 5.5 Step 2）

背景：Phase 4/5 的真实数据实验里 `--seed` 只喂给合成数据生成器和固定池采样，
**从未绑定 torch / numpy 的全局 RNG**（全仓库无 torch.manual_seed）。
真实数据上 loader 输入与 seed 解耦，所以那 5 个 "seed" 实际是同一段数据上的
5 次不受控随机重复（gate/adapter 权重初始化不同），不是 5 条独立数据流。
既有 p 值只能在这个意义下解释；Phase 1 不受影响（无 torch 参数，std 已是 0）。
"""

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """设置 python / numpy / torch 的全局种子，使同一 seed 的 run 可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch 是硬依赖，仅防御
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - CPU-only 实验路径
        torch.cuda.manual_seed_all(seed)
