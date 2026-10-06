"""Phase 5.5 Step 2 单元测试：全局种子绑定。

背景：Phase 4/5 真实数据实验的 --seed 只喂合成数据生成器，从未绑定 torch，
所以真实数据上的 5 个 seed 是同一段数据的 5 次不受控随机重复。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from src.utils.seeding import set_global_seed


def _draw():
    return (
        float(np.random.rand()),
        float(torch.rand(1).item()),
        [float(p.flatten()[0]) for p in torch.nn.Linear(4, 4).parameters()],
    )


class TestSetGlobalSeed:

    def test_same_seed_reproduces_numpy_torch_and_init(self):
        set_global_seed(123)
        a = _draw()
        set_global_seed(123)
        b = _draw()
        assert a == b, "同一 seed 应完全复现 numpy / torch / 模块初始化"

    def test_different_seed_diverges(self):
        set_global_seed(1)
        a = _draw()
        set_global_seed(2)
        b = _draw()
        assert a != b, "不同 seed 应产生不同随机流"

    def test_model_init_reproducible(self):
        """MultiTimescaleModel 的 gate/adapter 初始化在同一 seed 下一致。"""
        from src.models.multi_timescale import MultiTimescaleModel

        def first_weights(seed):
            set_global_seed(seed)
            m = MultiTimescaleModel(input_dim=4)
            return next(m.gated_ensemble.gate.parameters()).detach().clone()

        assert torch.equal(first_weights(7), first_weights(7))
        assert not torch.equal(first_weights(7), first_weights(8))
