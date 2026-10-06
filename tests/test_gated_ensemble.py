"""
Phase 3A 单元测试：GatedEnsemble（软门控融合网络）
不依赖 TabPFN，纯 torch，可离线快速运行。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest

from src.models.gated_ensemble import GatedEnsemble


# ===========================================================
# GatedEnsemble
# ===========================================================

class TestGatedEnsembleBehavior:

    def test_forward_output_shapes(self):
        """y_final shape == (B, n_outputs)，weights shape == (B, 3)。"""
        torch.manual_seed(42)
        B, input_dim, n_outputs = 8, 10, 1
        model = GatedEnsemble(input_dim=input_dim, hidden_dim=64, n_outputs=n_outputs)

        x      = torch.randn(B, input_dim)
        y_slow = torch.rand(B, n_outputs)
        correction = torch.rand(B, n_outputs)

        y_final, weights = model(x, y_slow, correction)

        assert y_final.shape == (B, n_outputs), (
            f"y_final shape 应为 ({B}, {n_outputs})，实为 {tuple(y_final.shape)}"
        )
        assert weights.shape == (B, 3), (
            f"weights shape 应为 ({B}, 3)，实为 {tuple(weights.shape)}"
        )

    def test_weights_sum_to_one(self):
        """weights.sum(dim=-1) 每一行应 == 1.0（容差 1e-6）。"""
        torch.manual_seed(42)
        B, input_dim = 8, 10
        model = GatedEnsemble(input_dim=input_dim, hidden_dim=64, n_outputs=1)

        x      = torch.randn(B, input_dim)
        y_slow = torch.rand(B, 1)
        correction = torch.rand(B, 1)

        _, weights = model(x, y_slow, correction)
        row_sums = weights.sum(dim=-1)  # (B,)

        for i, s in enumerate(row_sums):
            assert abs(s.item() - 1.0) < 1e-6, (
                f"第 {i} 行权重和为 {s.item():.8f}，期望 1.0"
            )

    def test_gate_weights_shape_via_helper(self):
        """get_gate_weights(x) 返回 (B, 3)。"""
        torch.manual_seed(42)
        B, input_dim = 8, 10
        model = GatedEnsemble(input_dim=input_dim, hidden_dim=64, n_outputs=1)

        x = torch.randn(B, input_dim)
        weights = model.get_gate_weights(x)

        assert weights.shape == (B, 3), (
            f"get_gate_weights 返回 shape 应为 ({B}, 3)，实为 {tuple(weights.shape)}"
        )

    def test_gradient_flows(self):
        """loss.backward() 后 gate 和 adapter 的参数 .grad 都不是 None。"""
        torch.manual_seed(42)
        B, input_dim = 8, 10
        model = GatedEnsemble(input_dim=input_dim, hidden_dim=64, n_outputs=1)

        x      = torch.randn(B, input_dim)
        y_slow = torch.rand(B, 1)
        correction = torch.rand(B, 1)
        y_true = torch.randint(0, 2, (B, 1)).float()

        y_final, _ = model(x, y_slow, correction)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_final, y_true)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, (
                f"参数 '{name}' 的 grad 为 None，梯度未流过"
            )

    def test_adapter_can_output_negative(self):
        """get_inter_prediction(x) 的输出分布中应有部分 < 0（adapter 无 sigmoid 约束）。"""
        torch.manual_seed(42)
        B, input_dim = 64, 10
        model = GatedEnsemble(input_dim=input_dim, hidden_dim=64, n_outputs=1)

        x = torch.randn(B, input_dim)
        y_inter = model.get_inter_prediction(x)  # (B, 1)

        has_negative = (y_inter < 0).any().item()
        assert has_negative, (
            "adapter 输出中未发现负值；怀疑最后一层被激活函数（如 sigmoid）夹住了"
        )

    def test_eval_mode_works(self):
        """model.eval() 下 forward 不崩，输出 shape 正确。"""
        torch.manual_seed(42)
        B, input_dim = 8, 10
        model = GatedEnsemble(input_dim=input_dim, hidden_dim=64, n_outputs=1)
        model.eval()

        x      = torch.randn(B, input_dim)
        y_slow = torch.rand(B, 1)
        correction = torch.rand(B, 1)

        with torch.no_grad():
            y_final, weights = model(x, y_slow, correction)

        assert y_final.shape == (B, 1)
        assert weights.shape == (B, 3)

    def test_y_final_raw_can_be_outside_unit_interval(self):
        """v2 residual-additive fusion：forward 返回的 y_final_raw 不做 clamp。

        构造大正 correction → 结果应 > 1；
        构造大负 correction → 结果应 < 0。
        两者都成立才能证明 forward 内没有隐式 clamp。
        """
        torch.manual_seed(42)
        input_dim = 10
        model = GatedEnsemble(input_dim=input_dim, hidden_dim=64, n_outputs=1)
        model.eval()

        x = torch.randn(1, input_dim)

        # 大正 correction：y_slow=0.9，correction=+5.0 → 期望 y_final_raw > 1
        y_slow_high = torch.tensor([[0.9]])
        correction_large_pos = torch.tensor([[5.0]])
        with torch.no_grad():
            y_raw_high, _ = model(x, y_slow_high, correction_large_pos)
        assert y_raw_high.item() > 1.0, (
            f"期望 y_final_raw > 1.0（未 clamp），实为 {y_raw_high.item():.4f}"
        )

        # 大负 correction：y_slow=0.1，correction=-5.0 → 期望 y_final_raw < 0
        y_slow_low = torch.tensor([[0.1]])
        correction_large_neg = torch.tensor([[-5.0]])
        with torch.no_grad():
            y_raw_low, _ = model(x, y_slow_low, correction_large_neg)
        assert y_raw_low.item() < 0.0, (
            f"期望 y_final_raw < 0.0（未 clamp），实为 {y_raw_low.item():.4f}"
        )
