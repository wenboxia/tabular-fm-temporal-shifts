"""
Phase 3B 单元测试：FastToInterConsolidation
不依赖 TabPFN，纯 numpy + torch，可离线快速运行。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.consolidation.fast_to_inter import FastToInterConsolidation
from src.models.gated_ensemble import GatedEnsemble
from src.models.fast_corrector import FastCorrector


# 固定的测试超参
INPUT_DIM = 8
WINDOW = 20
EPOCHS = 30


def _make_consolidator(window=WINDOW, epochs=EPOCHS):
    return FastToInterConsolidation(
        threshold=0.05,
        window=window,
        epochs=epochs,
    )


def _make_model(input_dim=INPUT_DIM):
    return GatedEnsemble(input_dim=input_dim)


def _make_corrector(buffer_size=200):
    return FastCorrector(buffer_size=buffer_size, method="knn")


def _fill_corrector(corrector, n, input_dim=INPUT_DIM, error=0.3, rng=None):
    """向 corrector buffer 中压入 n 条固定误差的样本。"""
    if rng is None:
        rng = np.random.default_rng(0)
    for _ in range(n):
        x = rng.standard_normal(input_dim).astype(np.float32)
        corrector.update(x, error)


class TestFastToInterConsolidationBehavior:

    def test_consolidate_reduces_loss_and_matches_target(self):
        """
        buffer 压入 window 条 error==+0.3 的样本，
        跑 consolidate 后 final_loss 显著小于初始 loss，
        且 get_inter_prediction 的均值趋近 0.3（容差 0.05）。
        """
        torch.manual_seed(0)
        np.random.seed(0)
        rng = np.random.default_rng(0)

        model = _make_model()
        corrector = _make_corrector()
        consolidator = _make_consolidator()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # 构造固定 X_recent
        X_recent = rng.standard_normal((WINDOW, INPUT_DIM)).astype(np.float32)

        # 压入 WINDOW 条 error=+0.3 的样本（特征与 X_recent 一致）
        for i in range(WINDOW):
            corrector.update(X_recent[i], 0.3)

        # 计算初始 loss（未训练）
        X_tensor = torch.tensor(X_recent, dtype=torch.float32)
        target = torch.full((WINDOW,), 0.3)
        model.eval()
        with torch.no_grad():
            y_inter_init = model.get_inter_prediction(X_tensor)  # (WINDOW, 1)
            initial_loss = F.mse_loss(y_inter_init.squeeze(-1), target).item()

        # 执行巩固
        model.train()
        final_loss = consolidator.consolidate(
            gated_ensemble=model,
            fast_corrector=corrector,
            X_recent=X_recent,
            optimizer=optimizer,
        )

        # final_loss 必须显著小于初始 loss
        assert final_loss < initial_loss, (
            f"final_loss={final_loss:.4f} 应 < initial_loss={initial_loss:.4f}"
        )

        # adapter 输出均值趋近目标 0.3
        model.eval()
        with torch.no_grad():
            y_inter_after = model.get_inter_prediction(X_tensor)  # (WINDOW, 1)
        mean_pred = y_inter_after.squeeze(-1).mean().item()
        assert abs(mean_pred - 0.3) < 0.05, (
            f"adapter 输出均值={mean_pred:.4f}，期望接近 0.3（容差 0.05）"
        )

    def test_consolidate_clears_buffer(self):
        """巩固完成后 fast_corrector.buffer 应当为空（len == 0）。"""
        torch.manual_seed(0)
        np.random.seed(0)
        rng = np.random.default_rng(0)

        model = _make_model()
        corrector = _make_corrector()
        consolidator = _make_consolidator()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        X_recent = rng.standard_normal((WINDOW, INPUT_DIM)).astype(np.float32)
        _fill_corrector(corrector, WINDOW, rng=rng)

        assert len(corrector.buffer) >= WINDOW  # 前置确认

        consolidator.consolidate(
            gated_ensemble=model,
            fast_corrector=corrector,
            X_recent=X_recent,
            optimizer=optimizer,
        )

        assert len(corrector.buffer) == 0, (
            f"巩固后 buffer 应为空，当前长度: {len(corrector.buffer)}"
        )

    def test_consolidate_asserts_on_short_buffer(self):
        """buffer 只有 window-1 条时，consolidate 应抛出 AssertionError。"""
        torch.manual_seed(0)
        np.random.seed(0)
        rng = np.random.default_rng(0)

        model = _make_model()
        corrector = _make_corrector()
        consolidator = _make_consolidator()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # 压入 window-1 条，比 window 少一条
        _fill_corrector(corrector, WINDOW - 1, rng=rng)

        X_recent = rng.standard_normal((WINDOW, INPUT_DIM)).astype(np.float32)

        with pytest.raises(AssertionError):
            consolidator.consolidate(
                gated_ensemble=model,
                fast_corrector=corrector,
                X_recent=X_recent,
                optimizer=optimizer,
            )

    def test_consolidate_asserts_on_wrong_X_shape(self):
        """X_recent.shape[0] != window 时，consolidate 应抛出 AssertionError。"""
        torch.manual_seed(0)
        np.random.seed(0)
        rng = np.random.default_rng(0)

        model = _make_model()
        corrector = _make_corrector()
        consolidator = _make_consolidator()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        _fill_corrector(corrector, WINDOW, rng=rng)

        # X_recent 行数与 window 不同（window+1 行）
        X_wrong = rng.standard_normal((WINDOW + 1, INPUT_DIM)).astype(np.float32)

        with pytest.raises(AssertionError):
            consolidator.consolidate(
                gated_ensemble=model,
                fast_corrector=corrector,
                X_recent=X_wrong,
                optimizer=optimizer,
            )

    def test_integration_driven_by_should_consolidate(self):
        """
        压入 60 条偏置误差 +0.25，验证 should_consolidate(window=20, bias_threshold=0.05)
        返回 True，然后 consolidate 能正常执行且 buffer 清空。
        """
        torch.manual_seed(0)
        np.random.seed(0)
        rng = np.random.default_rng(0)

        model = _make_model()
        corrector = _make_corrector(buffer_size=200)
        consolidator = _make_consolidator()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # 压入 60 条偏置为 +0.25 的样本
        n_push = 60
        X_all = rng.standard_normal((n_push, INPUT_DIM)).astype(np.float32)
        for i in range(n_push):
            corrector.update(X_all[i], 0.25)

        # should_consolidate 的签名：(window=50, bias_threshold=0.05)
        should = corrector.should_consolidate(window=WINDOW, bias_threshold=0.05)
        assert should, (
            "压入 60 条偏置 +0.25 的误差后，should_consolidate 应返回 True"
        )

        # 取最近 WINDOW 步作为 X_recent
        X_recent = X_all[-WINDOW:].copy()

        # consolidate 应成功执行
        final_loss = consolidator.consolidate(
            gated_ensemble=model,
            fast_corrector=corrector,
            X_recent=X_recent,
            optimizer=optimizer,
        )

        # 验证返回值是合法浮点数
        assert isinstance(final_loss, float), (
            f"final_loss 应为 float，收到: {type(final_loss)}"
        )
        assert final_loss >= 0.0, f"MSE loss 不应为负，收到: {final_loss}"

        # buffer 必须已清空
        assert len(corrector.buffer) == 0, (
            f"巩固后 buffer 应为空，当前长度: {len(corrector.buffer)}"
        )
