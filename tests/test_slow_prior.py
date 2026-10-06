"""
SlowPrior (TabPFN 包装器) 单元测试

注意：这些测试需要安装 tabpfn。
若未安装，tabpfn 相关测试将被跳过。
数据形状和边缘情况测试使用 mock，不需要真实模型。
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestSlowPriorInterface:
    """测试 SlowPrior 的对外接口和边缘情况，使用 mock。"""

    def setup_method(self):
        rng = np.random.default_rng(42)
        self.X_ctx = rng.standard_normal((200, 5)).astype(np.float32)
        self.y_ctx = rng.integers(0, 2, size=200)
        self.X_query = rng.standard_normal((3, 5)).astype(np.float32)

    def _make_mock_tabpfn(self):
        """创建一个模拟的 TabPFNClassifier。"""
        mock_clf = MagicMock()
        # predict_proba 返回合理的概率矩阵
        n_query = len(self.X_query)
        mock_clf.predict_proba.return_value = np.array(
            [[0.7, 0.3]] * n_query, dtype=float
        )
        return mock_clf

    def _make_prior_with_mock(self):
        """创建一个注入了 mock TabPFN 的 SlowPrior，不依赖真实安装。"""
        from src.models.slow_prior import SlowPrior
        prior = SlowPrior()
        prior._model = self._make_mock_tabpfn()
        return prior

    def test_predict_returns_proba_and_labels(self):
        prior = self._make_prior_with_mock()
        proba, labels = prior.predict(self.X_ctx, self.y_ctx, self.X_query)
        assert proba.shape == (3, 2)
        assert labels.shape == (3,)

    def test_proba_sums_to_one(self):
        prior = self._make_prior_with_mock()
        proba, _ = prior.predict(self.X_ctx, self.y_ctx, self.X_query)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(3), atol=1e-6)

    def test_labels_are_argmax_of_proba(self):
        prior = self._make_prior_with_mock()
        proba, labels = prior.predict(self.X_ctx, self.y_ctx, self.X_query)
        np.testing.assert_array_equal(labels, np.argmax(proba, axis=1))

    def test_single_class_context_fallback(self):
        """上下文只有一类时，不应崩溃，应返回多数类（此处不需要模型）。"""
        from src.models.slow_prior import SlowPrior
        prior = SlowPrior()
        # 注入 mock，但单类情况会在调用模型前提前返回
        prior._model = self._make_mock_tabpfn()
        y_all_zeros = np.zeros(200, dtype=int)
        proba, labels = prior.predict(self.X_ctx, y_all_zeros, self.X_query)
        assert proba.shape == (3, 2)
        assert labels.shape == (3,)
        assert all(labels == 0)

    def test_predict_proba_convenience(self):
        prior = self._make_prior_with_mock()
        proba = prior.predict_proba(self.X_ctx, self.y_ctx, self.X_query)
        assert proba.shape == (3, 2)

    def test_predict_label_convenience(self):
        prior = self._make_prior_with_mock()
        labels = prior.predict_label(self.X_ctx, self.y_ctx, self.X_query)
        assert labels.shape == (3,)
        assert set(labels).issubset({0, 1})

    def test_no_fine_tune_method(self):
        """SlowPrior 不应有 fine_tune 方法（永远冻结）。"""
        from src.models.slow_prior import SlowPrior
        prior = SlowPrior()
        assert not hasattr(prior, "fine_tune"), (
            "SlowPrior 不应有 fine_tune 方法 —— TabPFN 永远冻结"
        )

    def test_import_error_without_tabpfn(self):
        """未安装 tabpfn 时，调用 predict 应给出友好错误信息。"""
        from src.models.slow_prior import SlowPrior

        prior = SlowPrior()
        prior._model = None

        with patch.dict("sys.modules", {"tabpfn": None}):
            with pytest.raises((ImportError, TypeError)):
                prior.predict(self.X_ctx, self.y_ctx, self.X_query)


class TestMetrics:
    """评估指标单元测试。"""

    def test_prequential_accuracy_perfect(self):
        from src.utils.metrics import prequential_accuracy
        preds = np.array([0, 1, 0, 1, 1])
        labels = np.array([0, 1, 0, 1, 1])
        assert prequential_accuracy(preds, labels) == 1.0

    def test_prequential_accuracy_zero(self):
        from src.utils.metrics import prequential_accuracy
        preds = np.array([1, 0, 1, 0])
        labels = np.array([0, 1, 0, 1])
        assert prequential_accuracy(preds, labels) == 0.0

    def test_prequential_accuracy_half(self):
        from src.utils.metrics import prequential_accuracy
        preds = np.array([0, 0, 1, 1])
        labels = np.array([0, 1, 0, 1])
        assert prequential_accuracy(preds, labels) == pytest.approx(0.5)

    def test_window_accuracy_shape(self):
        from src.utils.metrics import window_accuracy
        preds = np.zeros(100, dtype=int)
        labels = np.zeros(100, dtype=int)
        accs = window_accuracy(preds, labels, window_size=10)
        assert len(accs) == 91  # 100 - 10 + 1

    def test_window_accuracy_all_correct(self):
        from src.utils.metrics import window_accuracy
        preds = np.array([0] * 50)
        labels = np.array([0] * 50)
        accs = window_accuracy(preds, labels, window_size=10)
        np.testing.assert_allclose(accs, 1.0)

    def test_adaptation_speed_immediate_recovery(self):
        from src.utils.metrics import adaptation_speed
        # 前 100 步正确，模拟漂移后立即恢复
        preds = np.ones(200, dtype=int)
        labels = np.ones(200, dtype=int)
        speeds = adaptation_speed(preds, labels, drift_points=[100], baseline_acc=0.9, offset=0)
        assert speeds[100] == 0  # 立即恢复

    def test_adaptation_speed_out_of_range(self):
        from src.utils.metrics import adaptation_speed
        preds = np.ones(100, dtype=int)
        labels = np.ones(100, dtype=int)
        # 漂移点超出范围
        speeds = adaptation_speed(preds, labels, drift_points=[200], baseline_acc=0.9, offset=0)
        assert speeds[200] is None
