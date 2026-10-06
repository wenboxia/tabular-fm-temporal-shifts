"""
Phase 2 单元测试：WorkingMemoryBuffer + FastCorrector
不依赖 TabPFN，纯 numpy，可离线快速运行。
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.memory.buffer import WorkingMemoryBuffer
from src.models.fast_corrector import FastCorrector


# ===========================================================
# WorkingMemoryBuffer
# ===========================================================

class TestWorkingMemoryBuffer:

    def test_push_and_len(self):
        buf = WorkingMemoryBuffer(capacity=10)
        assert len(buf) == 0
        buf.push(np.ones(4), 0.5)
        assert len(buf) == 1

    def test_fifo_capacity(self):
        buf = WorkingMemoryBuffer(capacity=3)
        for i in range(5):
            buf.push(np.array([float(i)]), float(i))
        assert len(buf) == 3
        # 只剩最新的 3 个：2, 3, 4
        recent = buf.recent_errors(3)
        np.testing.assert_array_equal(recent, [2.0, 3.0, 4.0])

    def test_clear(self):
        buf = WorkingMemoryBuffer(capacity=10)
        buf.push(np.ones(3), 1.0)
        buf.clear()
        assert buf.is_empty()
        assert len(buf) == 0

    def test_knn_empty_returns_zero(self):
        buf = WorkingMemoryBuffer(capacity=10)
        assert buf.knn_correction(np.zeros(4)) == 0.0

    def test_knn_exact_match(self):
        buf = WorkingMemoryBuffer(capacity=10)
        x = np.array([1.0, 0.0, 0.0])
        buf.push(x, 0.3)
        buf.push(np.array([10.0, 10.0, 10.0]), -0.9)   # 很远，不应被选中
        # 查询与第一个完全相同
        result = buf.knn_correction(x, k=1)
        assert abs(result - 0.3) < 1e-6

    def test_knn_k_larger_than_buffer(self):
        buf = WorkingMemoryBuffer(capacity=10)
        buf.push(np.ones(3), 0.2)
        buf.push(np.ones(3) * 2, 0.4)
        # k=5 但缓冲区只有 2 个，应自动截断
        result = buf.knn_correction(np.ones(3), k=5)
        assert isinstance(result, float)

    def test_ema_empty_returns_zero(self):
        buf = WorkingMemoryBuffer(capacity=10)
        assert buf.ema_correction() == 0.0

    def test_ema_single_entry(self):
        buf = WorkingMemoryBuffer(capacity=10)
        buf.push(np.zeros(3), 0.7)
        assert abs(buf.ema_correction(alpha=0.1) - 0.7) < 1e-6

    def test_ema_recent_bias(self):
        """EMA 应对最新误差赋予更高权重。"""
        buf = WorkingMemoryBuffer(capacity=10)
        buf.push(np.zeros(3), 0.0)   # 旧的：误差 0
        buf.push(np.zeros(3), 1.0)   # 新的：误差 1
        ema = buf.ema_correction(alpha=0.9)
        # alpha=0.9 时，最新误差权重 >> 旧误差，结果应接近 1
        assert ema > 0.5

    def test_mean_recent_error(self):
        buf = WorkingMemoryBuffer(capacity=20)
        for _ in range(10):
            buf.push(np.zeros(3), 0.5)
        assert abs(buf.mean_recent_error(10) - 0.5) < 1e-6

    def test_std_recent_error(self):
        buf = WorkingMemoryBuffer(capacity=20)
        buf.push(np.zeros(3), 0.0)
        buf.push(np.zeros(3), 1.0)
        std = buf.std_recent_error(2)
        assert std > 0

    def test_feature_stored_as_float32(self):
        buf = WorkingMemoryBuffer(capacity=5)
        x = np.array([1, 2, 3], dtype=np.float64)
        buf.push(x, 0.0)
        # 内部存储应为 float32
        stored = buf._features[0]
        assert stored.dtype == np.float32


# ===========================================================
# FastCorrector
# ===========================================================

class TestFastCorrector:

    def test_invalid_method_raises(self):
        with pytest.raises(AssertionError):
            FastCorrector(method="invalid")

    def test_correct_empty_returns_zero(self):
        fc = FastCorrector(buffer_size=10, method="knn")
        assert fc.correct(np.zeros(5)) == 0.0

    def test_correct_knn_after_updates(self):
        fc = FastCorrector(buffer_size=20, method="knn", k=3)
        x_pos = np.array([1.0, 0.0])
        x_neg = np.array([0.0, 1.0])
        # 训练：正方向的 x 误差为 +0.3
        for _ in range(5):
            fc.update(x_pos, 0.3)
        # 查询正方向 x，应得到接近 +0.3 的校正
        result = fc.correct(x_pos)
        assert abs(result - 0.3) < 0.05

    def test_correct_ema_tracks_recent_error(self):
        fc = FastCorrector(buffer_size=20, method="ema", alpha=0.5)
        for _ in range(5):
            fc.update(np.zeros(3), 0.0)
        fc.update(np.zeros(3), 0.8)   # 最新误差 0.8
        result = fc.correct(np.zeros(3))
        # EMA(alpha=0.5) 对最新值权重高，结果应明显 > 0
        assert result > 0.2

    def test_reset_clears_buffer(self):
        fc = FastCorrector(buffer_size=10, method="knn")
        fc.update(np.ones(3), 0.5)
        fc.reset()
        assert fc.buffer.is_empty()
        assert fc.correct(np.ones(3)) == 0.0

    def test_should_consolidate_false_when_empty(self):
        fc = FastCorrector(buffer_size=100)
        assert not fc.should_consolidate(window=50)

    def test_should_consolidate_false_below_threshold(self):
        fc = FastCorrector(buffer_size=200)
        # 误差很小，不应触发巩固
        for _ in range(60):
            fc.update(np.zeros(3), 0.001)
        assert not fc.should_consolidate(window=50, bias_threshold=0.05)

    def test_should_consolidate_true_when_biased(self):
        fc = FastCorrector(buffer_size=200)
        # 误差一致偏向正方向
        for _ in range(60):
            fc.update(np.zeros(3), 0.2)   # 均值 0.2，远大于 threshold=0.05
        assert fc.should_consolidate(window=50, bias_threshold=0.05)

    def test_correction_clipped_in_prequential_loop(self):
        """模拟 prequential 循环：验证校正后的概率被正确 clip 到 [0,1]。"""
        fc = FastCorrector(buffer_size=50, method="ema", alpha=0.2)

        preds, labels_arr = [], []
        rng = np.random.default_rng(0)

        for t in range(100):
            x = rng.standard_normal(5).astype(np.float32)
            y_true = int(rng.random() < 0.7)
            y_slow = 0.5  # 模拟 TabPFN 输出

            correction = fc.correct(x)
            y_final = float(np.clip(y_slow + correction, 0.0, 1.0))

            # 确保概率在合法范围内
            assert 0.0 <= y_final <= 1.0

            error = float(y_true) - y_slow
            fc.update(x, error)

            preds.append(int(y_final >= 0.5))
            labels_arr.append(y_true)

        acc = np.mean(np.array(preds) == np.array(labels_arr))
        # 在随机数据上不要求高准确率，只验证循环能跑通
        assert 0.0 <= acc <= 1.0

    def test_should_consolidate_true_under_noise(self):
        """
        压入 60 条交替 0.0 / 0.4 的 errors（mean=0.2, std≈0.2）：
        旧规则 std < |mean| 会返回 False；新规则只看 |mean|>threshold，应返回 True。
        这是本次修改的核心断言。
        """
        fc = FastCorrector(buffer_size=200, method="knn")
        rng = np.random.default_rng(0)
        for i in range(60):
            x = rng.standard_normal(8).astype(np.float32)
            err = 0.0 if i % 2 == 0 else 0.4
            fc.update(x, err)
        assert fc.should_consolidate(window=50, bias_threshold=0.05) == True

    def test_repr(self):
        fc = FastCorrector(buffer_size=50, method="knn")
        s = repr(fc)
        assert "knn" in s
        assert "50" in s
