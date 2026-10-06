"""
合成数据生成器单元测试（不依赖 TabPFN，可离线运行）
"""

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.synthetic import (
    make_rotating_boundary,
    make_regime_switching,
    make_combined_drift,
    make_dataset,
)
from src.data.temporal_loader import TemporalWindowLoader


class TestRotatingBoundary:
    def test_output_shapes(self):
        ds = make_rotating_boundary(n_samples=500, n_features=2)
        assert ds.X.shape == (500, 2)
        assert ds.y.shape == (500,)
        assert ds.regime_labels.shape == (500,)

    def test_binary_labels(self):
        ds = make_rotating_boundary(n_samples=500)
        assert set(np.unique(ds.y)).issubset({0, 1})

    def test_dtype(self):
        ds = make_rotating_boundary(n_samples=100)
        assert ds.X.dtype == np.float32

    def test_has_drift_points(self):
        ds = make_rotating_boundary(n_samples=5000, drift_speed=0.003)
        assert len(ds.drift_points) > 0

    def test_reproducible(self):
        ds1 = make_rotating_boundary(n_samples=200, random_seed=42)
        ds2 = make_rotating_boundary(n_samples=200, random_seed=42)
        np.testing.assert_array_equal(ds1.X, ds2.X)
        np.testing.assert_array_equal(ds1.y, ds2.y)

    def test_different_seeds(self):
        ds1 = make_rotating_boundary(n_samples=200, random_seed=1)
        ds2 = make_rotating_boundary(n_samples=200, random_seed=2)
        assert not np.array_equal(ds1.X, ds2.X)


class TestRegimeSwitching:
    def test_output_shapes(self):
        ds = make_regime_switching(n_samples=600, n_features=5, n_regimes=3)
        assert ds.X.shape == (600, 5)
        assert ds.y.shape == (600,)

    def test_regime_labels_correct(self):
        ds = make_regime_switching(n_samples=600, n_regimes=3, regime_length=200)
        # 前 200 步是体制 0，200-400 是体制 1，400-600 是体制 2
        assert all(ds.regime_labels[:200] == 0)
        assert all(ds.regime_labels[200:400] == 1)
        assert all(ds.regime_labels[400:600] == 2)

    def test_drift_points_at_boundaries(self):
        ds = make_regime_switching(n_samples=600, n_regimes=3, regime_length=200)
        assert 200 in ds.drift_points
        assert 400 in ds.drift_points

    def test_name(self):
        ds = make_regime_switching()
        assert ds.name == "regime_switching"


class TestCombinedDrift:
    def test_output_shapes(self):
        ds = make_combined_drift(n_samples=500, n_features=8)
        assert ds.X.shape == (500, 8)
        assert ds.y.shape == (500,)

    def test_has_drift_points(self):
        ds = make_combined_drift(n_samples=5000)
        assert len(ds.drift_points) > 0

    def test_feature_drift_visible(self):
        ds = make_combined_drift(n_samples=6000, n_features=10, noise=0.0)
        # 前 5 个特征均值应随时间增大
        early_mean = ds.X[:200, :5].mean()
        late_mean = ds.X[-200:, :5].mean()
        assert late_mean > early_mean  # 均值漂移应可观测


class TestMakeDataset:
    def test_known_names(self):
        for name in ["rotating_boundary", "regime_switching", "combined_drift"]:
            ds = make_dataset(name, n_samples=200)
            assert ds.name == name

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="未知数据集"):
            make_dataset("nonexistent_dataset")


class TestTemporalWindowLoader:
    def setup_method(self):
        rng = np.random.default_rng(0)
        self.X = rng.standard_normal((1000, 5)).astype(np.float32)
        self.y = rng.integers(0, 2, size=1000)

    def test_length(self):
        loader = TemporalWindowLoader(self.X, self.y, context_size=100, step_size=1)
        assert len(loader) == 900  # 1000 - 100

    def test_step_size_2(self):
        loader = TemporalWindowLoader(self.X, self.y, context_size=100, step_size=2)
        assert len(loader) == 450  # (900) / 2

    def test_batch_shapes(self):
        loader = TemporalWindowLoader(self.X, self.y, context_size=100)
        batch = next(iter(loader))
        assert batch.X_ctx.shape == (100, 5)
        assert batch.y_ctx.shape == (100,)
        assert batch.X_query.shape == (1, 5)
        assert batch.y_query.shape == (1,)

    def test_first_batch_time(self):
        loader = TemporalWindowLoader(self.X, self.y, context_size=100)
        batch = next(iter(loader))
        assert batch.t == 100

    def test_context_does_not_overlap_query(self):
        loader = TemporalWindowLoader(self.X, self.y, context_size=50)
        batch = loader.get_batch(100)
        # 上下文是 [50, 100)，查询是 [100, 101)
        np.testing.assert_array_equal(batch.X_ctx, self.X[50:100])
        np.testing.assert_array_equal(batch.X_query, self.X[100:101])

    def test_iteration_count(self):
        loader = TemporalWindowLoader(self.X, self.y, context_size=100)
        count = sum(1 for _ in loader)
        assert count == len(loader)

    def test_length_mismatch_raises(self):
        with pytest.raises(AssertionError):
            TemporalWindowLoader(self.X, self.y[:900], context_size=100)
