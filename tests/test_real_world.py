"""
Phase 5 真实数据 loader 单元测试

跑这些 test 需要：
  - openml 已装且 OpenML 151 已 cache（首次会下载 ~3 MB）
  - Insects abrupt_balanced.csv 已 cache 至 ~/.cache/insects_drift/
    （首次会从 Google Drive 下载 ~14 MB）

CI 跳过这些 test：用 -m "not network" 或环境变量门控。
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from src.data.real_world import (
    RealWorldDataset,
    _INSECTS_BINARIZE_MAP,
    _segment_bounds,
    load_electricity,
    load_insects,
    load_real_world,
    take_segment,
)


def _has_insects_cache() -> bool:
    return os.path.exists(os.path.expanduser("~/.cache/insects_drift/abrupt_balanced.csv"))


# ---------------------------------------------------------------------------
# 工具函数：take_segment / _segment_bounds
# ---------------------------------------------------------------------------


def test_segment_bounds_three_modes():
    assert _segment_bounds(10000, "start", 5000) == (0, 5000)
    assert _segment_bounds(10000, "end", 5000) == (5000, 10000)
    assert _segment_bounds(10000, "middle", 5000) == (2500, 7500)


def test_segment_bounds_rejects_invalid():
    with pytest.raises(ValueError):
        _segment_bounds(100, "start", 200)  # size > n
    with pytest.raises(ValueError):
        _segment_bounds(1000, "left", 100)  # bad segment_id


def test_take_segment_shapes_and_no_overlap():
    X = np.arange(30000, dtype=np.float32).reshape(-1, 1)
    y = np.arange(30000, dtype=np.int64)
    Xs, ys = take_segment(X, y, "start", size=5000)
    Xm, ym = take_segment(X, y, "middle", size=5000)
    Xe, ye = take_segment(X, y, "end", size=5000)
    assert Xs.shape == Xm.shape == Xe.shape == (5000, 1)
    # 三段不重叠：用 y 作为 unique index
    s_idx, m_idx, e_idx = set(ys.tolist()), set(ym.tolist()), set(ye.tolist())
    assert s_idx & m_idx == set()
    assert m_idx & e_idx == set()
    assert s_idx & e_idx == set()


def test_take_segment_preserves_order():
    X = np.arange(20000, dtype=np.float32).reshape(-1, 1)
    y = np.arange(20000, dtype=np.int64)
    Xs, _ = take_segment(X, y, "middle", size=4000)
    # 时序保持递增
    assert np.all(np.diff(Xs.flatten()) == 1)


# ---------------------------------------------------------------------------
# Insects loader（依赖 cache）
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_insects_cache(), reason="Insects CSV not cached")
def test_insects_load_basic_shape():
    ds = load_insects(segment_id="start", size=5000)
    assert isinstance(ds, RealWorldDataset)
    assert ds.X.shape == (5000, 33)
    assert ds.y.shape == (5000,)
    assert ds.X.dtype == np.float32
    assert ds.y.dtype == np.int64
    assert ds.name == "insects_abrupt_balanced_start"


@pytest.mark.skipif(not _has_insects_cache(), reason="Insects CSV not cached")
def test_insects_binary_labels():
    ds = load_insects(segment_id="start", size=5000)
    # 二值化后只允许 {0, 1}
    assert set(np.unique(ds.y).tolist()) <= {0, 1}
    # binarize map 必须覆盖原始 6 个 ID
    assert sorted(_INSECTS_BINARIZE_MAP.keys()) == [2, 3, 4, 5, 11, 12]
    # sex-pair 配对：每对 (2,3) (4,5) (11,12) 跨越 0/1
    for a, b in [(2, 3), (4, 5), (11, 12)]:
        assert _INSECTS_BINARIZE_MAP[a] != _INSECTS_BINARIZE_MAP[b]


@pytest.mark.skipif(not _has_insects_cache(), reason="Insects CSV not cached")
def test_insects_normalization_no_leak():
    ds = load_insects(segment_id="start", size=5000)
    # fit 在前 200 上 → 前 200 的 mean ≈ 0, std ≈ 1
    head_mean = ds.X[:200].mean(axis=0)
    head_std = ds.X[:200].std(axis=0)
    assert np.all(np.abs(head_mean) < 1e-4), f"head mean off: {head_mean.max():.4g}"
    assert np.all(np.abs(head_std - 1.0) < 1e-2), (
        f"head std off: min={head_std.min():.4g} max={head_std.max():.4g}"
    )
    # 后续段的 mean 不会被强制为 0（如果是会泄漏）
    tail_mean = ds.X[200:].mean(axis=0)
    assert np.any(np.abs(tail_mean) > 1e-3), (
        "全段 mean 太接近 0 → 怀疑 fit 用了全段，泄漏 future stats"
    )


@pytest.mark.skipif(not _has_insects_cache(), reason="Insects CSV not cached")
def test_insects_segments_no_overlap_via_X_signature():
    """三个 segment 用前 5 维特征作 signature 比较应不重合（每个 segment 5000 行）。"""
    s = load_insects("start", 5000)
    m = load_insects("middle", 5000)
    e = load_insects("end", 5000)
    # 通过 raw float vector 把行 hash 成元组检测重叠
    sig_s = {tuple(row) for row in s.X[:, :5].round(5).tolist()}
    sig_m = {tuple(row) for row in m.X[:, :5].round(5).tolist()}
    sig_e = {tuple(row) for row in e.X[:, :5].round(5).tolist()}
    # 任意两段交集为空（normalization 后的 X 仍能识别行身份，因为不同 segment fit 的 scaler 不同
    # → 每段 normalization 后行向量不同；用 raw index 更稳，但 take_segment 已 test）
    # 这里仅要求显著不重合：交集占比 < 1%
    assert len(sig_s & sig_m) / 5000 < 0.01
    assert len(sig_m & sig_e) / 5000 < 0.01
    assert len(sig_s & sig_e) / 5000 < 0.01


@pytest.mark.skipif(not _has_insects_cache(), reason="Insects CSV not cached")
def test_insects_drift_points_within_bounds():
    """每个 segment 的 drift_points（如有）必须在 [0, size)。"""
    for seg in ["start", "middle", "end"]:
        ds = load_insects(segment_id=seg, size=5000)
        assert all(0 <= d < 5000 for d in ds.drift_points), (
            f"{seg}: drift points out of [0, 5000): {ds.drift_points}"
        )


# ---------------------------------------------------------------------------
# Electricity loader（依赖 OpenML 网络）
# ---------------------------------------------------------------------------


@pytest.mark.network
@pytest.mark.skipif(
    "OPENML_OFFLINE" in os.environ, reason="OpenML offline mode set in env"
)
def test_electricity_load_basic_shape():
    ds = load_electricity(segment_id="start", size=5000)
    assert isinstance(ds, RealWorldDataset)
    assert ds.X.shape[0] == 5000
    assert ds.X.dtype == np.float32
    assert ds.y.dtype == np.int64
    assert set(np.unique(ds.y).tolist()) == {0, 1}
    assert ds.drift_points == []
    assert ds.name == "electricity_start"


@pytest.mark.network
@pytest.mark.skipif(
    "OPENML_OFFLINE" in os.environ, reason="OpenML offline mode set in env"
)
def test_electricity_normalization_no_leak():
    ds = load_electricity(segment_id="middle", size=5000)
    head_mean = ds.X[:200].mean(axis=0)
    head_std = ds.X[:200].std(axis=0)
    assert np.all(np.abs(head_mean) < 1e-4)
    # one-hot 列在前 200 内可能常数 (std=0)；放宽下限到 0
    assert np.all(head_std <= 1.0 + 1e-2)


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------


def test_load_real_world_dispatch_unknown_name():
    with pytest.raises(ValueError):
        load_real_world("cifar10", segment_id="start")
