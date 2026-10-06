"""
Phase 5.5 Step 3 单元测试：Insects label_scheme + aligned_v2 切段

重点是一个正确性风险：
**label_scheme 丢行后，段内漂移坐标必须按保留行重新映射**，
否则 oracle 触发时刻会指向错误的样本。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.data.real_world import (
    _INSECTS_ALIGNED_V2_BOUNDS,
    _INSECTS_V2_KNOWN_DEGENERATE,
    _INSECTS_V2_USABLE_DRIFT_SEGMENTS,
    _INSECTS_ALIGNED_V2_SEGMENTS,
    _INSECTS_EMPIRICAL_PY_SHIFT_POINTS,
    _INSECTS_LABEL_SCHEMES,
    _INSECTS_OFFICIAL_DRIFT_POINTS,
    _binarize_insects,
    load_insects,
)

_CSV = os.path.expanduser("~/.cache/insects_drift/abrupt_balanced.csv")
needs_csv = pytest.mark.skipif(
    not os.path.exists(_CSV), reason="Insects CSV not cached locally"
)


class TestBinarize:
    """纯函数，无需数据文件。"""

    def test_pair_parity_keeps_all_rows(self):
        raw = np.array([2, 3, 4, 5, 11, 12])
        y, keep = _binarize_insects(raw, "pair_parity")
        assert keep.all()
        assert y.tolist() == [0, 1, 0, 1, 0, 1]

    def test_pair_a_vs_b_drops_third_pair(self):
        raw = np.array([2, 3, 4, 5, 11, 12])
        y, keep = _binarize_insects(raw, "pair_A_vs_B")
        assert keep.tolist() == [True, True, True, True, False, False]
        assert y.tolist() == [0, 0, 1, 1]      # {2,3}→0, {4,5}→1

    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError, match="label_scheme"):
            _binarize_insects(np.array([2, 3]), "sex")

    def test_schemes_declared(self):
        assert set(_INSECTS_LABEL_SCHEMES) == {"pair_parity", "pair_A_vs_B"}


class TestDriftRemap:

    def test_official_points_differ_from_empirical(self):
        """两组坐标必须是不同的东西 —— 混用是 Phase 5 的原始错误。"""
        assert _INSECTS_OFFICIAL_DRIFT_POINTS == [14352, 19500, 33240, 38682, 39510]
        assert set(_INSECTS_OFFICIAL_DRIFT_POINTS).isdisjoint(
            _INSECTS_EMPIRICAL_PY_SHIFT_POINTS
        )

    @needs_csv
    def test_pair_parity_keeps_raw_coordinates(self):
        for seg, (lo, hi) in _INSECTS_ALIGNED_V2_BOUNDS.items():
            d = load_insects(segment_id=seg, aligned_v2=True,
                             label_scheme="pair_parity", allow_degenerate=True)
            expected = [x - lo for x in _INSECTS_OFFICIAL_DRIFT_POINTS if lo < x < hi]
            assert d.drift_points == expected, seg
            assert len(d.X) == hi - lo

    @needs_csv
    def test_filtered_scheme_shifts_coordinates(self):
        """d2 / d4 在 pair_A_vs_B 下丢了很多行 → 坐标必须左移。"""
        lo, hi = _INSECTS_ALIGNED_V2_BOUNDS["d2_19500"]
        raw_local = 19500 - lo
        d = load_insects(segment_id="d2_19500", aligned_v2=True,
                         label_scheme="pair_A_vs_B", allow_degenerate=True)
        assert len(d.drift_points) == 1
        assert d.drift_points[0] < raw_local, (
            "丢行后漂移坐标应左移；没左移说明重映射没生效"
        )
        assert len(d.X) < hi - lo

    @needs_csv
    def test_remapped_point_lands_on_the_right_sample(self):
        """重映射后的坐标，应正好是原漂移点之后的第一个保留样本。"""
        import pandas as pd
        lo, hi = _INSECTS_ALIGNED_V2_BOUNDS["d4_double"]
        raw = pd.read_csv(_CSV, header=None).iloc[lo:hi, 33].to_numpy()
        _, keep = _binarize_insects(raw, "pair_A_vs_B")
        d = load_insects(segment_id="d4_double", aligned_v2=True,
                         label_scheme="pair_A_vs_B")   # 有效段，无需 allow_degenerate
        for abs_d, mapped in zip(
            [x for x in _INSECTS_OFFICIAL_DRIFT_POINTS if lo < x < hi],
            d.drift_points,
        ):
            assert mapped == int(keep[: abs_d - lo].sum()), (
                f"漂移 {abs_d} 应映射到保留行计数 {int(keep[:abs_d-lo].sum())}，实为 {mapped}"
            )


class TestAlignedV2Segments:

    def test_segments_do_not_overlap(self):
        bounds = sorted(_INSECTS_ALIGNED_V2_BOUNDS.values())
        for (a_lo, a_hi), (b_lo, b_hi) in zip(bounds[:-1], bounds[1:]):
            assert a_hi <= b_lo, f"段 [{a_lo},{a_hi}) 与 [{b_lo},{b_hi}) 重叠"

    def test_every_official_drift_is_covered(self):
        covered = {
            d for lo, hi in _INSECTS_ALIGNED_V2_BOUNDS.values()
            for d in _INSECTS_OFFICIAL_DRIFT_POINTS if lo < d < hi
        }
        assert covered == set(_INSECTS_OFFICIAL_DRIFT_POINTS), (
            f"未覆盖的官方漂移点: {set(_INSECTS_OFFICIAL_DRIFT_POINTS) - covered}"
        )

    def test_drifts_have_room_on_both_sides(self):
        """变点距段两端 ≥ 1000 步：够 ADWIN 累积，也够观察恢复。"""
        for seg, (lo, hi) in _INSECTS_ALIGNED_V2_BOUNDS.items():
            for d in _INSECTS_OFFICIAL_DRIFT_POINTS:
                if lo < d < hi:
                    assert d - lo >= 1000, f"{seg}: 漂移 {d} 距段首仅 {d - lo}"
                    assert hi - d >= 1000, f"{seg}: 漂移 {d} 距段尾仅 {hi - d}"

    def test_control_segment_has_no_drift(self):
        lo, hi = _INSECTS_ALIGNED_V2_BOUNDS["d0_control"]
        assert not [d for d in _INSECTS_OFFICIAL_DRIFT_POINTS if lo < d < hi]

    def test_aligned_flags_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="互斥"):
            load_insects(segment_id="mid", insects_aligned=True, aligned_v2=True)

    def test_bad_segment_id_raises(self):
        with pytest.raises(ValueError, match="aligned_v2"):
            load_insects(segment_id="mid", aligned_v2=True)

    @needs_csv
    def test_all_v2_segments_load_under_both_schemes(self):
        for seg in _INSECTS_ALIGNED_V2_SEGMENTS:
            for scheme in _INSECTS_LABEL_SCHEMES:
                d = load_insects(segment_id=seg, aligned_v2=True, label_scheme=scheme,
                                 allow_degenerate=True)
                assert len(d.X) == len(d.y) >= 1000
                assert set(np.unique(d.y)) <= {0, 1}
                assert scheme in d.name or scheme == "pair_parity"


class TestSegmentValidity:
    """漂移前后必须都有两类，否则测的是 P(y) 构成变化而不是概念漂移。"""

    @needs_csv
    def test_degenerate_segments_are_rejected_by_default(self):
        for seg, scheme in _INSECTS_V2_KNOWN_DEGENERATE:
            with pytest.raises(ValueError, match="退化"):
                load_insects(segment_id=seg, aligned_v2=True, label_scheme=scheme)

    @needs_csv
    def test_usable_segments_load_cleanly(self):
        for seg in _INSECTS_V2_USABLE_DRIFT_SEGMENTS:
            for scheme in _INSECTS_LABEL_SCHEMES:
                d = load_insects(segment_id=seg, aligned_v2=True, label_scheme=scheme)
                for dp in d.drift_points:
                    assert len(np.unique(d.y[200:dp])) == 2
                    assert len(np.unique(d.y[dp:])) == 2

    @needs_csv
    def test_allow_degenerate_escape_hatch(self):
        d = load_insects(segment_id="d1_14352", aligned_v2=True,
                         label_scheme="pair_parity", allow_degenerate=True)
        assert len(d.X) > 0

    @needs_csv
    def test_control_segment_needs_no_override(self):
        """无漂移段没有漂移点可检查，不该被有效性守卫误伤。"""
        d = load_insects(segment_id="d0_control", aligned_v2=True,
                         label_scheme="pair_A_vs_B")
        assert d.drift_points == []
