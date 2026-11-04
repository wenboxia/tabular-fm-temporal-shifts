"""固定分段的 RET（src/eval/round2_metrics.py 的 ret_segments / ret_segments_per_segment），用手算的小例子核对。

流 10 行，原下标是偶数 0, 2, …, 18；留出行是奇数 1, 3, …, 19，所以留出行 k 的流坐标是 k + 1（1…10）。
L = 4 时：段 0 = [0, 4) 含留出行 0–2，段 1 = [4, 8) 含留出行 3–6；[8, 12) 不完整，不算；坐标 8–10 的留出行不进任何段。
"""
import numpy as np
import pytest

from src.eval.round2_metrics import holdout_positions, ret, ret_segments, ret_segments_per_segment


def fake():
    n = 10
    y_hold = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 0])
    ckpt_t = np.array([3, 4, 5, 8, 9, 10])
    P = np.full((6, 10), -1, dtype=np.int8)
    P[0, :3] = [1, 1, 1]          # t = 3：不晚于段 0 的结束（4），不算
    P[1, :3] = [0, 0, 0]          # t = 4：等于段末，不算（要求 t > 段末）
    P[2, :3] = [0, 1, 1]          # t = 5：段 0 → 1.0；段 1 的行还没预测
    P[3, :3] = [1, 1, 1]          # t = 8：段 0 → 0.5（类 0 召回 0，类 1 召回 1）；段 1 结束于 8，不算
    P[3, 3:7] = [1, 1, 1, 1]
    P[4, 3:7] = [0, 0, 1, 1]      # t = 9：段 0 的行是 -1 → 段 0 跳过这个检查点；段 1 → 1.0
    P[5, :7] = [0, 0, 1, 1, 1, 0, 1]   # t = 10：段 0 → (1 + 1/2)/2 = 0.75；段 1 [1,1,0,1] → (0 + 1/2)/2 = 0.25
    P[:, 7:] = 0                  # 不在任何完整段里的留出行，取值不影响结果
    return dict(y=np.zeros(n, int), stream_idx=np.arange(0, 2 * n, 2), hold_idx=np.arange(1, 2 * n + 1, 2),
                y_hold=y_hold, ckpt_t=ckpt_t, ckpt_pred=P,
                regime_hold=np.zeros(10, int), boundaries=np.array([], int))


def test_positions():
    assert holdout_positions(fake()).tolist() == list(range(1, 11))


def test_hand_computed_L4():
    d = fake()
    per = ret_segments_per_segment(d, L=4)
    assert set(per) == {0, 1}
    assert per[0] == pytest.approx((1.0 + 0.5 + 0.75) / 3)
    assert per[1] == pytest.approx((1.0 + 0.25) / 2)
    assert ret_segments(d, L=4) == pytest.approx((0.75 + 0.625) / 2)


def test_lag():
    d = fake()
    per = ret_segments_per_segment(d, L=4, lag=1)       # 段 0 要 t > 5：t = 8、10；段 1 要 t > 9：t = 10
    assert per[0] == pytest.approx((0.5 + 0.75) / 2) and per[1] == pytest.approx(0.25)
    per = ret_segments_per_segment(d, L=4, lag=2)       # 段 1 要 t > 10：没有检查点 → 不进平均
    assert set(per) == {0}
    assert ret_segments(d, L=4, lag=2) == pytest.approx(0.625)


def test_other_lengths():
    d = fake()
    # L = 8：只有段 [0, 8)，含留出行 0–6；t = 9 时这些行有 -1 → 只剩 t = 10：类 0 (行 0,3,4) 预测 0,1,1 → 1/3，类 1 → 1/2
    assert ret_segments_per_segment(d, L=8) == {0: pytest.approx((1 / 3 + 1 / 2) / 2)}
    # L = 1：段 0 = [0, 1) 没有留出行 → 跳过
    assert 0 not in ret_segments_per_segment(d, L=1)
    # 没有完整段
    assert ret_segments_per_segment(d, L=20) == {}
    assert np.isnan(ret_segments(d, L=20))


def test_does_not_touch_official_ret():
    d = fake()
    d["boundaries"] = np.array([5])
    d["regime_hold"] = np.array([0] * 4 + [1] * 6)
    before = ret(d)
    ret_segments(d, L=4)
    assert ret(d) == before
