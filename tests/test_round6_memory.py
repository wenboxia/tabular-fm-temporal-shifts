"""Round 6（预注册修订 5）新增的记忆：阶段均衡的蓄水池双记忆（*_rbr）。"""
import numpy as np
import pytest

from src.eval.stream_runner import needs_detector, run_stream
from src.memory.context_memory import (METHODS, RegimeBalancedDual, RegimeReservoirDual, ReservoirDual,
                                       make_memory)
from src.utils.forgetting import carve_regime_holdouts


def _imbalanced_y(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice(4, size=n, p=[0.7, 0.2, 0.07, 0.03])


def _feed_regimes(mem, lengths, step=50):
    """依次喂入若干阶段（长度 lengths），每个阶段结束时 close_regime；返回各阶段的 [起, 止)。"""
    spans, s = [], 0
    for L in lengths:
        a = s
        for t in range(a, a + L, step):
            mem.update(np.arange(t, min(t + step, a + L)))
        mem.close_regime(np.arange(a, a + L))
        spans.append((a, a + L))
        s = a + L
    return spans


def _within_positions(mem, spans):
    """长期池里每个已结束阶段的行在该阶段内的平均相对位置（0 = 开头，1 = 结尾）。"""
    long_ = np.asarray(mem._long)
    out = {}
    for j, (a, b) in enumerate(spans):
        sel = long_[(long_ >= a) & (long_ < b)]
        if len(sel):
            out[j] = float(((sel - a) / (b - a)).mean())
    return out


# ---------------------------------------------------------------- 登记

def test_rbr_methods_registered():
    y = np.zeros(100, dtype=int)
    X = np.zeros((100, 2), np.float32)
    for m, total in (("dual200_rbr", 200), ("dual1000_rbr", 1000)):
        assert m in METHODS
        mem = make_memory(m, X, y, 3, 0)
        assert type(mem) is RegimeReservoirDual
        assert mem.max_context() == total
        assert (mem.short_cap, mem.long_cap) == (int(round(total * 0.75)), total - int(round(total * 0.75)))
        assert not needs_detector(m)


# ---------------------------------------------------------------- 与 CBRS、阶段均衡双记忆的关系

@pytest.mark.parametrize("total", [200, 1000])
def test_rbr_without_regimes_equals_cbrs_step_by_step(total):
    """只有一个阶段时，决策与 ReservoirDual 逐条一致（同一随机数序列）。"""
    y = _imbalanced_y(8000, seed=3)
    a = RegimeReservoirDual(total, 0.75, y, seed=5, rng_base=20_000)
    b = ReservoirDual(total, 0.75, y, seed=5)
    for s in range(0, 8000, 50):
        a.update(np.arange(s, s + 50))
        b.update(np.arange(s, s + 50))
        assert a._long == b._long and a._short == b._short


def test_rbr_regime_counts_match_regime_balanced_dual():
    """阶段之间的名额与 RegimeBalancedDual 相同（相差至多 1 行）。"""
    rng = np.random.default_rng(1)
    lengths = [3000, 400, 2500, 300, 1800, 600]
    y = rng.integers(0, 3, sum(lengths))
    for total in (200, 1000):
        a = RegimeReservoirDual(total, 0.75, y, seed=0)
        b = RegimeBalancedDual(total, 0.75, y)
        _feed_regimes(a, lengths)
        _feed_regimes(b, lengths)
        ca, cb = a.segment_counts(), b.segment_counts()
        assert set(ca) == set(cb)
        assert all(abs(ca[g] - cb[g]) <= 1 for g in ca), (total, ca, cb)


def test_rbr_keeps_rows_spread_within_long_regimes():
    """阶段内部均匀抽样：长阶段留下的行平均位置接近 0.5；RegimeBalancedDual 只留阶段末尾。"""
    rng = np.random.default_rng(2)
    lengths = [6000, 5000, 4000]
    y = rng.integers(0, 3, sum(lengths))
    pos_rbr, pos_rb = [], []
    for seed in range(5):
        a = RegimeReservoirDual(1000, 0.75, y, seed=seed)
        spans = _feed_regimes(a, lengths)
        pos_rbr += list(_within_positions(a, spans[:-1]).values())
    b = RegimeBalancedDual(1000, 0.75, y)
    spans = _feed_regimes(b, lengths)
    pos_rb += list(_within_positions(b, spans[:-1]).values())
    assert 0.35 < np.mean(pos_rbr) < 0.65, pos_rbr
    assert np.mean(pos_rb) > 0.8, pos_rb


def test_rbr_short_regime_is_never_dropped():
    """短阶段也保住名额（CBRS 按长度分配，短阶段可能被挤掉）。"""
    rng = np.random.default_rng(4)
    lengths = [5000, 250, 5000, 5000]
    y = rng.integers(0, 3, sum(lengths))
    mem = RegimeReservoirDual(200, 0.75, y, seed=0)
    _feed_regimes(mem, lengths)
    counts = mem.segment_counts()
    assert counts.get(1, 0) >= mem.long_cap // len(lengths) - 2, counts


def test_rbr_class_balance_within_regime():
    """阶段内部按类别均衡：一个阶段里多数类占 90 %，长期池里该阶段的各类行数相差不大。"""
    rng = np.random.default_rng(5)
    n = 8000
    y = np.where(rng.random(n) < 0.9, 0, rng.integers(1, 3, n))
    mem = RegimeReservoirDual(1000, 0.75, y, seed=0)
    _feed_regimes(mem, [n])
    seg = np.asarray(mem._long)
    counts = np.bincount(y[seg], minlength=3)
    assert counts.min() > 0.2 * counts.max(), counts


# ---------------------------------------------------------------- 不变量

def test_rbr_capacity_full_sets_and_no_future_rows():
    rng = np.random.default_rng(6)
    lengths = [1200, 300, 2000, 700, 1500]
    y = rng.integers(0, 4, sum(lengths))
    mem = RegimeReservoirDual(200, 0.75, y, seed=1)
    prev_r, prev_c, s = set(), {}, 0
    for L in lengths:
        for t in range(s, s + L, 50):
            mem.update(np.arange(t, min(t + 50, s + L)))
            assert len(mem._short) <= mem.short_cap and len(mem._long) <= mem.long_cap
            assert mem.stored_rows() <= mem.max_context()
            assert max(mem._all()) < min(t + 50, s + L)                     # 没有未来的行
            assert prev_r <= mem._full_r                                     # 满阶段只增不减
            assert all(prev_c[g] <= mem._full_c.get(g, set()) for g in prev_c)
            prev_r, prev_c = set(mem._full_r), {g: set(c) for g, c in mem._full_c.items()}
        mem.close_regime(np.arange(s, s + L))
        s += L
    assert sorted(mem._long) == mem._long                                    # 池内按时间升序
    assert set(mem._seg) == set(mem._long) | set(mem._short)                 # 被删或丢弃的行不再记录


def test_rbr_deterministic_and_seed_dependent():
    rng = np.random.default_rng(7)
    lengths = [2000, 800, 2500]
    y = rng.integers(0, 3, sum(lengths))
    digests = []
    for seed in (0, 0, 1):
        mem = RegimeReservoirDual(200, 0.75, y, seed=seed)
        _feed_regimes(mem, lengths)
        digests.append(mem.digest())
    assert digests[0] == digests[1] != digests[2]


def test_rbr_digest_stable_under_contexts_and_changes_with_update():
    y = _imbalanced_y(3050, seed=8)
    mem = RegimeReservoirDual(200, 0.75, y, seed=0)
    _feed_regimes(mem, [1500, 1500])
    before = mem.digest()
    for _ in range(3):
        mem.contexts(np.zeros((10, 2)))
    assert mem.digest() == before
    mem.update(np.arange(3000, 3050))
    assert mem.digest() != before


@pytest.mark.parametrize("method", ["dual200_rbr", "dual1000_rbr"])
def test_runner_invariants_rbr(method):
    rng = np.random.default_rng(0)
    n, K, R = 3600, 3, 3
    regime = np.repeat(np.arange(R), n // R)
    y = rng.integers(0, K, n)
    X = rng.normal(size=(n, 4)).astype(np.float32) + regime[:, None] * 3.0
    s, h = carve_regime_holdouts(y, regime, 10, seed=0)
    Xs, ys, rs = X[s], y[s], regime[s]
    bounds = [int(np.searchsorted(s, b)) for b in (1200, 2400)]
    mem = make_memory(method, Xs, ys, K, seed=0)

    def fn(Xc, yc, Xq):
        p = np.zeros((len(Xq), K)); p[:, int(np.bincount(yc).argmax())] = 1.0
        return p

    r = run_stream(mem, fn, Xs, ys, rs, X[h], regime[h], bounds, stride=50, start=200, ckpt_every=500)
    assert r.leak_count == 0
    assert r.max_ctx <= mem.max_context()
    assert list(r.closures) == bounds                                         # 只在官方边界关闭阶段
    assert len(r.digests) == len(r.ckpt_t) > 0                               # 检查点不改变记忆（runner 内部断言）
    assert (r.pred[200:] >= 0).all()


# ---------------------------------------------------------------- 与修订 5 原文逐条对照的参考实现（审阅建议，2026-11-01）

class _ReferenceRBR:
    """直接按预注册修订 5 的文字写的参考实现（列表 + Counter，不求快），用来逐批对照 RegimeReservoirDual。"""

    def __init__(self, total, y, seed, rng_base=30_000):
        from collections import Counter
        self.Counter = Counter
        self.S = int(round(total * 0.75))
        self.M = total - self.S
        self.y = y
        self.short, self.long, self.seg, self.cur = [], [], {}, 0
        self.n, self.FR, self.FC = {}, set(), {}
        self.rng = np.random.default_rng(rng_base + seed)

    def update(self, idx):
        for i in idx:
            i = int(i)
            self.seg[i] = self.cur
            self.short.append(i)
            if len(self.short) > self.S:
                self.enter(self.short.pop(0))

    def close_regime(self, rows):
        cut = int(max(rows)) + 1
        self.cur += 1
        for i in self.short:
            if i >= cut:
                self.seg[i] = self.cur

    def mark(self):
        rc = self.Counter(self.seg[i] for i in self.long)
        self.FR |= {g for g, v in rc.items() if v == max(rc.values())}
        for g in rc:
            cc = self.Counter(int(self.y[i]) for i in self.long if self.seg[i] == g)
            self.FC.setdefault(g, set()).update(k for k, v in cc.items() if v == max(cc.values()))

    def largest_class_row(self, g):
        cc = self.Counter(int(self.y[i]) for i in self.long if self.seg[i] == g)
        tied = sorted(k for k, v in cc.items() if v == max(cc.values()))
        k = tied[self.rng.integers(len(tied))]
        cell = [p for p, i in enumerate(self.long) if self.seg[i] == g and int(self.y[i]) == k]
        return cell[self.rng.integers(len(cell))]

    def enter(self, x):
        g, c = self.seg[x], int(self.y[x])
        self.n[(g, c)] = self.n.get((g, c), 0) + 1
        if len(self.long) < self.M:                                   # 1. 池未满
            self.long.append(x)
            self.mark()
            return
        if g not in self.FR:                                          # 2. 非满阶段：最大阶段（并列取最新）里最大类的一条
            rc = self.Counter(self.seg[i] for i in self.long)
            g_star = max(h for h, v in rc.items() if v == max(rc.values()))
            victim = self.largest_class_row(g_star)
        elif c not in self.FC.get(g, set()):                          # 3. 满阶段里的非满类
            victim = self.largest_class_row(g)
        else:                                                         # 4. 满格子：概率 m / n
            cell = [p for p, i in enumerate(self.long) if self.seg[i] == g and int(self.y[i]) == c]
            if not self.rng.random() < len(cell) / self.n[(g, c)]:
                del self.seg[x]
                return
            victim = cell[self.rng.integers(len(cell))]
        del self.seg[self.long.pop(victim)]
        self.long.append(x)
        self.mark()


@pytest.mark.parametrize("trial", range(12))
def test_rbr_matches_reference_implementation_of_amendment5(trial):
    """运行器的调用方式：先 update(0..200)，再每批 50 行、批不跨边界，在边界处 close_regime；每批之后全部状态一致。"""
    rng = np.random.default_rng(100 + trial)
    K = int(rng.integers(2, 7))
    lengths = rng.integers(150, 2500, int(rng.integers(1, 8)))
    n = int(lengths.sum()) + 200
    y = rng.integers(0, K, n)
    total = (200, 1000)[trial % 2]
    a = RegimeReservoirDual(total, 0.75, y, seed=trial)
    r = _ReferenceRBR(total, y, seed=trial)
    bounds = list(np.cumsum(lengths) + 200)[:-1]
    for m in (a, r):
        m.update(np.arange(0, 200))
    t, seg_start = 200, 0
    while t < n:
        end = min(t + 50, n, *[b for b in bounds if b > t] or [n])
        for m in (a, r):
            m.update(np.arange(t, end))
        if end in bounds:
            for m in (a, r):
                m.close_regime(np.arange(seg_start, end))
            seg_start = end
        assert a._long == r.long and a._short == r.short
        assert a._seg == r.seg and a._n_seen == r.n
        assert a._full_r == r.FR and a._full_c == r.FC
        t = end


def test_rbr_rng_base_is_30000_plus_seed():
    y = np.zeros(10, dtype=int)
    X = np.zeros((10, 2), np.float32)
    for s in (0, 7):
        for m in ("dual200_rbr", "dual1000_rbr"):
            assert make_memory(m, X, y, 2, s)._rng.random() == np.random.default_rng(30_000 + s).random()
