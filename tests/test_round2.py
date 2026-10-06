"""Round 2 的不变量：留出集不泄漏、不偷看未来、批不跨边界、守预算、回测不改记忆、路由确定。"""
import numpy as np
import pytest

from src.data.temporal_loader import DualMemoryLoader
from src.eval.round2_metrics import balanced_accuracy, ret_per_regime, single_class_run_mask
from src.eval.stream_runner import run_stream
from src.memory.context_memory import (NO_ROUTE, DualMemory, RegimeArchive, SlidingMemory,
                                       make_memory)
from src.memory.router import WINDOW, route_queries
from src.models.slow_prior import SlowPrior
from src.utils.forgetting import carve_regime_holdouts


def _toy(n=3000, K=3, R=3, seed=0):
    rng = np.random.default_rng(seed)
    regime = np.repeat(np.arange(R), n // R)
    y = rng.integers(0, K, len(regime))
    X = rng.normal(size=(len(regime), 4)).astype(np.float32) + regime[:, None] * 3.0
    return X, y, regime


# ---------------------------------------------------------------- holdouts

def test_holdouts_stratified_disjoint_deterministic():
    X, y, regime = _toy()
    s1, h1 = carve_regime_holdouts(y, regime, {0: 10, 1: 10, 2: 5}, seed=3)
    s2, h2 = carve_regime_holdouts(y, regime, {0: 10, 1: 10, 2: 5}, seed=3)
    assert np.array_equal(h1, h2) and np.array_equal(s1, s2)
    assert len(np.intersect1d(s1, h1)) == 0 and len(s1) + len(h1) == len(y)
    for r, k in {0: 10, 1: 10, 2: 5}.items():
        hr = h1[regime[h1] == r]
        assert np.array_equal(np.bincount(y[hr], minlength=3), [k] * 3)
        rows = np.flatnonzero(regime == r)          # 抽自整个阶段，而不是末尾一段
        assert hr.min() < rows[len(rows) // 2] < hr.max()
    _, h3 = carve_regime_holdouts(y, regime, 10, seed=4)
    assert not np.array_equal(np.sort(h1), np.sort(h3))


# ---------------------------------------------------------------- runner invariants

class Recorder:
    def __init__(self, K):
        self.K, self.calls = K, []

    def __call__(self, Xc, yc, Xq):
        self.calls.append((len(Xc), len(Xq)))
        p = np.zeros((len(Xq), self.K)); p[:, int(np.bincount(yc).argmax())] = 1.0
        return p


@pytest.mark.parametrize("method", ["sw200", "sw1000", "dual400", "cbfifo400",
                                    "arch_union", "arch_routed"])
def test_runner_invariants(method):
    X, y, regime = _toy(n=3000)
    s, h = carve_regime_holdouts(y, regime, 10, seed=0)
    Xs, ys, rs = X[s], y[s], regime[s]
    bounds = [int(np.searchsorted(s, b)) for b in (1000, 2000)]
    mem = make_memory(method, Xs, ys, 3, seed=0)
    rec = Recorder(3)
    r = run_stream(mem, rec, Xs, ys, rs, X[h], regime[h], bounds, stride=50, start=200, ckpt_every=500)
    assert r.leak_count == 0
    assert (r.pred[200:] >= 0).all() and (r.pred[:200] == -1).all()
    assert r.max_ctx <= mem.max_context()
    assert len(r.digests) == len(r.ckpt_t)
    assert list(r.closures) == bounds
    for b in bounds:                                            # 边界处必须有检查点
        assert b in set(r.ckpt_t.tolist())
    # 每个检查点评估的是"已出现阶段"的全部留出行
    for c, upto in enumerate(r.ckpt_upto):
        assert ((r.ckpt_pred[c] >= 0) == (regime[h] <= upto)).all()


def test_no_context_row_from_the_future():
    X, y, regime = _toy(n=1200)
    checks = []

    class Mem(SlidingMemory):
        def contexts(self, Xq):
            g = super().contexts(Xq)
            checks.append(int(g[0][1].max()))
            return g

    def fn(Xc, yc, Xq):
        p = np.zeros((len(Xq), 3)); p[:, 0] = 1; return p

    starts = []
    orig_update = Mem.update

    def upd(self, idx):
        starts.append(int(np.min(idx)) if len(idx) else -1); orig_update(self, idx)

    Mem.update = upd
    r = run_stream(Mem(200), fn, X, y, regime, X[:0], regime[:0], [400, 800],
                   stride=50, start=200, ckpt_every=10_000)
    assert r.leak_count == 0
    # 第 k 次取上下文时，上下文最大下标 < 第 k+1 次 update 的起点（= 本批第一行）
    for mx, st in zip(checks, starts[1:]):
        assert mx < st


def test_batches_do_not_straddle_boundaries():
    X, y, regime = _toy(n=1500)
    sizes = []

    def fn(Xc, yc, Xq):
        sizes.append(len(Xq)); p = np.zeros((len(Xq), 3)); p[:, 0] = 1; return p

    run_stream(SlidingMemory(100), fn, X, y, regime, X[:0], regime[:0], [425, 990],
               stride=50, start=200, ckpt_every=10_000)
    # 200..425 → 50,50,50,50,25 ；425..990 → ... 最后一批到 990 截断
    assert sizes[:5] == [50, 50, 50, 50, 25]
    assert sum(sizes) == 1500 - 200


def test_checkpoint_does_not_change_memory():
    X, y, regime = _toy(n=2000)
    mem = make_memory("arch_routed", X, y, 3, seed=1)
    mem.update(np.arange(0, 700)); mem.close_regime(np.arange(0, 700))
    before = mem.digest()
    mem.contexts(X[1500:1600])
    assert mem.digest() == before


# ---------------------------------------------------------------- memories

def test_dual_memory_matches_loader_at_stride_one():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 900); y[300:500] = 1
    X = rng.normal(size=(900, 2)).astype(np.float32)
    loader = DualMemoryLoader(X, y, context_size=100, short_ratio=0.75, max_age=None, warmup=100)
    mem = DualMemory(100, 0.75, y)
    mem.update(np.arange(0, 100))
    for b in loader:
        ctx = mem.contexts(X[b.t: b.t + 1])[0][1]
        assert np.array_equal(np.sort(ctx), np.sort(_ctx_idx(loader, b)))
        mem.update([b.t])


def _ctx_idx(loader, b):
    # DualMemoryLoader 只给出 X_ctx；用行内容反查下标（X 为连续随机数，不会撞）
    lut = {tuple(np.round(r, 6)): i for i, r in enumerate(loader.X)}
    return np.array([lut[tuple(np.round(r, 6))] for r in b.X_ctx])


def test_class_balanced_fifo_budget():
    y = np.array([0] * 500 + [1] * 10 + [2] * 10)
    mem = make_memory("cbfifo400", np.zeros((520, 2), np.float32), y, 3, seed=0)
    mem.update(np.arange(520))
    ctx = mem.contexts(np.zeros((1, 2)))[0][1]
    assert (np.bincount(y[ctx], minlength=3) == [133, 10, 10]).all()


def test_archive_class_balanced_and_bounded():
    X, y, regime = _toy(n=3000, K=3)
    mem = RegimeArchive(200, 25, "union", X, y, 3, np.random.default_rng(0), max_archives=2)
    for r in range(3):
        rows = np.flatnonzero(regime == r)
        mem.update(rows); mem.close_regime(rows)
    assert len(mem.archives) == 2
    for a in mem.archives:
        assert (np.bincount(y[a], minlength=3) == 25).all()
    assert len(mem.contexts(X[:5])[0][1]) == 200 + 150


# ---------------------------------------------------------------- router

def test_router_deterministic_order_independent_and_abstains():
    rng = np.random.default_rng(0)
    W = rng.normal(0, 1, (200, 4)); A0 = rng.normal(6, 1, (75, 4)); A1 = rng.normal(-6, 1, (75, 4))
    Q = np.vstack([rng.normal(0, 1, (20, 4)), rng.normal(6, 1, (20, 4)), rng.normal(-6, 1, (20, 4))])
    r = route_queries(Q, W, [A0, A1])
    assert (r[:20] == WINDOW).all() and (r[20:40] == 0).all() and (r[40:] == 1).all()
    perm = rng.permutation(len(Q))
    assert np.array_equal(route_queries(Q[perm], W, [A0, A1]), r[perm])
    assert np.array_equal(route_queries(Q, W, [A0, A1]), r)
    assert (route_queries(Q, W, []) == WINDOW).all()


# ---------------------------------------------------------------- SlowPrior multiclass

class _FakeModel:
    def __init__(self, classes):
        self.classes_ = np.array(classes)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        p = np.zeros((len(X), len(self.classes_))); p[:, -1] = 1.0
        return p


def _prior(classes):
    sp = SlowPrior.__new__(SlowPrior)
    sp.device, sp.n_estimators, sp.random_state, sp._is_fitted = "cpu", 1, 0, False
    sp._model = _FakeModel(classes)
    return sp


def test_predict_proba_global_maps_missing_classes():
    sp = _prior([1, 4])                      # context 里只有类别 1 和 4
    p = sp.predict_proba_global(np.zeros((4, 2)), np.array([1, 4, 1, 4]), np.zeros((3, 2)), 6)
    assert p.shape == (3, 6) and (p.argmax(1) == 4).all()


def test_predict_proba_global_single_class_context():
    sp = _prior([3])
    p = sp.predict_proba_global(np.zeros((4, 2)), np.array([5, 5, 5, 5]), np.zeros((2, 2)), 6)
    assert p.shape == (2, 6) and (p.argmax(1) == 5).all()


# ---------------------------------------------------------------- metrics

def test_metrics_basics():
    assert balanced_accuracy(np.array([0, 0, 1]), np.array([0, 0, 0])) == 0.5
    m = single_class_run_mask(np.array([0] * 40 + [1, 0, 1] + [2] * 5), min_len=30)
    assert m[:40].all() and not m[40:].any()


def test_ret_uses_only_checkpoints_after_regime_end():
    d = dict(boundaries=np.array([100]), ckpt_t=np.array([50, 100, 150]),
             y_hold=np.array([0, 1, 0, 1]), regime_hold=np.array([0, 0, 1, 1]),
             ckpt_pred=np.array([[0, 1, -1, -1], [0, 1, 0, 1], [0, 0, 0, 1]]),
             y=np.zeros(200, int))
    assert ret_per_regime(d) == {0: 0.5}


@pytest.mark.slow
def test_tabpfn_batch_equals_per_row():
    pytest.importorskip("tabpfn")
    rng = np.random.default_rng(0)
    Xc = rng.normal(size=(80, 5)).astype(np.float32); yc = rng.integers(0, 3, 80)
    Xq = rng.normal(size=(6, 5)).astype(np.float32)
    sp = SlowPrior(device="cpu", n_estimators=1, random_state=0)
    batch = sp.predict_proba_global(Xc, yc, Xq, 3)
    rows = np.vstack([sp.predict_proba_global(Xc, yc, Xq[i:i + 1], 3) for i in range(len(Xq))])
    assert np.allclose(batch, rows, atol=1e-5)


# ---------------------------------------------------------------- 修订 1：阶段均衡双记忆

def test_regime_balanced_dual_equals_dual_without_regimes():
    from src.memory.context_memory import RegimeBalancedDual
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 3000); y[1000:1500] = 2
    a, b = DualMemory(400, 0.75, y), RegimeBalancedDual(400, 0.75, y)
    for s in range(0, 3000, 50):
        a.update(np.arange(s, s + 50)); b.update(np.arange(s, s + 50))
        assert np.array_equal(np.sort(a.contexts(np.zeros((1, 1)))[0][1]),
                              np.sort(b.contexts(np.zeros((1, 1)))[0][1]))


def test_regime_balanced_dual_keeps_every_old_regime():
    from src.memory.context_memory import RegimeBalancedDual
    rng = np.random.default_rng(2)
    y = rng.integers(0, 3, 6000)
    rb, du = RegimeBalancedDual(400, 0.75, y), DualMemory(400, 0.75, y)
    bounds = [1000, 2000, 3000, 4000, 5000]
    start = 0
    for s in range(0, 6000, 50):
        rb.update(np.arange(s, s + 50)); du.update(np.arange(s, s + 50))
        if s + 50 in bounds:
            rb.close_regime(np.arange(start, s + 50)); start = s + 50
    seg = rb.segment_counts()
    old = [seg.get(g, 0) for g in range(5)]                     # 阶段 0–4 都还在长期池里
    assert min(old) >= 10, seg
    du_long = np.asarray(du._long)
    assert (du_long < 1000).sum() < min(old)                    # 双记忆里最老的阶段几乎被挤光
    ctx = rb.contexts(np.zeros((1, 1)))[0][1]
    assert len(ctx) == 400 and ctx.max() < 6000


def test_regime_balanced_dual_retags_rows_after_closure():
    from src.memory.context_memory import RegimeBalancedDual
    y = np.zeros(400, dtype=int)
    m = RegimeBalancedDual(100, 0.75, y)
    m.update(np.arange(0, 300))
    m.close_regime(np.arange(0, 250))                         # ADWIN 在 249 行报警，250–299 已揭示
    assert all(m._seg[i] == 1 for i in m._short + m._long if i >= 250)
    assert all(m._seg[i] == 0 for i in m._short + m._long if i < 250)
