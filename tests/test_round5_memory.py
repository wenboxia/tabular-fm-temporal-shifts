"""Round 5（预注册修订 4）新增的记忆与消融：CBRS 双记忆、阶段编号特征（*_rid）、ADWIN 方法的登记。"""
import numpy as np
import pytest

from src.eval.stream_runner import needs_detector, run_stream
from src.memory.context_memory import (METHODS, DualMemory, RegimeBalancedDual, ReservoirDual,
                                       make_memory)
from src.utils.forgetting import carve_regime_holdouts


def _feed(mem, n, step=50):
    for s in range(0, n, step):
        mem.update(np.arange(s, min(s + step, n)))


def _imbalanced_y(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice(4, size=n, p=[0.7, 0.2, 0.07, 0.03])


# ---------------------------------------------------------------- 登记

def test_new_methods_registered():
    for m in ("dual200_cbrs", "dual1000_cbrs", "dual200_rid", "dual1000_rid",
              "dual200_rb_adwin", "dual1000_rb_adwin"):
        assert m in METHODS
    y = np.zeros(100, dtype=int)
    X = np.zeros((100, 2), np.float32)
    assert isinstance(make_memory("dual200_cbrs", X, y, 3, 0), ReservoirDual)
    assert isinstance(make_memory("dual1000_cbrs", X, y, 3, 0), ReservoirDual)
    assert make_memory("dual200_cbrs", X, y, 3, 0).max_context() == 200
    assert make_memory("dual1000_cbrs", X, y, 3, 0).max_context() == 1000
    # *_rid 的记忆与 dual200 / dual1000 完全相同
    assert type(make_memory("dual200_rid", X, y, 3, 0)) is DualMemory
    assert type(make_memory("dual1000_rid", X, y, 3, 0)) is DualMemory
    assert isinstance(make_memory("dual200_rb_adwin", X, y, 3, 0), RegimeBalancedDual)
    # 只有 *_adwin 需要检测器
    assert needs_detector("dual200_rb_adwin") and needs_detector("dual1000_rb_adwin")
    for m in ("dual200_cbrs", "dual1000_cbrs", "dual200_rid", "dual1000_rid"):
        assert not needs_detector(m)


# ---------------------------------------------------------------- ReservoirDual

@pytest.mark.parametrize("total", [200, 1000])
def test_reservoir_capacity_never_exceeded(total):
    y = _imbalanced_y(6000)
    m = ReservoirDual(total, 0.75, y, seed=0)
    for s in range(0, 6000, 50):
        m.update(np.arange(s, s + 50))
        assert m.stored_rows() <= m.max_context() == total
        assert len(m._long) <= m.long_cap and len(m._short) <= m.short_cap
        ctx = m.contexts(np.zeros((1, 1)))[0][1]
        assert len(ctx) == len(set(ctx.tolist())) == m.stored_rows()      # 无重复
        assert ctx.max() < s + 50
    assert m.stored_rows() == total
    assert len(m._long) == m.long_cap and len(m._short) == m.short_cap


def test_reservoir_short_term_is_plain_fifo():
    y = _imbalanced_y(2000)
    m = ReservoirDual(200, 0.75, y, seed=1)
    _feed(m, 2000)
    assert m._short == list(range(2000 - 150, 2000))


def test_reservoir_equals_dual_until_long_pool_full():
    y = _imbalanced_y(2000)
    a, b = DualMemory(200, 0.75, y), ReservoirDual(200, 0.75, y, seed=0)
    n = 150 + 50                                   # 短期 150 + 长期恰好装满 50
    _feed(a, n); _feed(b, n)
    assert np.array_equal(a.contexts(np.zeros((1, 1)))[0][1], b.contexts(np.zeros((1, 1)))[0][1])


def test_reservoir_long_pool_is_class_balanced():
    y = _imbalanced_y(30000, seed=3)               # 类别比例 70/20/7/3 %
    m = ReservoirDual(1000, 0.75, y, seed=0)
    _feed(m, 30000)
    cnt = np.bincount(y[m._long], minlength=4)
    assert cnt.sum() == 250
    assert cnt.min() >= 40 and cnt.max() <= 90, cnt            # 近似均衡（62.5 每类），而不是 70/20/7/3
    assert cnt.max() - cnt.min() < 0.5 * cnt.mean()


@pytest.mark.parametrize("K", [2, 3, 6])
def test_reservoir_balanced_stream_keeps_rows_uniform_in_time(K):
    """均衡 K 类流：所有类很快都成为满类（填充阶段并列最多也算），之后每条到达行都走"以 m_c/n_c 替换同类"
    分支，池 = 每类一个均匀蓄水池。入池时间应近似均匀（约一半来自前半流，中位数接近流中点），
    每类行数均衡；对照：双记忆几乎全是近期行。"""
    n = 20000
    fr, med, oldest = [], [], []
    for seed in range(5):
        y = np.random.default_rng(100 + seed).integers(0, K, n)
        m = ReservoirDual(1000, 0.75, y, seed=seed)
        _feed(m, n)
        long_idx = np.asarray(m._long)
        assert m._full == set(range(K))
        fr.append(float((long_idx < n // 2).mean()))
        med.append(float(np.median(long_idx)) / n)
        oldest.append(int(long_idx.min()))
        cnt = np.bincount(y[long_idx], minlength=K)
        assert cnt.sum() == 250
        assert cnt.min() >= 0.6 * cnt.mean() and cnt.max() <= 1.4 * cnt.mean(), cnt
    assert 0.42 <= np.mean(fr) <= 0.62, fr
    assert 0.40 <= np.mean(med) <= 0.60, med
    assert min(oldest) < n // 10, oldest                         # 流最开头附近的行仍能留下
    y = np.random.default_rng(100).integers(0, K, n)
    du = DualMemory(1000, 0.75, y)
    _feed(du, n)
    assert (np.asarray(du._long) < n // 2).mean() < 0.1         # 对照：双记忆几乎全是近期行


def test_reservoir_full_class_rule_hand_checked():
    """满类 = 当前是或曾经是最多的类，一旦满就一直满（原文）。

    长期池 3 行（短期 1 行）。进入长期池的类别依次为 0,0,0,1,1,0：
      0,0,0 -> 池 {0,0,0}，满类 {0}；1 -> 非满类，覆盖一条 0 -> {0,0,1}；
      1 -> 非满类，覆盖一条 0 -> {0,1,1}，1 成为最多 -> 满类 {0,1}；
      0 -> 0 是满类（虽然当前不是最多）：只能以 m_0/n_0 = 1/4 替换池里那条 0 类行，否则丢弃；
           绝不能覆盖类 1（若把"满类"误解为"当前最多的类"就会覆盖类 1，池变 {0,0,1}）。
    """
    y = np.array([0, 0, 0, 1, 1, 0, 1])
    for seed in range(20):
        m = ReservoirDual(4, 0.25, y, seed=seed)
        assert (m.short_cap, m.long_cap) == (1, 3)
        m.update(np.arange(4))                     # 行 0,1,2 进入长期池
        assert m._full == {0}
        m.update([4, 5])                           # 行 3,4（类 1）
        assert np.array_equal(np.bincount(y[m._long], minlength=2), [1, 2]) and m._full == {0, 1}
        m.update([6])                              # 行 5（类 0，满类但不是最多）
        assert np.array_equal(np.bincount(y[m._long], minlength=2), [1, 2]), (seed, m._long)
        assert m._n_seen == {0: 4, 1: 2}


def test_reservoir_fill_phase_marks_ties_full():
    y = np.array([0, 1, 2, 2, 0])
    m = ReservoirDual(5, 0.2, y, seed=0)            # 短期 1，长期 4
    m.update([0, 1, 2])                            # 行 0,1 进入（类 0、1 并列最多）
    assert m._full == {0, 1}
    m.update([3])                                  # 行 2（类 2）进入：三类并列
    assert m._full == {0, 1, 2}


def test_reservoir_full_set_is_monotone():
    """类别比例随时间变化（先只有类 0，再是 1/2，最后 3 很少）时，满类集合只增不减。"""
    rng = np.random.default_rng(4)
    y = np.r_[np.zeros(600, int), rng.integers(1, 3, 3000), rng.choice([0, 1, 2, 3], 4000, p=[.1, .1, .1, .7])]
    m = ReservoirDual(200, 0.75, y, seed=0)
    prev = set()
    for s in range(0, len(y), 50):
        m.update(np.arange(s, min(s + 50, len(y))))
        assert prev <= m._full
        prev = set(m._full)
    assert m._full == {0, 1, 2, 3}


def test_reservoir_deterministic_and_seed_dependent():
    y = _imbalanced_y(8000, seed=5)
    def run(seed):
        m = ReservoirDual(200, 0.75, y, seed=seed)
        _feed(m, 8000)
        return m
    a, b, c = run(0), run(0), run(1)
    assert a.digest() == b.digest() and a._long == b._long
    assert a.digest() != c.digest() and a._long != c._long


def test_reservoir_ignores_regime_information():
    y = _imbalanced_y(8000, seed=6)
    a = ReservoirDual(200, 0.75, y, seed=2)
    b = ReservoirDual(200, 0.75, y, seed=2)
    start = 0
    for s in range(0, 8000, 50):
        a.update(np.arange(s, s + 50)); b.update(np.arange(s, s + 50))
        if (s + 50) % 1000 == 0:
            b.close_regime(np.arange(start, s + 50)); start = s + 50
    assert a.digest() == b.digest() and a._long == b._long


def test_reservoir_digest_stable_under_contexts_and_changes_with_update():
    y = _imbalanced_y(3000, seed=7)
    m = ReservoirDual(200, 0.75, y, seed=0)
    _feed(m, 3000)
    before = m.digest()
    for _ in range(3):
        m.contexts(np.zeros((10, 2)))
    assert m.digest() == before
    y2 = np.r_[y, _imbalanced_y(50, seed=8)]
    m.y = y2
    m.update(np.arange(3000, 3050))
    assert m.digest() != before


@pytest.mark.parametrize("method", ["dual200_cbrs", "dual1000_cbrs", "dual200_rid"])
def test_runner_invariants_new_methods(method):
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
    assert len(r.digests) == len(r.ckpt_t) > 0                   # 检查点不改变记忆（runner 内部断言）
    assert (r.pred[200:] >= 0).all()


# ---------------------------------------------------------------- *_rid 特征列

def test_add_regime_id_feature():
    from scripts.run_round2 import add_regime_id_feature, uses_regime_id
    assert uses_regime_id("dual200_rid") and uses_regime_id("dual1000_rid")
    assert not uses_regime_id("dual200") and not uses_regime_id("dual200_rb")
    rng = np.random.default_rng(0)
    d = dict(X=rng.normal(size=(30, 5)).astype(np.float32), regime=np.repeat([0, 1, 2], 10),
             X_hold=rng.normal(size=(6, 5)).astype(np.float32), regime_hold=np.array([0, 0, 1, 1, 2, 2]))
    X0, H0 = d["X"].copy(), d["X_hold"].copy()
    out = add_regime_id_feature(d)
    assert out is d
    assert d["X"].shape == (30, 6) and d["X_hold"].shape == (6, 6)
    assert d["X"].dtype == np.float32 and d["X_hold"].dtype == np.float32
    assert np.array_equal(d["X"][:, :5], X0) and np.array_equal(d["X_hold"][:, :5], H0)
    assert np.array_equal(d["X"][:, 5], np.repeat([0, 1, 2], 10).astype(np.float32))
    assert np.array_equal(d["X_hold"][:, 5], np.array([0, 0, 1, 1, 2, 2], np.float32))


@pytest.mark.slow
def test_rid_feature_runs_through_tabpfn():
    """多一列特征时 SlowPrior（真 TabPFN，CPU）照常给出全局类别概率。"""
    from src.models.slow_prior import SlowPrior
    rng = np.random.default_rng(0)
    Xc = rng.normal(size=(60, 5)).astype(np.float32); Xc[:, -1] = np.repeat([0, 1], 30)
    yc = rng.integers(0, 3, 60)
    Xq = rng.normal(size=(8, 5)).astype(np.float32); Xq[:, -1] = 1
    p = SlowPrior(device="cpu", n_estimators=1, random_state=0).predict_proba_global(Xc, yc, Xq, 4)
    assert p.shape == (8, 4) and np.allclose(p.sum(1), 1, atol=1e-4)
