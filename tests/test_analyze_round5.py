"""第五轮分析脚本（scripts/analyze_round5.py）：载入规则（只取 stride 50 / n_estimators 4 的完整正式运行，
跳过 calib / smoke / pilot 与缺字段的文件，去重、内容不同就报错）、配对差与族内 Holm、缺方法时整族跳过、端到端写报告与 CSV、
--quick 不写文件。全部用小的合成 npz，不调用 TabPFN。"""
import csv
import os
import sys

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import analyze_round5 as A5  # noqa: E402
from src.eval.round2_metrics import ret, ret_segments  # noqa: E402
from src.eval.round3_analysis import holm  # noqa: E402

N = 4000
B = 2000


def fake_run(method, ds="gas", seed=0, stride=50, n_est=4, p_ok=0.8, max_rows=-1, drop=()):
    """N 行的流（原下标为偶数），留出行是每 20 个下标里的一个奇数；一个变点在 2000；检查点每 500 行。
    检查点上已开始阶段的留出行以概率 p_ok 预测正确，未开始阶段为 -1。"""
    rng = np.random.default_rng(1000 * seed + len(method))
    stream_idx = np.arange(0, 2 * N, 2)
    hold_idx = np.arange(1, 2 * N, 20)
    pos = np.searchsorted(stream_idx, hold_idx)
    y = rng.integers(0, 2, N)
    y_hold = np.random.default_rng(seed).integers(0, 2, len(hold_idx))     # 同一 seed 的方法共用留出集
    y = np.random.default_rng(seed + 7).integers(0, 2, N)
    regime_hold = (pos >= B).astype(int)
    ckpt_t = np.arange(500, N + 1, 500)
    P = np.full((len(ckpt_t), len(hold_idx)), -1, dtype=np.int8)
    for c, t in enumerate(ckpt_t):
        seen = regime_hold <= (1 if t > B else 0)
        ok = rng.random(len(hold_idx)) < p_ok
        P[c, seen] = np.where(ok, y_hold, 1 - y_hold)[seen]
    pred = y.copy()
    pred[:200] = -1
    d = dict(dataset=np.array([ds]), method=np.array([method]), seed=np.array([seed]), stride=np.array([stride]),
             n_estimators=np.array([n_est]), max_stream_rows=np.array([max_rows]), start=np.array([200]),
             commit=np.array(["c5"]), device=np.array(["cpu"]), leak_count=np.array([0]), max_ctx=np.array([10]),
             budget=np.array([200]), y=y, pred=pred, stream_idx=stream_idx, hold_idx=hold_idx, y_hold=y_hold,
             regime_hold=regime_hold, boundaries=np.array([B]), ckpt_t=ckpt_t, ckpt_pred=P)
    for k in drop:
        d.pop(k)
    return d


def save(dir_, name, d):
    os.makedirs(dir_, exist_ok=True)
    np.savez(os.path.join(dir_, name), **d)


def fname(ds, m, s, stride=50, n_est=4, tag=""):
    pre = "" if ds == "insects" else f"{ds}_"
    return f"r2_{pre}{m}_n{n_est}_S{stride}_seed{s}{tag}.npz"


# ---------------------------------------------------------------- 载入规则

def test_collect_filters(tmp_path):
    d2 = str(tmp_path / "r2")
    save(d2, fname("gas", "sw200", 0), fake_run("sw200"))
    save(d2, fname("gas", "dual200", 0, stride=10), fake_run("dual200", stride=10))           # stride 10
    save(d2, fname("gas", "dual200_rb", 0, n_est=1), fake_run("dual200_rb", n_est=1))         # n_estimators 1
    save(d2, fname("gas", "dual1000", 0, tag="_calib3"), fake_run("dual1000"))
    save(d2, fname("gas", "dual1000_rb", 0, tag="_smoke"), fake_run("dual1000_rb"))
    save(d2, "r2_gas_sw1000_pilot_seed0.npz", fake_run("sw1000"))
    save(d2, "r2_transfer_n4.npz", dict(matrix=np.zeros(3), n_estimators=np.array([4])))      # 缺 method / ckpt_pred
    save(d2, fname("gas", "dual200_cbrs", 0), fake_run("dual200_cbrs", drop=("ckpt_pred",)))
    save(d2, fname("gas", "dual200_rid", 0), fake_run("dual200_rid", max_rows=1000))         # 截断
    save(d2, fname("gas", "arch_union", 0), fake_run("arch_union"))                          # 不属于本分析
    runs, src, notes = A5.collect_runs([d2])
    assert set(runs) == {("gas", "sw200", 0)}
    txt = "\n".join(notes)
    for frag in ("calib/smoke/pilot", "缺 method 或 ckpt_pred", "stride ≠ 50", "截断运行"):
        assert frag in txt
    assert "3 个文件（名字含 calib/smoke/pilot）" in txt
    assert "2 个文件（缺 method 或 ckpt_pred）" in txt
    assert "2 个文件（stride ≠ 50 或 n_estimators ≠ 4）" in txt


def test_dataset_defaults_to_insects(tmp_path):
    d = fake_run("sw200")
    d.pop("dataset")
    save(str(tmp_path), fname("insects", "sw200", 3), {**d, "seed": np.array([3])})
    runs, _, _ = A5.collect_runs([str(tmp_path)])
    assert set(runs) == {("insects", "sw200", 3)}


def test_duplicates_identical_ok_different_raises(tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    save(a, fname("gas", "sw200", 0), fake_run("sw200"))
    save(b, fname("gas", "sw200", 0), {**fake_run("sw200"), "commit": np.array(["other"])})   # 结果相同，只是元数据不同
    runs, src, notes = A5.collect_runs([a, b])
    assert src[("gas", "sw200", 0)].startswith(a)
    assert any("内容相同" in n for n in notes)
    c = str(tmp_path / "c")
    save(c, fname("gas", "sw200", 0), fake_run("sw200", p_ok=0.3))
    with pytest.raises(A5.DuplicateRunError):
        A5.collect_runs([a, c])


def test_seed_filter(tmp_path):
    for s in range(3):
        save(str(tmp_path), fname("gas", "sw200", s), fake_run("sw200", seed=s))
    runs, _, _ = A5.collect_runs([str(tmp_path)], seeds={0, 2})
    assert sorted(k[2] for k in runs) == [0, 2]


# ---------------------------------------------------------------- 指标与配对差

def test_run_metrics_uses_round2_functions():
    d = fake_run("dual200_rb")
    m = A5.run_metrics5(d)
    assert m["RET"] == ret(d)
    assert m["RET_seg2000"] == ret_segments(d, 2000)
    assert m["RET_seg1000"] == ret_segments(d, 1000)
    # N = 4000、最后一个检查点 t = 4000：段 [2000, 4000) 要 t > 4000，没有 → 只剩段 0；lag 5000 与 L = 5000 都没有可用段
    assert np.isnan(m["RET_seg2000_lag5000"]) and m["n_RET_seg2000_lag5000"] == 0
    assert m["n_RET_seg2000"] == 1 and m["n_RET_seg1000"] == 3 and m["n_RET_seg5000"] == 0


def test_compare_paired_and_holm():
    seeds = list(range(5))
    rng = np.random.default_rng(0)
    S = {}
    for ds in A5.STREAMS:
        for n in A5.SIZES:
            base = rng.random(5)
            gain = rng.normal(0.02, 0.01, 5)
            for s in seeds:
                S[(ds, f"dual{n}", s)] = {"X": base[s]}
                S[(ds, f"dual{n}_rb", s)] = {"X": base[s] + gain[s]}
    res = A5.compare(S, seeds, "ours", "cbdm", "X")
    assert len(res) == 8
    ps = []
    for (ds, n), c in res.items():
        a = A5.vals(S, ds, f"dual{n}_rb", "X", seeds)
        b = A5.vals(S, ds, f"dual{n}", "X", seeds)
        assert c["mean"] == pytest.approx((a - b).mean())
        assert c["p"] == pytest.approx(stats.ttest_rel(a, b).pvalue)
        assert c["npos"] == int(((a - b) > 0).sum())
        ps.append(c["p"])
    assert [c["p_holm"] for c in res.values()] == pytest.approx(list(holm(ps)))
    assert A5.compare(S, seeds, "ours", "cbrs", "X") == {}          # 缺方法 → 空


# ---------------------------------------------------------------- 端到端

def build_dirs(tmp_path, with_cbrs=True):
    r2, r3, r5 = (str(tmp_path / k) for k in ("r2", "r3", "r5"))
    for s in range(3):
        for m, p in (("sw200", 0.6), ("dual200", 0.7), ("dual200_rb", 0.8)):
            save(r3, fname("gas", m, s), fake_run(m, seed=s, p_ok=p))
        if with_cbrs:
            save(r5, fname("gas", "dual200_cbrs", s), fake_run("dual200_cbrs", seed=s, p_ok=0.72))
            save(r5, fname("gas", "dual200_rb_adwin", s), fake_run("dual200_rb_adwin", seed=s, p_ok=0.75))
    save(r2, fname("insects", "sw200", 0, stride=10), fake_run("sw200", ds="insects", stride=10))
    return r2, r3, r5


def test_end_to_end(tmp_path):
    r2, r3, r5 = build_dirs(tmp_path)
    out, cs = str(tmp_path / "rep.md"), str(tmp_path / "ps.csv")
    S = A5.main(["--round2_dir", r2, "--round3_dir", r3, "--round5_dir", r5, "--round6_dir", "", "--seeds", "0-2",
                 "--out", out, "--csv", cs,
                 "--round2_csv", "", "--round3_csv", ""])
    assert len(S) == 15
    rep = open(out).read()
    assert "### B：ours − cbrs" in rep and "### A：ours_adwin − cbdm" in rep
    assert "### C1：rid − cbdm\n\n尚无数据" in rep
    assert "| gas | 200 |" in rep
    rows = list(csv.DictReader(open(cs)))
    assert len(rows) == 15
    r = next(x for x in rows if x["method"] == "dual200_rb" and x["seed"] == "1")
    assert r["role"] == "ours" and r["size"] == "200"
    assert float(r["RET_seg2000"]) == pytest.approx(S[("gas", "dual200_rb", 1)]["RET_seg2000"] * 100, abs=1e-6)
    # ours（p_ok 0.8）应高于 cbdm（0.7）：B 族也应该是正的（cbrs 0.72）
    res = A5.compare(S, [0, 1, 2], "ours", "cbrs", "RET_seg2000")
    assert res[("gas", 200)]["mean"] > 0


def test_quick_writes_nothing(tmp_path, capsys, monkeypatch):
    r2, r3, r5 = build_dirs(tmp_path, with_cbrs=False)
    monkeypatch.chdir(tmp_path)
    A5.main(["--round2_dir", r2, "--round3_dir", r3, "--round5_dir", r5, "--round6_dir", "", "--quick",
             "--round2_csv", "", "--round3_csv", ""])
    out = capsys.readouterr().out
    assert "seed 0–0（1 个）" in out and "均值与配对差没有比" in out
    assert not os.path.exists(tmp_path / "results")


# ---------------------------------------------------------------- 第五、六轮补充：RET_worst、ADWIN 报警、阶段编号列计数

def test_ret_worst_is_min_over_regimes():
    from src.eval.round2_metrics import ret_per_regime
    d = fake_run("dual200_rbr", seed=3)
    m = A5.run_metrics5(d)
    assert m["RET_worst"] == pytest.approx(min(ret_per_regime(d).values()))
    assert "dual200_rbr" in A5.METHOD_ROLE and A5.METHOD_ROLE["dual200_rbr"] == ("rbr", 200)


def test_alarm_stats():
    d = dict(boundaries=np.array([1000, 5000, 9000]), closures=np.array([1300, 3000, 5600, 12000]))
    a = A5.alarm_stats(d)
    assert a["n_alarms"] == 4 and a["n_cps"] == 3
    assert a["cp_hit"] == pytest.approx(2 / 3)          # 1000→1300、5000→5600 有；9000 之后 1000 行内没有
    assert a["far_alarms"] == 2                         # 3000 与 12000 离任何变点都超过 1000 行
    # 检出：1000 之后第一次报警 1300（延迟 300）；5000 之后 5600（600）；9000 之后 12000（3000，流末之前）
    assert a["detected"] == pytest.approx(1.0) and a["median_delay"] == pytest.approx(600)
    d2 = dict(boundaries=np.array([1000, 5000]), closures=np.array([6000]), y=np.zeros(8000))
    b2 = A5.alarm_stats(d2)                             # 1000 在 5000 之前没有报警（漏报）；5000 → 6000
    assert b2["detected"] == pytest.approx(0.5) and b2["median_delay"] == pytest.approx(1000)


def test_alarm_chance():
    # 报警紧跟变点（延迟 10 行）；整体循环平移后，1000 行内有报警的比例约 0.2（两次报警各覆盖 1000 / 10000），
    # 检出不再是 100 %（平移后两次报警可能都落在另一段），延迟远大于 10 行
    b, y = np.array([2000, 6000]), np.zeros(10000)
    d = dict(boundaries=b, closures=np.array([2010, 6010]), y=y)
    real, ch = A5.alarm_stats(d), A5.alarm_chance(d)
    assert real["cp_hit"] == 1.0 and real["detected"] == 1.0 and real["median_delay"] == 10
    assert ch["chance_hit"] == pytest.approx(0.2, abs=0.05)
    assert 0.5 < ch["chance_detected"] < 1.0 and ch["chance_delay"] > 500
    assert A5.alarm_chance(d) == ch                     # 确定性：平移量等距，不用随机数
    assert np.isnan(A5.alarm_chance(dict(boundaries=b, closures=np.array([], dtype=np.int64), y=y))["chance_hit"])


def test_id_column_counts():
    # 三个阶段，终点 100 / 200；检查点 50（没有已结束阶段，不计）、150、250、300
    ts = np.array([50, 150, 250, 300])
    mem = np.array([[5, 5, 0, 1],      # 阶段 0
                    [0, 3, 2, 2],      # 阶段 1
                    [0, 0, 4, 4]])     # 阶段 2
    rh = np.array([0] * 10 + [1] * 30 + [2] * 5)
    c = A5.id_column_counts(mem, ts, [50, 150, 250, 300], [100, 200], rh)
    # t=150：2 种编号（类别），已结束 {0}，0 在上下文里 → 缺失 0
    # t=250：2 种（类别），已结束 {0,1}，0 不在 → 10/40；t=300：3 种（类别），都在 → 0
    assert (c["const"], c["cat"], c["num"]) == (0, 3, 0)
    assert c["absent"] == pytest.approx((0 + 10 / 40 + 0) / 3)
