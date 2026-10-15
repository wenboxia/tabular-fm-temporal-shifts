"""第三轮分析（src/eval/round3_analysis.py 与 scripts/analyze_round3.py）的关键函数：
载入键、一致性检查、配对检验与 Holm、过渡区切片、ADWIN 偏移符号、重放 digest 与真实 run_stream 一致、--quick 报缺失。
全部用合成数据与哑预测器，不调用 TabPFN。"""
import os
import sys

import numpy as np
import pytest
from scipy import stats

import src.eval.round3_analysis as R3
from src.eval.round3_analysis import (adwin_alarm_info, age_summary, baseline_acc, baselines_full, check_consistency,
                                      class_coverage, class_share, expected_checkpoints, final_pool_by_position,
                                      find_missing, first_diff_row, holdout_class_recall, holm, last_present_time,
                                      load_runs, mask_removed_share, mean_sd_ci, noninferiority, overlap_summary,
                                      paired, parse_method, quick_check, regime_leftover, replay_run, rolling_ba,
                                      run_batches, run_metrics, spearman, stage_auc, stream_acc_per_regime, zone_metrics)
from src.eval.stream_runner import run_stream
from src.memory.context_memory import make_memory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import analyze_round3 as A  # noqa: E402


class StubDetector:
    """在给定的"行序号"（第几次 update，从 0 起）返回 True。"""

    def __init__(self, fire_at):
        self.fire_at, self.i = set(fire_at), -1

    def update(self, _e):
        self.i += 1
        return self.i in self.fire_at


def make_run(method="dual200_rb", dataset="gas", seed=0, n=1500, K=3, boundaries=(500, 1000), mem_boundaries=None,
             fire_at=(), zone=None, max_rows=None, commit="abc", device="cpu", n_est=1):
    """用真实 run_stream + 多数类预测器产生一个 npz 风格的 dict。"""
    rng = np.random.default_rng(100 + seed)
    regime = np.zeros(n, int)
    for b in boundaries:
        regime[b:] += 1
    y = rng.integers(0, K, n)
    X = rng.normal(size=(n, 2)).astype(np.float32)
    hold_idx = np.arange(0, 60) * 20
    y_hold, regime_hold = y[hold_idx], regime[hold_idx]
    Xh = X[hold_idx]
    mem_boundaries = list(boundaries) if mem_boundaries is None else list(mem_boundaries)
    adwin = method.endswith("_adwin")
    mem = make_memory(method, X, y, K, seed)

    def predict(Xc, yc, Xq):
        p = np.zeros((len(Xq), K))
        p[:, int(np.bincount(yc, minlength=K).argmax())] = 1.0
        return p

    r = run_stream(mem, predict, X, y, regime, Xh, regime_hold, list(boundaries), stride=50, start=200, ckpt_every=500,
                   segmentation="adwin" if adwin else "official", detector=StubDetector(fire_at) if adwin else None,
                   mem_boundaries=mem_boundaries, max_rows=max_rows)
    return dict(
        dataset=np.array([dataset]), mem_boundaries=np.array(mem_boundaries), method=np.array([method]),
        seed=np.array([seed]), n_estimators=np.array([n_est]), stride=np.array([50]), ckpt_every=np.array([500]),
        start=np.array([200]), device=np.array([device]), commit=np.array([commit]),
        max_stream_rows=np.array([-1 if max_rows is None else max_rows]),
        zone=np.array(zone if zone is not None else [], dtype=np.int64).reshape(-1, 2),
        class_ids=np.array([str(i) for i in range(K)]), pred=r.pred, route=r.route, y=y, regime=regime,
        y_hold=y_hold, regime_hold=regime_hold, stream_idx=np.arange(n), hold_idx=hold_idx,
        boundaries=np.array(boundaries), ckpt_t=r.ckpt_t, ckpt_upto=r.ckpt_upto, ckpt_pred=r.ckpt_pred,
        closures=r.closures, n_fits=np.array([r.n_fits]), fit_seconds=np.array([r.fit_seconds]),
        elapsed=np.array([1.0]), max_ctx=np.array([r.max_ctx]), budget=np.array([mem.max_context()]),
        leak_count=np.array([r.leak_count]), digests=np.array(r.digests), overall_acc=np.array([0.5])), r


# ---------------------------------------------------------------- 载入与键

def test_load_keys_dataset_detection_and_calib_skipped(tmp_path):
    d_new, _ = make_run("dual200", dataset="gas", seed=1)
    d_old, _ = make_run("sw200", dataset="insects", seed=0)
    for k in ("dataset", "mem_boundaries", "zone", "class_ids"):      # 第二轮文件没有这些字段
        d_old.pop(k)
    np.savez(tmp_path / "r2_gas_dual200_n1_S50_seed1_t.npz", **d_new)
    np.savez(tmp_path / "r2_sw200_n1_S50_seed0_t.npz", **d_old)
    np.savez(tmp_path / "r2_sw200_n1_S50_seed0_calib3_t.npz", **d_old)      # calib3 不属于分析
    np.savez(tmp_path / "r2_gas_sw200_n1_S50_seed0.npz", **d_new)           # tag 不符
    runs, probs = load_runs([str(tmp_path)], 1, 50, "_t")
    assert set(runs) == {("gas", "dual200", 1), ("insects", "sw200", 0)} and probs == []
    # 文件名与内容不符会被报告
    np.savez(tmp_path / "r2_gas_dual1000_n1_S50_seed1_t.npz", **d_new)
    _, probs = load_runs([str(tmp_path)], 1, 50, "_t")
    assert any("文件名与内容不符" in p for p in probs)


def test_parse_method():
    assert parse_method("sw1000") == ("sw", 1000, False)
    assert parse_method("dual200") == ("kdd", 200, False)
    assert parse_method("dual200_rb") == ("ours", 200, False)
    assert parse_method("dual1000_rb_adwin") == ("ours", 1000, True)
    with pytest.raises(ValueError):
        parse_method("arch_routed")


# ---------------------------------------------------------------- 一致性

def test_consistency_detects_mismatch_and_incomplete():
    a, _ = make_run("sw200")
    b, _ = make_run("dual200")
    runs = {("gas", "sw200", 0): a, ("gas", "dual200", 0): b}
    assert check_consistency(runs) == []
    b2 = dict(b)
    b2["y"] = b["y"].copy()
    b2["y"][3] = (b2["y"][3] + 1) % 3
    probs = check_consistency({("gas", "sw200", 0): a, ("gas", "dual200", 0): b2})
    assert any("y 不同" in p for p in probs)
    b3 = dict(b)
    b3["commit"] = np.array(["zzz"])
    assert any("代码版本" in p for p in check_consistency({("gas", "sw200", 0): a, ("gas", "dual200", 0): b3}))
    # insects 合并两轮允许 commit / device 不同
    a_i, b_i = dict(a, dataset=np.array(["insects"])), dict(b3, dataset=np.array(["insects"]))
    assert check_consistency({("insects", "sw200", 0): a_i, ("insects", "dual200", 0): b_i}) == []
    # 完整运行里出现 -1
    c = dict(b)
    c["pred"] = b["pred"].copy()
    c["pred"][700] = -1
    assert any("-1" in p for p in check_consistency({("gas", "dual200", 0): c}))
    # 截断运行不要求 pred 覆盖到流末尾
    t, _ = make_run("dual200", max_rows=900)
    assert check_consistency({("gas", "dual200", 0): t}) == []


def test_find_missing():
    a, _ = make_run("sw200")
    miss = find_missing({("gas", "sw200", 0): a}, [0, 1], {"gas": ["sw200", "dual200"]})
    assert miss == {("gas", "sw200"): [1], ("gas", "dual200"): [0, 1]}


# ---------------------------------------------------------------- 统计

def test_paired_matches_scipy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0.8, 0.05, 10), rng.normal(0.78, 0.05, 10)
    pr = paired(a, b)
    d = a - b
    t = stats.ttest_rel(a, b)
    h = stats.t.ppf(0.975, 9) * d.std(ddof=1) / np.sqrt(10)
    assert pr["p"] == pytest.approx(t.pvalue) and pr["mean"] == pytest.approx(d.mean())
    assert pr["lo"] == pytest.approx(d.mean() - h) and pr["hi"] == pytest.approx(d.mean() + h)
    assert pr["npos"] == int((d > 0).sum()) and pr["n"] == 10
    # nan 的 seed 被丢掉
    a2 = np.r_[a, np.nan]
    assert paired(a2, np.r_[b, 0.5])["n"] == 10
    ms = mean_sd_ci(a)
    assert ms["sd"] == pytest.approx(a.std(ddof=1)) and ms["lo"] < ms["mean"] < ms["hi"]


def test_noninferiority_matches_round2_formula():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0.80, 0.01, 10), rng.normal(0.803, 0.01, 10)
    ni = noninferiority(a, b, 0.005)
    d = a - b
    assert ni["p"] == pytest.approx(stats.ttest_1samp(d + 0.005, 0, alternative="greater").pvalue)
    assert ni["lower"] == pytest.approx(d.mean() - stats.t.ppf(0.95, 9) * d.std(ddof=1) / np.sqrt(10))


def test_holm_hand_values():
    adj = holm([0.01, 0.04, 0.03])        # 排序 0.01×3=0.03, 0.03×2=0.06, 0.04×1→max(0.06,0.04)=0.06
    assert np.allclose(adj, [0.03, 0.06, 0.06])
    adj = holm([0.5, np.nan, 0.001])      # 家族大小 3；nan 按 1 处理且返回 nan
    assert adj[0] == pytest.approx(1.0) and np.isnan(adj[1]) and adj[2] == pytest.approx(0.003)
    assert np.all(holm([0.2]) == [0.2])


def test_spearman_small_n_is_nan():
    assert np.isnan(spearman([1, 2], [2, 1])[0])
    rho, p, n = spearman([1, 2, 3, 4], [10, 20, 25, 40])
    assert rho == pytest.approx(1.0) and n == 4


def test_stage_auc_separates_shifted_regimes():
    rng = np.random.default_rng(0)
    regime = np.repeat([0, 1, 2], 200)
    X = rng.normal(size=(600, 3))
    X[regime == 0] += 3.0                  # 阶段 0 与后面差别大，阶段 1 与阶段 2 相同
    auc = stage_auc(X, regime)
    assert set(auc) == {0, 1} and auc[0] > 0.99 and abs(auc[1] - 0.5) < 0.15


# ---------------------------------------------------------------- 过渡区

def test_zone_metrics_slices_thirds_and_first500():
    n = 3000
    y = np.tile([0, 1], n // 2)
    pred = y.copy()
    a, b = 600, 1500                               # 三等分：[600,900) [900,1200) [1200,1500)
    pred[900:1200] = 1 - y[900:1200]               # 第二段全错
    pred[1200:1500] = np.where(np.arange(1200, 1500) % 4 == 0, 1 - y[1200:1500], y[1200:1500])   # 第三段错 1/4
    pred[:200] = -1
    d = dict(y=y, pred=pred, zone=np.array([[a, b]]))
    z = zone_metrics(d)
    assert z["ZONE_T1"] == pytest.approx(1.0) and z["ZONE_T2"] == pytest.approx(0.0)
    assert z["ZONE_T3"] == pytest.approx(0.75)
    assert z["ZONE"] == pytest.approx((0.5 + 2 / 3) / 2)          # 类 0 召回 225/450，类 1 召回 300/450
    sl = slice(a, a + 500)
    from src.eval.round2_metrics import balanced_accuracy
    assert z["ZONE_500"] == pytest.approx(balanced_accuracy(y[sl], pred[sl]))
    assert zone_metrics(dict(y=y, pred=pred, zone=np.zeros((0, 2), int))) == {}
    # 过渡区之前尚未预测（截断）→ nan，不崩溃
    p2 = np.full(n, -1)
    p2[:300] = y[:300]
    assert np.isnan(zone_metrics(dict(y=y, pred=p2, zone=np.array([[a, b]])))["ZONE"])


def test_rolling_ba_shape_and_values():
    y = np.tile([0, 1], 1000)
    pred = y.copy()
    t, acc = rolling_ba(y, pred, 500, 50)
    assert np.allclose(acc, 1.0) and t[0] >= 250


# ---------------------------------------------------------------- ADWIN 偏移

def test_adwin_offsets_sign_and_r0_share():
    d, _ = make_run("dual200_rb_adwin", dataset="insects_gradual", boundaries=(1000,), zone=[[600, 900]],
                    fire_at=(450, 1000))                         # 报警行 = 200 + 450 = 650 → 关闭于 651
    a = adwin_alarm_info(d, replay_run(d, "dual200_rb_adwin"))
    assert a["alarms"][0] == 651 and a["off_zone"] == 51 and a["off_cp"] == -349 and a["further"] == len(a["alarms"]) - 1
    assert 0.0 <= a["r0_share"] <= 1.0
    d2, _ = make_run("dual200_rb_adwin", dataset="insects_gradual", boundaries=(1000,), zone=[[600, 900]],
                     fire_at=(1000,))                            # 关闭于 1201，在文档变点之后
    a2 = adwin_alarm_info(d2)
    assert a2["off_cp"] > 0 and a2["off_zone"] > 300
    d3, _ = make_run("dual200_rb_adwin", dataset="insects_gradual", boundaries=(1000,), zone=[[600, 900]])
    rep3 = replay_run(d3, "dual200_rb_adwin")
    a3 = adwin_alarm_info(d3, rep3)
    assert a3["first"] is None and a3["n_alarms"] == 0
    assert a3["r0_share"] == pytest.approx(rep3["comp"][0, -1] / rep3["long_len"][-1])      # 无报警 = dual*，池里是最近的行


# ---------------------------------------------------------------- 重放

@pytest.mark.parametrize("method", ["dual200", "dual200_rb", "dual1000_rb"])
def test_replay_digests_match_real_run(method):
    d, r = make_run(method, boundaries=(500, 1000), mem_boundaries=(500, 1000))
    rep = replay_run(d, method)                       # 内部断言 digests / closures / 检查点时刻一致
    assert all(rep["verified"].values()) and rep["ckpt_t"].tolist() == r.ckpt_t.tolist()
    assert rep["comp"].shape[0] == 3 and rep["comp"][:, -1].sum() == rep["long_len"][-1]


def test_replay_mem_boundaries_differ_from_eval_boundaries():
    """渐变臂 (b)：评估边界 1000，记忆在 800 关闭；批次必须在两处都切开。"""
    d, _ = make_run("dual200_rb", boundaries=(1000,), mem_boundaries=(800,))
    assert d["closures"].tolist() == [800]
    rep = replay_run(d, "dual200_rb")
    assert 800 in rep["ts"].tolist() and 1000 in rep["ts"].tolist()


def test_replay_adwin_uses_recorded_closures():
    d, _ = make_run("dual200_rb_adwin", boundaries=(500, 1000), fire_at=(350, 900))
    assert len(d["closures"]) == 2
    rep = replay_run(d, "dual200_rb_adwin")
    assert rep["closures"].tolist() == d["closures"].tolist()


def test_replay_truncated_run():
    d, _ = make_run("dual200_rb", max_rows=900)
    replay_run(d, "dual200_rb")


def test_replay_detects_tampered_digest():
    d, _ = make_run("dual200_rb")
    d["digests"] = d["digests"].copy()
    d["digests"][2] = "0" * 16
    with pytest.raises(AssertionError):
        replay_run(d, "dual200_rb")
    with pytest.raises(AssertionError):                    # 方法名不对（rb 与非 rb 的 digest 不同）也会被发现
        replay_run(make_run("dual200_rb")[0], "dual200")


def test_overlap_of_dual_with_sliding_window():
    d, _ = make_run("dual200")
    s = overlap_summary(replay_run(d, "dual200"))
    assert 0.0 < s["min_full"] <= s["mean_full"] <= 1.0
    d1000, _ = make_run("dual1000")           # 记忆未满（1500 行流里 t ≥ 1000 才满）：全部批次均值 ≥ 已满均值
    s = overlap_summary(replay_run(d1000, "dual1000"))
    assert s["mean_all"] >= s["mean_full"] - 1e-12


def test_expected_checkpoints_match_runner():
    for bnds, mb, mr in [((500, 1000), (500, 1000), None), ((1000,), (800,), None), ((500, 1000), (500, 1000), 900)]:
        d, r = make_run("dual200_rb", boundaries=bnds, mem_boundaries=mb, max_rows=mr)
        n = 1500 if mr is None else mr
        assert expected_checkpoints(n, 200, 50, 500, bnds, mb) == r.ckpt_t.tolist()


# ---------------------------------------------------------------- 指标与 --quick

def test_run_metrics_keys_and_values():
    d, _ = make_run("dual200_rb", dataset="insects_gradual", boundaries=(500, 1000), zone=[[300, 600]])
    m = run_metrics(d)
    for k in ("ADAPT", "ADAPT_masked", "RET", "RET_R0", "RET_R1", "FGT", "ZONE", "ZONE_T3", "ZONE_500"):
        assert k in m and np.isfinite(m[k]), k
    assert m["RET"] == pytest.approx((m["RET_R0"] + m["RET_R1"]) / 2)


def test_quick_check_flags_problems():
    d, _ = make_run("dual200_rb", dataset="gas", max_rows=900)
    q = quick_check(d)
    assert [r for r in q["reasons"] if "ADAPT" not in r] == [], q["reasons"]      # 随机标签的 ADAPT 低于下限是预期的
    assert any("低于 1/K+10pp" in r for r in q["reasons"])
    bad = dict(d)
    bad["closures"] = np.array([500, 999])
    bad["zone"] = np.array([[1, 2]])
    bad["leak_count"] = np.array([1])
    bad["pred"] = d["pred"].copy()
    bad["pred"][300:400] = -1
    reasons = " ".join(quick_check(bad)["reasons"])
    for frag in ("closures", "zone", "leak_count", "行数不完整"):
        assert frag in reasons
    nan_ret, _ = make_run("dual200_rb", dataset="gas", max_rows=900, boundaries=(500, 1000))
    nr = dict(nan_ret)
    nr["ckpt_pred"] = np.full_like(nan_ret["ckpt_pred"], -1)          # 检查点没有评估任何留出行 → RET = nan
    assert any("RET 为 nan" in r for r in quick_check(nr)["reasons"])


def test_quick_reports_missing_rows(capsys):
    d, _ = make_run("sw200", dataset="gas", seed=0)

    class Args:
        dir, tag, n_estimators, quick_seed = "x", "_t", 1, 0
    A.quick(Args, {("gas", "sw200", 0): d}, {("gas", "sw200", 0)}, [])
    out = capsys.readouterr().out
    assert "(dual200) 缺失" in out and "缺 dual1000_rb" in out and "需要看" in out
    assert "[insects]\n  暂无结果文件" in out and "总结：1 个数据集已有结果" in out


def test_full_analysis_runs_on_tiny_dir(tmp_path, monkeypatch):
    """端到端：小目录跑完整分析（含缺失方法），不崩溃，写出报告与 CSV。"""
    out_dir = tmp_path / "npz"
    out_dir.mkdir()
    for s in (0, 1):
        for m in ("sw200", "dual200", "dual200_rb"):
            d, _ = make_run(m, dataset="gas", seed=s)
            np.savez(out_dir / f"r2_gas_{m}_n1_S50_seed{s}_t.npz", **d)
    monkeypatch.setattr(sys, "argv", ["analyze_round3.py", "--dir", str(out_dir), "--round2_dir", str(tmp_path / "none"),
                                      "--tag", "_t", "--n_estimators", "1", "--seeds", "0-1",
                                      "--out", str(tmp_path / "r.md"), "--csv", str(tmp_path / "r.csv")])
    monkeypatch.setattr(A, "section_stage", lambda *a, **k: None)      # 不加载真实数据流
    A.main()
    txt = (tmp_path / "r.md").read_text()
    assert "没有成功门槛" in txt and "缺失" in txt and "Holm" in txt
    assert (tmp_path / "r.csv").read_text().count("\n") == 7


# ---------------------------------------------------------------- 后补的测试（锁住关键量）

def _fake_S_runs(seeds, n_seed_missing=0):
    """gas 200 行：为 sw200 / dual200 / dual200_rb 造 runs（只用于存在性与边界数）和手填的指标 S。"""
    base, _ = make_run("dual200_rb", dataset="gas")
    runs, S = {}, {}
    rng = np.random.default_rng(5)
    for s in seeds:
        e = rng.normal(0, 0.003)
        vals_ = {   # ours / KDD / sw 的指标刻意互不相同，用来发现取错列或调换基线
            "dual200_rb": dict(ADAPT=0.80 + e, ADAPT_masked=0.79 + e, RET=0.60 + 2 * e, RET_R0=0.50 + e, RET_R1=0.70 - e,
                               FGT=0.02 + abs(e)),
            "dual200": dict(ADAPT=0.79 + 0.5 * e, ADAPT_masked=0.78 + 0.5 * e, RET=0.55 - e, RET_R0=0.40 + 2 * e,
                            RET_R1=0.70 + 2 * e, FGT=0.05 + 2 * abs(e)),
            "sw200": dict(ADAPT=0.77 - e, ADAPT_masked=0.76 - e, RET=0.45 + 3 * e, RET_R0=0.20 - e, RET_R1=0.65 + e,
                          FGT=0.10 + 3 * abs(e)),
        }
        for m, v in vals_.items():
            runs[("gas", m, s)] = base
            S[("gas", m, s)] = v
    return runs, S


def _section(fn, *args):
    out = []
    fn(lambda *a: out.append(" ".join(str(x) for x in a)), *args)
    return "\n".join(out)


def test_section_paired_uses_right_columns_baselines_and_holm():
    seeds = list(range(6))
    runs, S = _fake_S_runs(seeds)
    txt = _section(A.section_paired, runs, S, seeds)
    o = lambda k, m: np.array([S[("gas", m, s)][k] for s in seeds])      # noqa: E731
    # ours − KDD 与 ours − 滑窗不能对调：每个块的第一行是 KDD，第二行是滑窗
    for k_idx, k in enumerate(("ADAPT", "ADAPT_masked", "RET")):
        for name, base in (("ours − KDD", "dual200"), ("ours − 滑窗", "sw200")):
            line = next(ln for ln in txt.splitlines() if ln.startswith(f"| {name} |") and ln.count("p=") == 3)
            cells = [c.strip() for c in line.strip("|").split("|")][1:]
            pr = paired(o(k, "dual200_rb"), o(k, base))
            assert cells[k_idx].startswith(f"{pr['mean'] * 100:+.2f}"), (name, k, cells[k_idx])
    # 各阶段与 FGT 的配对差表存在，且数值对
    assert "FGT 越低越好" in txt and "FGT 差为负 = ours 遗忘更少" in txt
    hdr = next(ln for ln in txt.splitlines() if ln.startswith("| 比较 | R0 | R1 | FGT"))
    assert hdr
    i = txt.splitlines().index(hdr)
    kdd_row, sw_row = txt.splitlines()[i + 2], txt.splitlines()[i + 3]
    for line, base in ((kdd_row, "dual200"), (sw_row, "sw200")):
        cells = [c.strip() for c in line.strip("|").split("|")][1:]
        for c, k in zip(cells, ("RET_R0", "RET_R1", "FGT")):
            assert c.startswith(f"{paired(o(k, 'dual200_rb'), o(k, base))['mean'] * 100:+.2f}"), (base, k, c)
    # Holm：用 RET、对 KDD、家族大小 8（只有 gas 200 有数据，其余 nan）
    pr = paired(o("RET", "dual200_rb"), o("RET", "dual200"))
    lines_ = txt.splitlines()
    holm_line = next(ln for ln in lines_[next(i for i, x in enumerate(lines_) if "Holm 校正" in x):]
                     if ln.startswith("| gas | 200 |"))
    assert f"| {pr['p']:.3g} | {min(1.0, 8 * pr['p']):.3g} |" in holm_line
    assert f"{pr['mean'] * 100:+.2f}" in holm_line
    # 拿 sw 或 ADAPT 算出来的 p 不同，不会误过
    assert paired(o("ADAPT", "dual200_rb"), o("ADAPT", "dual200"))["p"] != pytest.approx(pr["p"])
    assert paired(o("RET", "dual200_rb"), o("RET", "sw200"))["p"] != pytest.approx(pr["p"])


def test_cell_marks_n_when_seeds_missing():
    st = mean_sd_ci([0.8, 0.82, np.nan, 0.81])
    assert "（n=3）" in A.cell(st, 4) and "（n=" not in A.cell(st, 3) and "（n=" not in A.cell(st)
    assert A.cell(mean_sd_ci([0.8]), 4).endswith("（n=1）")
    seeds = list(range(5))
    runs, S = _fake_S_runs(seeds)
    S[("gas", "dual200", 4)]["RET"] = float("nan")          # 文件在，但某个指标是 nan：表里也要看得出来
    txt = _section(A.section_reliability, runs, S, seeds)
    dual_row = next(ln for ln in txt.splitlines() if ln.startswith("| dual200 |"))
    assert "（n=4）" in dual_row and "（n=" not in next(ln for ln in txt.splitlines() if ln.startswith("| sw200 |"))
    assert "各已结束阶段 FGT_j" in txt


def test_consistency_catches_each_field_and_budget():
    a, _ = make_run("sw200")
    b, _ = make_run("dual200")
    for field, new in (("hold_idx", a["hold_idx"] + 1), ("stream_idx", a["stream_idx"] + 1),
                       ("boundaries", np.array([500, 1100])), ("mem_boundaries", np.array([500, 900]))):
        bad = dict(b)
        bad[field] = new
        probs = check_consistency({("gas", "sw200", 0): a, ("gas", "dual200", 0): bad})
        assert any(f"{field} 不同" in p for p in probs), field
    bad = dict(b)
    bad["max_ctx"] = np.array([int(b["budget"][0]) + 1])
    assert any("超预算" in p for p in check_consistency({("gas", "dual200", 0): bad}))


def test_consistency_across_gradual_arms():
    a, _ = make_run("dual200_rb", dataset="insects_gradual", boundaries=(1000,), mem_boundaries=(1000,), zone=[[600, 900]])
    b, _ = make_run("dual200_rb", dataset="insects_gradual_start", boundaries=(1000,), mem_boundaries=(600,),
                    zone=[[600, 900]])
    runs = {("insects_gradual", "dual200_rb", 0): a, ("insects_gradual_start", "dual200_rb", 0): b}
    assert check_consistency(runs) == []                     # mem_boundaries 本来就该不同
    for field in ("y", "hold_idx", "stream_idx", "boundaries", "regime_hold"):
        bad = dict(b)
        bad[field] = (np.asarray(b[field]) + 1) % 3 if field in ("y", "regime_hold") else np.asarray(b[field]) + 1
        probs = check_consistency({**runs, ("insects_gradual_start", "dual200_rb", 0): bad})
        assert any(f"insects_gradual 的 {field} 不同" in p for p in probs), field
    bad = dict(b, zone=np.array([[601, 900]]))
    assert any("zone 不同" in p for p in check_consistency({**runs, ("insects_gradual_start", "dual200_rb", 0): bad}))
    bad = dict(b, commit=np.array(["other"]))
    assert any("commit 不同" in p for p in check_consistency({**runs, ("insects_gradual_start", "dual200_rb", 0): bad}))


def test_overlap_matches_independent_recomputation_and_full_window():
    d, _ = make_run("dual200", boundaries=(500, 1000))
    rep = replay_run(d, "dual200")
    # 独立重算：用 run_stream 与同一记忆类，窗口 = 记忆预算 N
    y, regime = d["y"], d["regime"]
    mem = make_memory("dual200", np.zeros((len(y), 1), np.float32), y, 3, 0)
    N, got, orig = mem.max_context(), [], mem.update

    def upd(idx):
        orig(idx)
        t = int(np.max(idx)) + 1
        got.append(float((mem._all() >= t - N).mean()))
    mem.update = upd
    run_stream(mem, lambda Xc, yc, Xq: np.full((len(Xq), 3), 1 / 3), np.zeros((len(y), 1), np.float32), y, regime,
               np.zeros((60, 1), np.float32), d["regime_hold"], [500, 1000], stride=50, start=200, ckpt_every=500,
               segmentation="official", mem_boundaries=[500, 1000])
    assert np.allclose(rep["overlap"], got)
    # 长期池里有早期行，所以记忆已满之后重合 < 1；记忆刚好装下所有行（t ≤ N）时 = 1
    assert (rep["overlap"][rep["ts"] <= N] == 1.0).all() and rep["overlap"][rep["full"]].min() < 1.0


def test_pool_composition_is_per_regime_not_all_r0():
    d, _ = make_run("dual200_rb", boundaries=(500, 1000))
    rep = replay_run(d, "dual200_rb")
    assert rep["comp"][1:, -1].sum() > 0 and rep["comp"][0, -1] < rep["long_len"][-1]
    assert (rep["comp"].sum(axis=0) == rep["long_len"]).all()


def test_baselines_use_previous_label_only():
    n = 800
    y = np.tile([0, 1], n // 2)
    pred = np.full(n, -1)
    pred[200:] = 0
    nc, maj = baseline_acc(dict(y=y, pred=pred))
    assert nc == pytest.approx(0.0)                        # 交替标签：上一个标签永远错；用当前标签（泄露）会是 1
    y2 = np.r_[np.zeros(400, int), np.ones(400, int)]
    pred2 = np.full(n, -1)
    pred2[200:] = 0
    nc2, maj2 = baseline_acc(dict(y=y2, pred=pred2))
    done = np.arange(200, n)
    exp_nc = (y2[done - 1], y2[done])
    from src.eval.round2_metrics import balanced_accuracy
    assert nc2 == pytest.approx(balanced_accuracy(exp_nc[1], exp_nc[0]))
    exp_maj = np.array([np.bincount(y2[t - 200:t], minlength=2).argmax() for t in done])
    assert maj2 == pytest.approx(balanced_accuracy(y2[done], exp_maj))


def test_fgt_is_mean_over_regimes_not_max():
    d, _ = make_run("dual200_rb", boundaries=(500, 1000), n=1500)
    m = run_metrics(d)
    f = [m["FGT_R0"], m["FGT_R1"]]
    assert m["FGT"] == pytest.approx(np.mean(f))
    if f[0] != f[1]:
        assert m["FGT"] != pytest.approx(max(f))
    # 手填 ckpt_pred：R0 遗忘 0.3，R1 遗忘 0.1 → FGT = 0.2
    from src.eval import round3_analysis as mod
    import unittest.mock as um
    with um.patch.object(mod, "fgt_per_regime", lambda _d: {0: 0.3, 1: 0.1}):
        assert mod.run_metrics(d)["FGT"] == pytest.approx(0.2)


def test_quick_check_structural_failures(monkeypatch):
    d, _ = make_run("dual200_rb", dataset="gas", max_rows=900, boundaries=(500, 1000))
    # 少一个检查点
    bad = dict(d, ckpt_t=d["ckpt_t"][:-1], digests=d["digests"][:-1])
    assert any("与按规则推出的" in r for r in quick_check(bad)["reasons"])
    shifted = d["ckpt_t"].copy()
    shifted[0] += 50
    assert any("与按规则推出的" in r for r in quick_check(dict(d, ckpt_t=shifted))["reasons"])
    # digests 数不匹配
    assert any("digests" in r for r in quick_check(dict(d, digests=d["digests"][:-1]))["reasons"])
    # 完整运行却在 CPU 上跑
    full, _ = make_run("dual200_rb", dataset="gas", device="cpu")
    assert any("CPU" in r for r in quick_check(full)["reasons"])
    assert not any("CPU" in r for r in quick_check(dict(full, device=np.array(["cuda"])))["reasons"])
    # 没有阶段结束却算出了 RET
    early, _ = make_run("dual200_rb", dataset="gas", max_rows=400, boundaries=(500, 1000))
    assert not any("RET" in r for r in quick_check(early)["reasons"])
    monkeypatch.setattr(R3, "run_metrics", lambda _d: dict(ADAPT=0.9, ADAPT_masked=0.9, RET=0.5))
    assert any("没有评估阶段结束却算出了 RET" in r for r in quick_check(early)["reasons"])


class _Args:
    dir, tag, n_estimators, quick_seed = "x", "_t", 1, 0


def test_quick_truncated_official_run_is_flagged_but_not_in_smoke(capsys):
    d, _ = make_run("sw200", dataset="gas", max_rows=900)
    runs = {("gas", "sw200", 0): d}
    A.quick(_Args, runs, set(runs), [])
    assert "截断运行" in capsys.readouterr().out and "不是正式结果" not in capsys.readouterr().out      # _t 视为自测
    args = type("A2", (), dict(dir="x", tag="", n_estimators=1, quick_seed=0))
    A.quick(args, runs, set(runs), [])
    assert "!! 截断运行（max_stream_rows=900），不是正式结果" in capsys.readouterr().out


def test_quick_dataset_without_round3_files_is_not_flagged(capsys):
    d, _ = make_run("sw200", dataset="insects", seed=0)
    runs = {("insects", "sw200", 0): d}
    A.quick(_Args, runs, set(), [])                                 # 只有第二轮文件，第三轮还没跑到
    out = capsys.readouterr().out
    assert "(dual200) 尚未运行" in out and "缺 dual200" not in out and "还没有该数据集的结果" in out
    assert "需要看：insects" not in out
    A.quick(_Args, runs, {("insects", "sw200", 0)}, [])           # 第三轮已有 insects 文件但缺方法 → 需要看
    assert "缺 dual200" in capsys.readouterr().out


def test_load_all_warns_on_wrong_directory(tmp_path):
    args = type("A3", (), dict(dir=str(tmp_path / "nope"), round2_dir=str(tmp_path / "none2"), n_estimators=4, tag=""))
    _, probs, keys3 = A.load_all(args)
    assert keys3 == set() and any("不存在" in p and "reproduce_round3.sh" in p for p in probs)
    (tmp_path / "empty").mkdir()
    args.dir = str(tmp_path / "empty")
    _, probs, _ = A.load_all(args)
    assert any("里没有 r2_" in p for p in probs)


def test_main_exits_when_round3_dir_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["analyze_round3.py", "--dir", str(tmp_path / "nope"), "--round2_dir",
                                      str(tmp_path / "none"), "--out", str(tmp_path / "r.md"), "--csv", ""])
    with pytest.raises(SystemExit) as e:
        A.main()
    assert "不存在" in str(e.value)


def test_replay_failure_counted_in_summary():
    ok_d, _ = make_run("dual200_rb", dataset="gas", seed=0)
    bad_d, _ = make_run("dual200", dataset="gas", seed=0)
    bad_d["digests"] = bad_d["digests"].copy()
    bad_d["digests"][1] = "0" * 16
    runs = {("gas", "dual200_rb", 0): ok_d, ("gas", "dual200", 0): bad_d}
    cache, probs = {}, []
    A.get_replay(cache, probs, runs, "gas", "dual200_rb", 0)
    A.get_replay(cache, probs, runs, "gas", "dual200", 0)
    A.get_replay(cache, probs, runs, "gas", "dual1000", 0)          # 没有这个运行：不算失败
    txt = _section(A.section_replay_summary, runs, cache)
    assert "通过 1 个，失败 1 个" in txt and len(probs) == 1


def test_full_analysis_report_flags_truncated_and_notes(tmp_path, monkeypatch):
    out_dir = tmp_path / "npz"
    out_dir.mkdir()
    for s in (0, 1):
        for m in ("sw200", "dual200", "dual200_rb"):
            d, _ = make_run(m, dataset="gas", seed=s, max_rows=900)
            np.savez(out_dir / f"r2_gas_{m}_n1_S50_seed{s}.npz", **d)
    monkeypatch.setattr(sys, "argv", ["analyze_round3.py", "--dir", str(out_dir), "--round2_dir", str(tmp_path / "none"),
                                      "--n_estimators", "1", "--seeds", "0-1", "--out", str(tmp_path / "r.md"),
                                      "--csv", str(tmp_path / "r.csv")])
    monkeypatch.setattr(A, "section_stage", lambda *a, **k: None)
    A.main()
    txt = (tmp_path / "r.md").read_text()
    assert "截断运行" in txt and "不是正式结果" in txt and "重放校验合计" in txt and "描述性" in txt
    assert "tag ''" not in txt.splitlines()[0] and "同写法" not in txt          # 调试痕迹与不准确的措辞已去掉



# ---------------------------------------------------------------- 最终检查后补的测试

def test_fgt_is_mean_not_median_over_three_regimes():
    from src.eval import round3_analysis as mod
    import unittest.mock as um
    d, _ = make_run("dual200_rb", boundaries=(500, 1000), n=1500)
    with um.patch.object(mod, "fgt_per_regime", lambda _d: {0: 0.6, 1: 0.0, 2: 0.0}):
        assert mod.run_metrics(d)["FGT"] == pytest.approx(0.2)        # 中位数会是 0


def test_stage_auc_compares_with_all_later_regimes():
    rng = np.random.default_rng(1)
    regime = np.repeat([0, 1, 2], 300)
    X = rng.normal(size=(900, 3))
    X[regime == 2] += 4.0                  # 阶段 0 与阶段 1 相同，只和阶段 2 不同
    auc = stage_auc(X, regime)
    assert auc[0] > 0.65                   # 只比下一个阶段时约为 0.5


def test_gradual_section_reports_all_three_arms():
    seeds = [0, 1]
    S = {}
    for s in seeds:
        for m in ("sw200", "dual200", "dual200_rb", "dual200_rb_adwin"):
            S[("insects_gradual", m, s)] = dict(RET=0.5 + 0.01 * s, ADAPT=0.7, ADAPT_masked=0.7, ZONE=0.6, ZONE_T1=0.6,
                                               ZONE_T2=0.6, ZONE_T3=0.6, ZONE_500=0.6)
        S[("insects_gradual_start", "dual200_rb", s)] = dict(S[("insects_gradual", "dual200_rb", s)])
    runs = {("insects_gradual", "sw200", 0): None}          # 只用来表示"有渐变段结果"；ADWIN 表里记为缺失
    txt = _section(A.section_gradual, runs, S, seeds, {}, [])
    for arm in ("ours (a) 官方变点 − KDD", "ours (b) 过渡区起点 − KDD", "ours (c) ADWIN − KDD"):
        assert arm in txt


# ---------------------------------------------------------------- 后并入脚本的数字（按批基线、掩码、阶段结束后的记忆）

def _step_run(n=800, boundaries=()):
    """y = 前 400 行为 0、后 400 行为 1；从第 200 行起每 50 行一批，全部预测（预测值不重要）。"""
    y = np.r_[np.zeros(n // 2, int), np.ones(n - n // 2, int)]
    pred = np.full(n, -1)
    pred[200:] = 0
    return dict(y=y, pred=pred, start=np.array([200]), stride=np.array([50]), boundaries=np.array(boundaries, int))


def test_run_batches_cover_predicted_rows_and_respect_boundaries():
    d, _ = make_run("dual200_rb", boundaries=(500, 1000), mem_boundaries=(830,))
    bs = run_batches(d)
    covered = np.concatenate([np.arange(a, b) for a, b in bs])
    assert np.array_equal(covered, np.flatnonzero(d["pred"] >= 0))
    ends = {b for _, b in bs}
    assert {500, 830, 1000} <= ends and all(b - a <= 50 for a, b in bs)       # 批不跨越评估边界与记忆边界
    assert not any(a < 500 < b or a < 830 < b or a < 1000 < b for a, b in bs)
    t, _ = make_run("dual200", max_rows=900)                                   # 截断运行：只到有效行数
    assert np.array_equal(np.concatenate([np.arange(a, b) for a, b in run_batches(t)]), np.flatnonzero(t["pred"] >= 0))


def test_baselines_full_batch_vs_row_hand_values():
    d = _step_run()
    b = baselines_full(d)
    # 按批：第 [400, 450) 批用 y[399] = 0，整批都错 → 类 1 召回 350/400
    assert b["nc_batch"] == pytest.approx((1.0 + 350 / 400) / 2)
    # 逐行（预注册口径，含批内泄漏）：只有第 400 行错
    assert b["nc_row"] == pytest.approx((1.0 + 399 / 400) / 2) and b["nc_row"] == pytest.approx(baseline_acc(d)[0])
    # 多数类：按批取批起点前 200 行；批起点 400、450、500 的窗口里 1 不占多数（平票取 0）→ 第 400–549 行错
    assert b["maj_batch"] == pytest.approx((1.0 + 250 / 400) / 2)
    assert b["maj_row"] == pytest.approx((1.0 + 299 / 400) / 2) and b["maj_row"] == pytest.approx(baseline_acc(d)[1])
    assert b["nc_row_plain"] == pytest.approx(599 / 600) and b["nc_batch_plain"] == pytest.approx(550 / 600)
    assert b["n_pred"] == 600 and b["n_batches"] == 12
    # 标签逐行交替时按批的 No-Change 与逐行不同：批内所有行用同一个标签
    y = np.tile([0, 1], 400)
    pred = np.full(800, -1)
    pred[200:] = 0
    alt = baselines_full(dict(y=y, pred=pred, start=np.array([200]), stride=np.array([50]), boundaries=np.array([], int)))
    assert alt["nc_row"] == pytest.approx(0.0) and alt["nc_batch"] == pytest.approx(0.5)


def test_mask_removed_share_denominators():
    y = np.r_[np.zeros(40, int), np.tile([1, 0], 30)]
    pred = np.full(100, -1)
    pred[20:] = 0
    d = dict(y=y, pred=pred)
    assert mask_removed_share(d) == pytest.approx(20 / 80)                   # 掩码行 0–39，已预测 20–99
    assert mask_removed_share(d, "stream") == pytest.approx(40 / 100)
    with pytest.raises(AssertionError):
        mask_removed_share(d, "bogus")


def test_stream_acc_per_regime_and_class_share_and_first_diff():
    y = np.array([0, 1, 0, 1, 0, 0, 0, 1])
    regime = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    pred = np.array([-1, 1, 0, 0, 0, 0, 1, 1])
    acc = stream_acc_per_regime(dict(y=y, pred=pred, regime=regime))
    assert acc[0] == pytest.approx((1.0 + 0.5) / 2)          # 阶段 0 已预测行 1..3：类 0 召回 1/1，类 1 召回 ... 见下
    assert acc[1] == pytest.approx((2 / 3 + 1.0) / 2)
    assert np.allclose(class_share(dict(y=y)), [5 / 8, 3 / 8])
    a, b = dict(pred=np.array([-1, 0, 1, 1, 0])), dict(pred=np.array([-1, 0, 1, 0, 1]))
    assert first_diff_row(a, b) == 3 and first_diff_row(a, a) is None
    assert first_diff_row(dict(pred=np.array([0, 0, -1])), dict(pred=np.array([0, 0, 1]))) is None      # 只比两边都已预测的行


def test_holdout_class_recall_hand_values():
    d = dict(y=np.zeros(300, int), y_hold=np.array([0, 0, 1, 1, 2]), regime_hold=np.array([0, 0, 0, 0, 1]),
             boundaries=np.array([100]), ckpt_t=np.array([50, 100, 150, 200]),
             ckpt_pred=np.array([[1, 1, 0, 0, -1],        # t=50 ≤ 阶段结束：不计
                                 [1, 1, 0, 0, -1],        # t=100 = 阶段结束那一刻：也不计（只取结束之后）
                                 [0, 1, 1, 1, 1],         # 类 0 召回 0.5，类 1 召回 1.0
                                 [0, 0, 1, 1, 1]], dtype=np.int8))        # 类 0 召回 1.0，类 1 召回 1.0
    r = holdout_class_recall(d, 0)
    assert r[0] == pytest.approx(0.75) and r[1] == pytest.approx(1.0) and np.isnan(r[2]) and len(r) == 3


def _fake_rep():
    return dict(ts=np.array([100, 150, 200, 250]),
                comp=np.array([[4, 4, 3, 1], [0, 6, 7, 9]]), mem_comp=np.array([[8, 6, 5, 2], [0, 5, 5, 8]]),
                long_len=np.array([10, 10, 10, 10]), full=np.array([False, True, True, True]),
                age_med=np.array([5.0, 10.0, 20.0, 40.0]))


def test_regime_leftover_hand_values():
    d = dict(boundaries=np.array([150]))
    out = regime_leftover(_fake_rep(), d)
    assert set(out) == {0}
    r = out[0]
    assert r["lt5"] == pytest.approx(0.5)                   # 结束之后（t = 200, 250）整个记忆里的 R0 行：5, 2 → 只有一次 < 5
    assert r["avg_long"] == pytest.approx(2.0) and r["min_long"] == 1
    assert r["share_end"] == pytest.approx(4 / 10)          # t = 150 那次更新时长期池里 R0 占 4/10
    # 阶段还没结束（截断）→ 不出现
    assert regime_leftover(_fake_rep(), dict(boundaries=np.array([999]))) == {}


def test_age_last_present_and_coverage_hand_values():
    rep = _fake_rep()
    assert age_summary(rep) == pytest.approx((10 + 20 + 40) / 3)            # 只含记忆已满的更新（第一次更新的 5 不计）
    assert np.isnan(age_summary(dict(full=np.array([False]), age_med=np.array([np.nan]))))
    assert last_present_time(rep, 0) == 250 and last_present_time(dict(ts=rep["ts"], comp=np.zeros((2, 4))), 1) is None
    assert last_present_time(dict(ts=rep["ts"], comp=np.array([[3, 2, 0, 0]])), 0) == 150
    n = 200
    y = np.arange(n) % 3
    regime = np.r_[np.zeros(150, int), np.ones(50, int)]
    d = dict(y=y, regime=regime, boundaries=np.array([150]), y_hold=np.array([0, 1, 2, 0]), regime_hold=np.zeros(4, int))
    c = class_coverage(dict(final_long=np.array([0, 1, 2, 3, 160])), d)[0]
    assert c["n_rows"] == 4 and c["per_class"].tolist() == [2, 1, 1] and c["covered"] == 3 and c["n_classes"] == 3
    c = class_coverage(dict(final_long=np.array([0, 1, 160])), d)[0]       # 类 2 没有行 → 覆盖 2/3
    assert c["per_class"].tolist() == [1, 1, 0] and c["covered"] == 2 and c["n_classes"] == 3


def test_final_pool_by_position_hand_values():
    d = dict(zone=np.array([[100, 130]]), y=np.arange(200) % 3)
    f = final_pool_by_position(dict(final_long=np.array([50, 100, 105, 106, 115, 125, 126, 127, 140])), d)   # 100 = 过渡区第一行
    assert (f["pure_first"], f["t1"], f["t2"], f["t3"], f["pure_last"], f["total"]) == (1, 3, 1, 3, 1, 9)
    assert f["classes_first"] == 1
    assert final_pool_by_position(dict(final_long=np.array([1])), dict(zone=np.zeros((0, 2), int), y=d["y"])) == {}


def test_replay_extra_fields_match_independent_recomputation():
    """mem_comp / age_med / final_long / regime_leftover 与直接 hook 记忆类算出的一致。"""
    d, _ = make_run("dual200_rb", boundaries=(500, 1000), mem_boundaries=(500, 1000))
    rep = replay_run(d, "dual200_rb")
    y, regime = d["y"], d["regime"]
    mem = make_memory("dual200_rb", np.zeros((len(y), 1), np.float32), y, 3, 0)
    rec, orig = [], mem.update

    def upd(idx):
        orig(idx)
        t = int(np.max(idx)) + 1
        L = np.asarray(mem._long, dtype=np.int64)
        rec.append((t, np.bincount(regime[mem._all()], minlength=3), float(np.median(t - L)) if len(L) else np.nan, L.copy()))
    mem.update = upd
    run_stream(mem, lambda Xc, yc, Xq: np.full((len(Xq), 3), 1 / 3), np.zeros((len(y), 1), np.float32), y, regime,
               np.zeros((60, 1), np.float32), d["regime_hold"], [500, 1000], stride=50, start=200, ckpt_every=500,
               segmentation="official", mem_boundaries=[500, 1000])
    assert np.array_equal(rep["mem_comp"], np.stack([r[1] for r in rec], axis=1))
    assert np.allclose(rep["age_med"], [r[2] for r in rec], equal_nan=True)
    assert np.array_equal(rep["final_long"], rec[-1][3])
    assert rep["mem_comp"][:, -1].sum() == len(mem._all())                     # 整个记忆（长期加短期）的行数
    out = regime_leftover(rep, d)
    for j, e in enumerate((500, 1000)):
        after = [r for r in rec if r[0] > e]
        assert out[j]["lt5"] == pytest.approx(np.mean([r[1][j] < 5 for r in after]))
    assert out[0]["lt5"] == 0.0 and out[1]["lt5"] == 0.0     # 阶段均衡的长期池总保留每个旧阶段至少 5 行


def test_new_sections_text_and_values():
    seeds = [0, 1]
    d = _step_run()
    runs = {("gas", "sw200", s): d for s in seeds}
    S = {("gas", "sw200", s): dict(ADAPT=0.9) for s in seeds}
    txt = _section(A.section_baselines, runs, S, seeds)
    assert "No-Change_batch" in txt and "偏乐观" in txt and "93.75 ± 0.00" in txt and "81.25 ± 0.00" in txt and "99.88" in txt
    assert "不能写" in txt and "批内尚未揭示" in txt
    y = np.r_[np.zeros(40, int), np.tile([1, 0], 30)]
    pred = np.full(100, -1)
    pred[20:] = 0
    mruns = {("gas", "sw200", 0): dict(y=y, pred=pred)}
    mtxt = _section(A.section_mask, mruns, [0])
    assert "| gas | 1 | 25.00 |" in mtxt and "40.00" in mtxt
    # 配对表有 KDD − 滑窗，且数值是 dual − sw
    seeds6 = list(range(6))
    runs6, S6 = _fake_S_runs(seeds6)
    ptxt = _section(A.section_paired, runs6, S6, seeds6)
    line = next(ln for ln in ptxt.splitlines() if ln.startswith("| KDD − 滑窗 |") and ln.count("p=") == 3)
    cells = [c.strip() for c in line.strip("|").split("|")][1:]
    arr = lambda k, m: np.array([S6[("gas", m, s)][k] for s in seeds6])      # noqa: E731
    assert cells[0].startswith(f"{paired(arr('ADAPT', 'dual200'), arr('ADAPT', 'sw200'))['mean'] * 100:+.2f}")
    assert cells[2].startswith(f"{paired(arr('RET', 'dual200'), arr('RET', 'sw200'))['mean'] * 100:+.2f}")
    assert "不合并取较大的 p" in ptxt and "同写法" not in ptxt


def test_leftover_and_stream_regime_sections_on_tiny_runs():
    seeds = [0, 1]
    runs = {}
    for s in seeds:
        for m in ("dual200", "dual200_rb"):
            runs[("gas", m, s)], _ = make_run(m, dataset="gas", seed=s)
    cache, probs = {}, []
    txt = _section(A.section_leftover, runs, seeds, cache, probs)
    assert probs == [] and "i. 阶段结束之后记忆里还剩什么" in txt
    row = next(ln for ln in txt.splitlines() if ln.startswith("| gas | 200 |") and "逐阶段" in ln)
    assert row.split("|")[4].strip() == "0.0（逐阶段：0.0, 0.0）"                 # ours 总保留每个旧阶段至少 5 行
    cov = next(ln for ln in txt.splitlines() if ln.startswith("| gas | 200 | ours |"))
    n_old = len(runs[("gas", "dual200_rb", 0)]["boundaries"]) * len(seeds)
    assert f"/{n_old} |" in cov                                               # 类全被覆盖的 (阶段, seed) 数/总数
    share_row = next(ln for ln in txt.splitlines() if ln.startswith("| gas | 200 |") and ln.count("|") == 5
                     and "逐阶段" not in ln and "." in ln.split("|")[4] and float(ln.split("|")[3]) > 0)
    reps = [get_rep for get_rep in (A.get_replay(cache, probs, runs, "gas", "dual200_rb", s) for s in seeds)]
    exp = np.nanmean([regime_leftover(r, runs[("gas", "dual200_rb", s)])[j]["share_end"] for r, s in zip(reps, seeds)
                      for j in range(2)])
    assert share_row.split("|")[4].strip() == f"{100 * exp:.1f}"             # 刚结束阶段的行在长期池里的占比
    stxt = _section(A.section_stream_regime, runs, seeds)
    assert "| +0.00 (0/2) |" in stxt                                          # R0 在第一次关闭阶段之前两者相同
