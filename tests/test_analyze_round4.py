"""第四轮分析（src/eval/round4_analysis.py 与 scripts/analyze_round4.py）：方法名解析（旧名不变）、跨轮载入与 0.75 列来源、
跨轮一致性（放行 commit 不同、抓数据不一致）、配对差数值、--quick 用重放 digest 抓"文件名与实际容量不符"、
"缺失"与"校验失败"分开写、报告单元格里的数值与 ours/KDD 方向（用已知数值的合成运行）、calib4 逐位核对、整条流程不因缺 0.75 列崩溃。
全部用合成数据与哑预测器，不调用 TabPFN。"""
import os
import sys

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from test_analyze_round3 import make_run  # noqa: E402

import analyze_round4 as A4  # noqa: E402
import src.eval.round3_analysis as R3  # noqa: E402
import src.eval.round4_analysis as R4  # noqa: E402
import src.memory.context_memory as CM  # noqa: E402


def r4_run(method, ds="gas", seed=0, commit="c4", **kw):
    d, _ = make_run(method, dataset=ds, seed=seed, commit=commit, **kw)
    return d


def save_runs(dir_, items, tag=""):
    """items: [(数据集, 方法, seed, dict)]；文件名按第二/三轮的规则（insects 无前缀）。"""
    os.makedirs(dir_, exist_ok=True)
    for ds, m, s, d in items:
        pre = "" if ds == "insects" else f"{ds}_"
        np.savez(os.path.join(dir_, f"r2_{pre}{m}_n1_S50_seed{s}{tag}.npz"), **d)


# ---------------------------------------------------------------- 方法名解析

def test_parse_method4_new_and_old_names():
    assert R4.parse_method4("dual200_s65") == ("kdd", 0.65)
    assert R4.parse_method4("dual200_s85") == ("kdd", 0.85)
    assert R4.parse_method4("dual200_rb_s65") == ("ours", 0.65)
    assert R4.parse_method4("dual200_rb_s85") == ("ours", 0.85)
    assert R4.parse_method4("dual200") == ("kdd", 0.75)            # 第三轮的名字就是 0.75 列
    assert R4.parse_method4("dual200_rb") == ("ours", 0.75)
    assert R4.parse_method4("sw200") == ("sw", None)
    for bad in ("dual200_s75", "dual1000_s65", "dual200_s70", "sw1000", "dual200_rb_adwin", "dual200_s65_rb"):
        with pytest.raises(ValueError):
            R4.parse_method4(bad)


def test_method_name_roundtrip_and_round3_parser_unchanged():
    for role in ("kdd", "ours"):
        for share in R4.SHARES:
            assert R4.parse_method4(R4.method_name(role, share)) == (role, share)
    assert R4.method_name("sw", None) == "sw200"
    # 第三轮的解析器行为不变：旧名照旧，不认识 _s65
    assert R3.parse_method("dual200_rb") == ("ours", 200, False)
    assert R3.parse_method("dual1000_rb_adwin") == ("ours", 1000, True)
    with pytest.raises(ValueError):
        R3.parse_method("dual200_s65")


def test_methods_registered_in_memory_and_expected_caps():
    assert set(R4.R4_METHODS) <= set(CM.METHODS)
    for m in R4.R4_METHODS:
        _, share = R4.parse_method4(m)
        mem = CM.make_memory(m, np.zeros((10, 1), np.float32), np.zeros(10, int), 3, 0)
        assert (mem.short_cap, mem.long_cap) == R4.EXPECTED_CAPS[share]
    assert R4.EXPECTED_CAPS == {0.65: (130, 70), 0.75: (150, 50), 0.85: (170, 30)}


# ---------------------------------------------------------------- 载入与 0.75 列来源

def test_load_all4_sources(tmp_path):
    d4, d3, d2 = str(tmp_path / "r4"), str(tmp_path / "r3"), str(tmp_path / "r2")
    save_runs(d4, [("gas", "dual200_s65", 0, r4_run("dual200_s65")),
                   ("gas", "dual200_rb_s85", 0, r4_run("dual200_rb_s85")),
                   ("gas", "dual200", 0, r4_run("dual200"))], tag="_t")             # 误放在第四轮目录里的 0.75 文件
    save_runs(d3, [("gas", "dual200", 0, r4_run("dual200", commit="c3")), ("gas", "dual200_rb", 0, r4_run("dual200_rb", commit="c3")),
                   ("gas", "sw200", 0, r4_run("sw200", commit="c3")), ("gas", "dual1000", 0, r4_run("dual1000", commit="c3")),
                   ("insects", "sw200", 0, r4_run("sw200", ds="insects", commit="c3"))])    # insects 的 sw200 不该取自第三轮
    save_runs(d2, [("insects", "sw200", 0, r4_run("sw200", ds="insects", commit="c2")),
                   ("insects", "sw1000", 0, r4_run("sw1000", ds="insects", commit="c2"))])
    runs, origin, probs = R4.load_all4(d4, d3, d2, n_estimators=1, tag="_t", datasets=("gas", "insects"))
    assert origin == {("gas", "dual200_s65", 0): "r4", ("gas", "dual200_rb_s85", 0): "r4", ("gas", "dual200", 0): "r3",
                      ("gas", "dual200_rb", 0): "r3", ("gas", "sw200", 0): "r3", ("insects", "sw200", 0): "r2"}
    assert str(runs[("gas", "dual200", 0)]["commit"][0]) == "c3"           # 0.75 列取自第三轮目录，不是第四轮目录里的同名文件
    assert str(runs[("insects", "sw200", 0)]["commit"][0]) == "c2"
    assert probs == []
    # 第四轮目录里没有任何第四轮文件时提示
    _, _, p = R4.load_all4(str(tmp_path / "nope"), d3, d2, n_estimators=1, datasets=("gas",))
    assert any("第四轮目录不存在" in x for x in p)


def test_load_all4_hints_wrong_directory_level_and_missing_prior_rounds(tmp_path):
    sub = tmp_path / "results_x" / "code" / "results" / "round4"        # 解压后 npz 在子目录里
    save_runs(str(sub), [("gas", "dual200_s65", 0, r4_run("dual200_s65"))], tag="_t")
    save_runs(str(sub), [("gas", "dual200_rb_s85", 0, r4_run("dual200_rb_s85"))], tag="_t")
    d3 = str(tmp_path / "r3")
    save_runs(d3, [("gas", "dual200", 0, r4_run("dual200", commit="c3"))])
    _, _, p = R4.load_all4(str(tmp_path), d3, str(tmp_path / "r2"), n_estimators=1, tag="_t", datasets=("gas",))
    assert len(p) == 1 and "不存在" not in p[0] and str(sub) in p[0] and "--dir" in p[0] and "共 2 个" in p[0]
    empty = tmp_path / "empty"
    empty.mkdir()
    _, _, p = R4.load_all4(str(empty), d3, str(tmp_path / "r2"), n_estimators=1, tag="_t", datasets=("gas",))
    assert len(p) == 1 and "存在，但" in p[0] and "子目录" in p[0]
    # 有第四轮文件、但第三轮 / 第二轮目录里什么都没有：单独提示"没有做跨轮核对"，不管别的问题有没有
    d4 = str(tmp_path / "r4")
    save_runs(d4, [("gas", "dual200_s65", 0, r4_run("dual200_s65")), ("insects", "dual200_s65", 0, r4_run("dual200_s65", ds="insects"))],
              tag="_t")
    _, _, p = R4.load_all4(d4, str(tmp_path / "none3"), str(tmp_path / "none2"), n_estimators=1, tag="_t", datasets=("gas", "insects"))
    assert any("第三轮目录" in x and "没有做跨轮核对" in x for x in p) and any("第二轮目录" in x and "insects" in x for x in p)
    _, _, p = R4.load_all4(d4, d3, str(tmp_path / "none2"), n_estimators=1, tag="_t", datasets=("gas",))
    assert p == []                                                    # 第三轮有文件；insects 不在分析范围里，不要求第二轮


def test_level_points_use_075_from_round3_column():
    S = {}
    for share, a_kdd, a_ours in ((0.65, 0.70, 0.71), (0.75, 0.80, 0.82), (0.85, 0.90, 0.93)):
        for s in range(3):
            S[("gas", R4.method_name("kdd", share), s)] = dict(ADAPT=a_kdd + 0.001 * s)
            S[("gas", R4.method_name("ours", share), s)] = dict(ADAPT=a_ours + 0.001 * s)
    pts = {(p["role"], p["share"]): p for p in R4.level_points(S, [0, 1, 2], ["gas"], "ADAPT")}
    assert pts[("kdd", 0.75)]["mean"] == pytest.approx(100 * 0.801)
    assert pts[("ours", 0.85)]["mean"] == pytest.approx(100 * 0.931)
    assert pts[("ours", 0.65)]["n"] == 3


# ---------------------------------------------------------------- 跨轮一致性

def _col(ds="gas", commit75="c3", seed=0):
    runs = {(ds, m, seed): r4_run(m, ds=ds, seed=seed, commit=commit75 if m in ("dual200", "dual200_rb", "sw200") else "c4")
            for m in ("sw200", "dual200", "dual200_rb", "dual200_s65", "dual200_rb_s85")}
    origin = {k: ("r3" if k[1] in ("sw200", "dual200", "dual200_rb") else "r4") for k in runs}
    return runs, origin


def test_consistency4_allows_commit_difference_across_rounds():
    runs, origin = _col()
    assert R3.check_consistency(runs)                                     # 第三轮自己的检查对跨轮 commit 不同会报
    assert R4.check_consistency4(runs, origin) == []


def test_consistency4_catches_data_mismatch_and_mem_boundaries():
    runs, origin = _col()
    bad = dict(runs[("gas", "dual200", 0)])
    bad["y"] = bad["y"].copy()
    bad["y"][5] = (bad["y"][5] + 1) % 3
    runs[("gas", "dual200", 0)] = bad
    assert any("y 不同" in p for p in R4.check_consistency4(runs, origin))
    runs, origin = _col()
    bad = dict(runs[("gas", "dual200_rb", 0)], mem_boundaries=np.array([500]))
    runs[("gas", "dual200_rb", 0)] = bad
    assert any("mem_boundaries 不同" in p for p in R4.check_consistency4(runs, origin))
    runs, origin = _col()
    runs[("gas", "dual200", 0)] = dict(runs[("gas", "dual200", 0)], boundaries=np.array([400, 1000]))
    assert any("boundaries 不同" in p for p in R4.check_consistency4(runs, origin))
    runs, origin = _col()
    runs[("gas", "dual200", 0)] = dict(runs[("gas", "dual200", 0)], hold_idx=np.arange(60) * 21)
    assert any("hold_idx 不同" in p for p in R4.check_consistency4(runs, origin))


def test_consistency4_commit_must_be_unique_within_source():
    runs, origin = _col()
    runs[("gas", "dual200_s65", 0)] = dict(runs[("gas", "dual200_s65", 0)], commit=np.array(["zzz"]))
    probs = R4.check_consistency4(runs, origin)
    assert any("来源 r4" in p for p in probs) and any("第四轮文件的 commit 不唯一" in p for p in probs)
    runs, origin = _col()
    runs[("gas", "sw200", 0)] = dict(runs[("gas", "sw200", 0)], commit=np.array(["other"]))
    assert any("来源 r3" in p for p in R4.check_consistency4(runs, origin))


# ---------------------------------------------------------------- 配对差

def test_diff_points_numbers():
    rng = np.random.default_rng(0)
    seeds = list(range(10))
    S = {}
    truth = {}
    for share in R4.SHARES:
        a, b = rng.normal(0.8, 0.05, 10), rng.normal(0.7, 0.05, 10)
        truth[share] = (a, b)
        for s in seeds:
            S[("gas", R4.method_name("ours", share), s)] = dict(RET=a[s])
            S[("gas", R4.method_name("kdd", share), s)] = dict(RET=b[s])
    del S[("gas", "dual200_rb_s85", 3)]                                     # 缺一个 seed：配对时丢掉
    pts = {p["share"]: p for p in R4.diff_points(S, seeds, ["gas"], "RET")}
    a, b = truth[0.75]
    d = a - b
    h = stats.t.ppf(0.975, 9) * d.std(ddof=1) / np.sqrt(10)
    assert pts[0.75]["mean"] == pytest.approx(100 * d.mean())
    assert (pts[0.75]["lo"], pts[0.75]["hi"]) == pytest.approx((100 * (d.mean() - h), 100 * (d.mean() + h)))
    assert pts[0.75]["npos"] == int((d > 0).sum()) and pts[0.75]["n"] == 10
    a, b = truth[0.85]
    keep = np.arange(10) != 3
    assert pts[0.85]["n"] == 9 and pts[0.85]["mean"] == pytest.approx(100 * (a - b)[keep].mean())
    assert len(pts[0.85]["per_seed"]) == 9
    miss = {p["share"]: p for p in R4.diff_points({}, seeds, ["gas"], "RET")}
    assert miss[0.65]["n"] == 0 and np.isnan(miss[0.65]["mean"])


# ---------------------------------------------------------------- 容量与 --quick

class _Args:
    dir, round3_dir, tag, n_estimators, quick_seed = "x", "y", "_t", 1, 0


@pytest.fixture
def healthy(monkeypatch):
    """合成数据的标签是随机的，ADAPT 低于 1/K+10pp 会被快检挂起；这里把指标固定成正常值，只测结构检查。"""
    fake = lambda _d: dict(ADAPT=0.9, ADAPT_masked=0.9, RET=0.5, RET_R0=0.5, RET_R1=0.5)      # noqa: E731
    monkeypatch.setattr(R3, "run_metrics", fake)
    monkeypatch.setattr(A4, "run_metrics", fake)


def _full_gas(commit="c4"):
    runs = {("gas", m, 0): r4_run(m, commit=commit, device="cuda") for m in R4.R4_METHODS}
    return runs, {k: "r4" for k in runs}


def test_capacity_problems_ok_budget_and_wrong_split(monkeypatch):
    for m in R4.R4_METHODS + ("sw200", "dual200", "dual200_rb"):
        assert R4.capacity_problems(r4_run(m)) == []
    d = dict(r4_run("dual200_s65"), budget=np.array([1000]))
    assert any("budget 1000" in p for p in R4.capacity_problems(d))
    monkeypatch.setitem(CM.METHODS, "dual200_s65", ("dual", 200, 0.75))      # 代码里的占比被改错：150/50 ≠ 130/70
    assert any("150/50" in p and "130/70" in p for p in R4.capacity_problems(r4_run("dual200_s65")))
    monkeypatch.setitem(CM.METHODS, "dual200_rb_s85", ("dual_rb", 400, 0.85))
    assert any("预算" in p or "总容量" in p or "budget" in p for p in R4.capacity_problems(r4_run("dual200_rb_s85")))


def _r3_cols(runs, origin, ds="gas"):
    for m in ("dual200", "dual200_rb", "sw200"):
        runs[(ds, m, 0)] = r4_run(m, ds=ds, commit="c3", device="cuda")
        origin[(ds, m, 0)] = "r3"


def test_quick_ok_missing_and_capacity(capsys, monkeypatch, healthy):
    runs, origin = _full_gas()
    _r3_cols(runs, origin)
    A4.quick(_Args, runs, origin, [])
    out = capsys.readouterr().out
    assert "[gas]" in out and "容量（本地代码里）：与方法名一致" in out and "判断：可以继续" in out
    assert "重放核对（digests / closures / 检查点时刻与 npz 比对）：通过" in out
    assert "[electricity]" in out and "暂无结果文件" in out                # 没有任何文件的数据集不算问题
    assert "没有发现问题。" in out and "跨轮核对" not in out              # 第三轮文件都在：不该有"没做跨轮核对"
    # 缺一个第四轮方法
    del runs[("gas", "dual200_rb_s85", 0)]
    A4.quick(_Args, runs, origin, [])
    out = capsys.readouterr().out
    assert "(dual200_rb_s85) 缺失" in out and "判断：需要看" in out and "缺 dual200_rb_s85" in out
    # 本地代码里的容量不符
    runs, origin = _full_gas()
    monkeypatch.setitem(CM.METHODS, "dual200_s85", ("dual", 200, 0.75))
    A4.quick(_Args, runs, origin, [])
    out = capsys.readouterr().out
    assert "容量（本地代码里）：不符" in out and "150/50" in out and "170/30" in out and "判断：需要看" in out


def test_quick_missing_prior_rounds_is_said_in_summary(capsys, healthy):
    runs, origin = _full_gas()                                           # 没有任何第三轮 / 第二轮文件
    A4.quick(_Args, runs, origin, ["!! 第三轮目录 y 里没有载入任何 dual200 / dual200_rb / sw200 文件"])
    out = capsys.readouterr().out
    assert "!! 第三轮目录 y" in out and "注意：!!" not in out                # 我们自己的提示不再叠"注意："
    assert "判断：可以继续" in out and "总结：" in out
    summary = out.split("总结：")[1]
    assert "没有发现问题。" not in summary and "没有做跨轮核对" in summary and "gas/dual200" in summary


def test_quick_replay_catches_file_whose_real_capacity_differs_from_its_name(capsys, healthy):
    """文件名写 _s65、实际按 0.75（150/50）跑：本地代码级的容量检查看不出来，重放 digest 看得出来。"""
    runs, origin = _full_gas()
    fake = r4_run("dual200", device="cuda")                              # 真按 150/50 跑出来的 digests
    fake["method"] = np.array(["dual200_s65"])
    assert R4.capacity_problems(fake) == []                              # 代码级检查：通过（它不看 npz 实际用了什么）
    ok, why = R4.replay_check(fake)
    assert not ok and "digests" in why
    runs[("gas", "dual200_s65", 0)] = fake
    A4.quick(_Args, runs, origin, [])
    out = capsys.readouterr().out
    assert "重放核对：**失败**" in out and "npz 实际用的记忆可能与方法名 dual200_s65 不符" in out
    assert "判断：需要看" in out and "dual200_s65: npz 实际用的记忆" in out
    assert out.count("重放核对（digests / closures / 检查点时刻与 npz 比对）：通过") == 3     # 其余三个正常


def test_quick_check4_replay_flag_and_replay_skipped_for_sliding_window(healthy):
    q = R4.quick_check4(r4_run("dual200_rb_s85", device="cuda"))
    assert q["replay_ok"] is True and not q["reasons"]
    assert R4.quick_check4(r4_run("dual200_rb_s85"), replay=False)["replay_ok"] is None
    assert R4.quick_check4(r4_run("sw200"))["replay_ok"] is None         # 滑窗没有长期池，不重放


def test_quick_flags_incomplete_cpu_and_cross_round_data_mismatch(capsys, healthy):
    runs, origin = _full_gas()
    d = r4_run("dual200_s65", device="cuda")
    d["pred"] = d["pred"].copy()
    d["pred"][700] = -1
    runs[("gas", "dual200_s65", 0)] = d
    # 一个 0.75 列文件的 y 与第四轮不同
    bad = r4_run("dual200", commit="c3")
    bad["y"] = bad["y"].copy()
    bad["y"][9] = (bad["y"][9] + 1) % 3
    runs[("gas", "dual200", 0)] = bad
    origin[("gas", "dual200", 0)] = "r3"
    A4.quick(_Args, runs, origin, [])
    out = capsys.readouterr().out
    assert "行数不完整" in out and "一致性" in out and "y 不同" in out and "判断：需要看" in out


# ---------------------------------------------------------------- calib4 核对

def test_calib4_check_identical_differs_no_r3_and_absent(tmp_path):
    d4 = str(tmp_path / "r4")
    r3 = r4_run("dual200_rb", ds="electricity", commit="c3")
    assert R4.calib4_check(d4, {}, {}, 1, "_t") == []                           # 目录不存在 / 没有 calib4 文件
    same = dict(r3, commit=np.array(["c4"]))
    os.makedirs(d4)
    np.savez(os.path.join(d4, "r2_electricity_dual200_rb_n1_S50_seed0_calib4_t.npz"), **same)
    key = ("electricity", "dual200_rb", 0)
    out = R4.calib4_check(d4, {key: r3}, {key: "r3"}, 1, "_t")
    assert len(out) == 1 and out[0]["status"] == "identical" and out[0]["key"] == key
    assert R4.calib4_check(d4, {key: r3}, {key: "r3"}, 1, "_other") == []         # tag 不符
    assert R4.calib4_check(d4, {}, {}, 1, "_t")[0]["status"] == "no_r3"
    for field in ("pred", "digests", "ckpt_pred"):
        bad = dict(same)
        arr = np.array(bad[field]).copy()
        if field == "digests":
            arr[-1] = "x" + str(arr[-1])
        else:
            arr.flat[int(np.flatnonzero(arr.ravel() >= 0)[3])] += 1
        bad[field] = arr
        np.savez(os.path.join(d4, "r2_electricity_dual200_rb_n1_S50_seed0_calib4_t.npz"), **bad)
        out = R4.calib4_check(d4, {key: r3}, {key: "r3"}, 1, "_t")
        assert out[0]["status"] == "differs" and field in out[0]["detail"]


# ---------------------------------------------------------------- 整条流程

def _write_all(tmp_path, with_r3=True):
    d4, d3, d2 = str(tmp_path / "r4"), str(tmp_path / "r3"), str(tmp_path / "r2")
    for s in (0, 1):
        save_runs(d4, [("gas", m, s, r4_run(m, seed=s, commit="c4")) for m in R4.R4_METHODS], tag="_t")
        if with_r3:
            save_runs(d3, [("gas", m, s, r4_run(m, seed=s, commit="c3")) for m in ("dual200", "dual200_rb", "sw200")])
    return d4, d3, d2


def test_analyze_main_with_and_without_075_column(tmp_path, monkeypatch, capsys):
    for with_r3 in (True, False):
        d4, d3, d2 = _write_all(tmp_path / str(with_r3), with_r3)
        out, csv = str(tmp_path / f"a{with_r3}.md"), str(tmp_path / f"a{with_r3}.csv")
        monkeypatch.setattr(sys, "argv", ["analyze_round4.py", "--dir", d4, "--round3_dir", d3, "--round2_dir", d2, "--tag", "_t",
                                          "--n_estimators", "1", "--seeds", "0-1", "--replay_seeds", "0-1", "--out", out, "--csv", csv])
        A4.main()
        text = open(out, encoding="utf-8").read()
        assert "探索性" in text and "不做多重比较校正" in text and "不给 p 值" in text
        # 读法按预注册的说法：做这个分析是事后决定的，0.65 / 0.85 不是事后挑的；calib4 只证明一个运行；RET 是"平均"
        assert "事后选的" not in text and "600 / 800 / 1000" in text and "KDD 论文网格里的另外两个值" in text
        assert "逐位复现" in text and "一个**运行" in text and "已证明新代码与第三轮逐位一致" not in text
        assert "所有检查点上该阶段留出集准确率的平均" in text and "−0.51 pp" in text and "200 行 −2.10 pp" in text
        assert "那里差异小或为负，不能读成" in text and "\"—\"= 有文件，但没有可用的" in text
        assert "重放校验（digests" in text and "失败 0 个" in text
        assert "| 0.75（150/50） | KDD 双记忆 | dual200 |" in text
        if with_r3:
            assert "| 0.75（150/50） | KDD 双记忆 | dual200 | 第三轮 |" in text and "gas/dual200（应来自" not in text
            assert "没有做跨轮核对" not in text and "第三轮目录" not in text
        else:
            assert "第三轮目录" in text and "没有做跨轮核对" in text.split("### 一致性与校验问题")[1].split("## a.")[0]
            assert "gas/dual200（应来自第三轮）: 缺 seed [0, 1]" in text
            assert "| 0.75（150/50） | KDD 双记忆 | dual200 | — | 缺失 | 缺失 | 缺失 |" in text
        assert "### electricity" not in text.split("## a.")[1]                              # 没有数据的数据集不出表、不崩溃
        rows = open(csv).read().strip().splitlines()
        assert len(rows) == 1 + (14 if with_r3 else 8)
    capsys.readouterr()


def test_analyze_main_exits_when_no_round4_files(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["analyze_round4.py", "--dir", str(tmp_path / "none"), "--round3_dir", str(tmp_path),
                                      "--round2_dir", str(tmp_path), "--out", "", "--csv", ""])
    with pytest.raises(SystemExit):
        A4.main()


def test_plot_round4_runs_with_missing_columns(tmp_path, monkeypatch):
    import plot_round4 as P4
    d4, d3, d2 = _write_all(tmp_path, with_r3=False)
    out = str(tmp_path / "figs")
    monkeypatch.setattr(sys, "argv", ["plot_round4.py", "--dir", d4, "--round3_dir", d3, "--round2_dir", d2, "--tag", "_t",
                                      "--n_estimators", "1", "--seeds", "0-1", "--out", out])
    P4.main()
    assert {"fig_ratio.png", "fig_ratio_diff.png"} <= set(os.listdir(out))


# ---------------------------------------------------------------- 报告里的数值与方向（已知数值的合成运行）

def controlled(method, seed, wrong_regimes=(), wrong_every=0, commit="c4"):
    """真的重放得出的 digests（所以 d、e 段的重放校验能通过），但把预测改成已知的：
    ours 式全对；wrong_regimes 里的阶段的留出集预测全错（留出准确率 0）；wrong_every > 0 时流上每 wrong_every 行错一行。"""
    d = r4_run(method, seed=seed, commit=commit, device="cuda")
    y, K = d["y"], 3
    pred = d["pred"].copy()
    done = pred >= 0
    pred[done] = y[done]
    if wrong_every:
        bad = done & (np.arange(len(y)) % wrong_every == 0)
        pred[bad] = (y[bad] + 1) % K
    ok = d["ckpt_pred"] >= 0
    cp = np.where(ok, d["y_hold"][None, :], -1)
    for j in wrong_regimes:
        sel = d["regime_hold"] == j
        cp[:, sel] = np.where(ok[:, sel], (d["y_hold"][sel] + 1) % K, -1)
    return dict(d, pred=pred, ckpt_pred=cp)


# 每个占比 KDD 的"出错方式"：(seed 0, seed 1)；ours 永远全对
BAD = {0.65: [dict(wrong_regimes=(0,), wrong_every=7), dict()],              # seed 1 与 ours 无差别：差值 0，不是正号
       0.75: [dict(wrong_regimes=(1,), wrong_every=5)] * 2,
       0.85: [dict(wrong_regimes=(0, 1), wrong_every=3)] * 2}


def _write_controlled(tmp_path):
    d4, d3, d2 = str(tmp_path / "r4"), str(tmp_path / "r3"), str(tmp_path / "r2")
    for share in R4.SHARES:
        for s_ in (0, 1):
            kdd, ours = R4.method_name("kdd", share), R4.method_name("ours", share)
            dest, c = (d3, "c3") if share == 0.75 else (d4, "c4")
            tg = "" if share == 0.75 else "_t"
            save_runs(dest, [("gas", kdd, s_, controlled(kdd, s_, commit=c, **BAD[share][s_])),
                             ("gas", ours, s_, controlled(ours, s_, commit=c))], tag=tg)
    for s_ in (0, 1):
        save_runs(d3, [("gas", "sw200", s_, controlled("sw200", s_, wrong_every=2, commit="c3"))])
    return d4, d3, d2


def _run_main(monkeypatch, d4, d3, d2, out, extra=()):
    monkeypatch.setattr(sys, "argv", ["analyze_round4.py", "--dir", d4, "--round3_dir", d3, "--round2_dir", d2, "--tag", "_t",
                                      "--n_estimators", "1", "--seeds", "0-1", "--out", out, "--csv", "", *extra])
    A4.main()
    return open(out, encoding="utf-8").read()


def _rows(text, start, stop):
    """text 里 start 标题到 stop 标题之间所有表格行（按 | 切开去空白）。"""
    body = text.split(start)[1].split(stop)[0]
    return [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in body.splitlines() if ln.startswith("|") and "---" not in ln]


def test_report_numbers_and_signs_with_known_values(tmp_path, monkeypatch, capsys):
    from sklearn.metrics import balanced_accuracy_score

    d4, d3, d2 = _write_controlled(tmp_path)
    text = _run_main(monkeypatch, d4, d3, d2, str(tmp_path / "rep.md"), ["--replay_seeds", "1"])
    capsys.readouterr()
    assert "- 重放校验失败" not in text and "- !!" not in text

    def adapt_gap(share, s_):                                   # 独立算：ours ADAPT = 100，KDD 用 sklearn 算
        d = controlled("dual200", s_, **BAD[share][s_])
        done = d["pred"] >= 0
        return 100 * (1 - balanced_accuracy_score(d["y"][done], d["pred"][done]))

    gaps = {sh: [adapt_gap(sh, 0), adapt_gap(sh, 1)] for sh in R4.SHARES}
    # b.0：每个占比的均值（正号 seed 数/n）。ΔRET：.65 → (50+0)/2、2 个 seed 中只有 1 个为正；.75 → 50；.85 → 100
    row = {r[1]: r for r in _rows(text, "### b.0", "### gas") if r[0] == "gas"}
    assert row["ΔRET"][2:] == ["+25.00（1/2）", "+50.00（2/2）", "+100.00（2/2）"]
    want = [f"{np.mean(gaps[sh]):+.2f}（{sum(g > 1e-9 for g in gaps[sh])}/2）" for sh in R4.SHARES]
    assert row["ΔADAPT"][2:] == want and want[0].endswith("（1/2）") and want[0].startswith("+")
    # b：每个占比的配对差单元格，方向 = ours − KDD（ours 好 → 正号）
    tab = {r[0]: r for r in _rows(text, "### gas\n\n| 短期占比（短/长） | ΔADAPT |", "## c.")}
    assert tab["0.65（130/70）"][3].startswith("+25.00 [") and tab["0.65（130/70）"][3].endswith("(1/2)")
    assert tab["0.75（150/50）"][3].startswith("+50.00 [+50.00, +50.00] (2/2)")
    assert tab["0.85（170/30）"][1].startswith(f"{np.mean(gaps[0.85]):+.2f} [")
    # c：各阶段 ΔRET：.65 R0 (100+0)/2；.75 只有 R1 差；.85 两个阶段都差
    stage = {r[0]: r[1:] for r in _rows(text, "各阶段 RET 的 ours − KDD", "## d.")}
    assert stage["0.65"][0].startswith("+50.00 [") and stage["0.65"][1].startswith("+0.00 [+0.00, +0.00] (0/2)")
    assert stage["0.75"][0].startswith("+0.00 [+0.00, +0.00] (0/2)") and stage["0.75"][1].startswith("+100.00 [+100.00, +100.00] (2/2)")
    assert stage["0.85"][0].startswith("+100.00") and stage["0.85"][1].startswith("+100.00")
    # d：长期池，seed 取 --replay_seeds 的第一个（1）；逐行与独立重放（seed 1 的运行）比对，总行数 = 长期容量
    caps = {0.65: 70, 0.75: 50, 0.85: 30}
    rows = [r for r in _rows(text, "### gas（seed 1）", "## e.") if r[0].startswith(("边界", "终点"))]
    assert len(rows) == 3 * 6                                    # 2 个边界 + 终点，各 6 个 (占比, 方法)
    for r in rows:
        t, share, m = int(r[0].split()[1]), float(r[1]), r[2]
        d = controlled(m, 1, **(BAD[share][1] if m == R4.method_name("kdd", share) else {}))
        rep = R3.replay_run(d, m, verify=True)
        i = int(np.flatnonzero(rep["ts"] == t)[0])
        assert [int(x) for x in r[3:]] == [int(rep["comp"][j, i]) for j in range(rep["comp"].shape[0])] + [int(rep["long_len"][i])]
        assert int(r[-1]) == caps[share] == sum(int(x) for x in r[3:-1])
        if r[0].startswith("终点"):
            assert t == len(d["y"])
    final = {r[2]: [int(x) for x in r[3:-1]] for r in rows if r[0].startswith("终点")}
    for share in R4.SHARES:                                       # 阶段均衡：各阶段的行数最多差 1
        assert max(final[R4.method_name("ours", share)]) - min(final[R4.method_name("ours", share)]) <= 1


def test_levels_section_masked_column_is_not_the_plain_adapt_column():
    S, runs, origin = {}, {}, {}
    for m in ["sw200"] + [R4.method_name(r, sh) for r in ("kdd", "ours") for sh in R4.SHARES]:
        for s_ in (0, 1):
            S[("gas", m, s_)] = dict(ADAPT=0.80 + 0.01 * s_, ADAPT_masked=0.60 + 0.02 * s_, RET=0.40)
            runs[("gas", m, s_)] = {}
            origin[("gas", m, s_)] = "r3" if m in ("dual200", "dual200_rb", "sw200") else "r4"
    lines = []
    A4.section_levels(lambda *a: lines.append(" ".join(str(x) for x in a)), runs, origin, S, [0, 1])
    row = next(ln for ln in lines if ln.startswith("| 0.75（150/50） | KDD 双记忆 | dual200 |"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[3] == "第三轮" and cells[4].startswith("80.50 ± 0.71 [") and cells[5].startswith("61.00 ± 1.41 [")
    assert cells[6].startswith("40.00 ± 0.00 [")


def test_level_points_interval_and_sw_reference_numbers():
    from scipy import stats

    vals = [0.70, 0.74, 0.79]
    S = {("gas", "dual200_rb_s85", s_): dict(ADAPT=v) for s_, v in enumerate(vals)}
    S.update({("gas", "sw200", s_): dict(ADAPT=v) for s_, v in enumerate((0.5, 0.6, 0.7))})
    S.update({("electricity", "sw200", 0): dict(ADAPT=0.9)})
    p = next(q for q in R4.level_points(S, [0, 1, 2], ["gas"], "ADAPT") if (q["role"], q["share"]) == ("ours", 0.85))
    h = stats.t.ppf(0.975, 2) * np.std(vals, ddof=1) / np.sqrt(3)             # 95% t 区间，不是 ± SD
    assert p["mean"] == pytest.approx(100 * np.mean(vals)) and p["n"] == 3
    assert (p["lo"], p["hi"]) == pytest.approx((100 * (np.mean(vals) - h), 100 * (np.mean(vals) + h)))
    assert abs(p["hi"] - p["lo"]) / 2 != pytest.approx(100 * np.std(vals, ddof=1))
    ref = R4.sw_reference(S, [0, 1, 2], ["gas", "electricity", "insects"], "ADAPT")
    assert ref["gas"] == pytest.approx(60.0) and ref["electricity"] == pytest.approx(90.0) and np.isnan(ref["insects"])


def test_failed_replay_is_written_as_failed_not_missing(tmp_path, monkeypatch, capsys):
    """文件在、但重放与 npz 不一致 -> d、e 段写"校验失败"；文件不在 -> 写"缺失"；两个词不混用。"""
    d4, d3, d2 = _write_all(tmp_path, with_r3=False)                       # 没有 0.75 列文件 -> "缺失"
    f = os.path.join(d4, "r2_gas_dual200_s65_n1_S50_seed1_t.npz")
    z = dict(np.load(f, allow_pickle=True))
    z["digests"] = z["digests"].copy()
    z["digests"][-1] = "bad"
    np.savez(f, **z)
    text = _run_main(monkeypatch, d4, d3, d2, str(tmp_path / "rep.md"), ["--replay_seeds", "1"])
    capsys.readouterr()
    assert "重放校验失败 ('gas', 'dual200_s65', 1)" in text
    pool = {(r[1], r[2]): r[3:] for r in _rows(text, "### gas（seed 1）", "## e.") if r[0].startswith("终点")}
    assert set(pool[("0.65", "dual200_s65")]) == {"校验失败"}               # 有文件但失败
    assert set(pool[("0.75", "dual200")]) == {"缺失"}                      # 根本没有文件
    assert all(x.isdigit() for x in pool[("0.65", "dual200_rb_s65")])      # 同一占比的另一个方法照常
    ov = {r[0:2].__str__(): r[2:] for r in _rows(text, "## e.", "### 重放校验合计")}
    kdd = next(v for k, v in ov.items() if "KDD" in k)
    assert kdd[0] == "校验失败" and kdd[1] == "缺失" and kdd[2] not in ("缺失", "校验失败")
    assert "只核对了" in text and "不能证明" in text                       # 只重放了 seed 1：范围写明白


def test_report_says_scope_of_replay_when_all_seeds_replayed(tmp_path, monkeypatch, capsys):
    d4, d3, d2 = _write_controlled(tmp_path)
    text = _run_main(monkeypatch, d4, d3, d2, str(tmp_path / "rep.md"), ["--replay_seeds", "0-1"])
    capsys.readouterr()
    assert "重放核对范围：重放 digest 核对了全部 8 个第四轮运行" in text and "不能证明" not in text
    assert "本地代码里的短期 / 长期容量与方法名一致" in text


def test_calib4_section_in_report(tmp_path, monkeypatch, capsys):
    d4, d3, d2 = _write_all(tmp_path, with_r3=True)
    r3 = r4_run("dual200_rb", ds="electricity", commit="c3")
    save_runs(d3, [("electricity", "dual200_rb", 0, r3)])
    np.savez(os.path.join(d4, "r2_electricity_dual200_rb_n1_S50_seed0_calib4_t.npz"), **dict(r3, commit=np.array(["c4"])))
    text = _run_main(monkeypatch, d4, d3, d2, str(tmp_path / "ok.md"))
    sec = text.split("### calib4 核对")[1].split("代码版本")[0]
    assert "('electricity', 'dual200_rb', 0)：逐位一致" in sec
    assert "calib4 (" not in text.split("### 一致性与校验问题")[1].split("重放核对范围")[0]
    bad = dict(r3, pred=r3["pred"].copy())
    bad["pred"][300] = (bad["pred"][300] + 1) % 3
    np.savez(os.path.join(d4, "r2_electricity_dual200_rb_n1_S50_seed0_calib4_t.npz"), **bad)
    text = _run_main(monkeypatch, d4, d3, d2, str(tmp_path / "bad.md"))
    assert "**不一致**" in text and "calib4 ('electricity', 'dual200_rb', 0)：以下字段与第三轮不同：pred" in text
    os.remove(os.path.join(d4, "r2_electricity_dual200_rb_n1_S50_seed0_calib4_t.npz"))
    text = _run_main(monkeypatch, d4, d3, d2, str(tmp_path / "none.md"))
    assert "没有核对\"新代码复现第三轮\"" in text
    capsys.readouterr()


# ---------------------------------------------------------------- 图

def _capture_figs(monkeypatch):
    import plot_round4 as P4
    figs = {}
    monkeypatch.setattr(P4, "save", lambda fig, out, name: figs.setdefault(name, fig))
    return P4, figs


def test_plot_helpers_n_text_and_no_data_ticks():
    import plot_round4 as P4
    assert P4.n_text([1]) == "n = 1 seed" and P4.n_text([10, 10]) == "n = 10 seeds"
    assert P4.n_text([8, 10]) == "n = 8-10 seeds" and P4.n_text([0]) == "no data"
    fig, ax = P4.plt.subplots()
    P4._no_data(ax)
    assert list(ax.get_yticks()) == []
    P4.plt.close(fig)


def test_fig_ratio_markers_gaps_and_axis_cap(monkeypatch):
    P4, figs = _capture_figs(monkeypatch)
    S = {}
    for role in ("kdd", "ours"):
        for share in (0.65, 0.85):                                     # 0.75 缺失
            for s_ in range(3):
                S[("gas", R4.method_name(role, share), s_)] = dict(RET=0.97 + 0.02 * s_, ADAPT=0.8)
    S.update({("gas", "sw200", s_): dict(RET=0.5, ADAPT=0.7) for s_ in range(3)})
    P4.fig_ratio(S, [0, 1, 2], "unused")
    fig = figs["fig_ratio"]
    ax = fig.axes[0]                                                  # gas 的 RET 子图
    marks = {ln.get_color(): ln.get_marker() for ln in ax.lines if ln.get_linestyle() == "None"}
    assert marks[P4.C_OURS] == "*" and marks[P4.C_KDD] == "s"           # 与第三轮图 1 一致：ours 红星、KDD 蓝方块
    line = next(ln for ln in ax.lines if ln.get_color() == P4.C_OURS and ln.get_linestyle() != "None")
    y = np.asarray(line.get_ydata(), float)
    assert len(y) == 3 and np.isnan(y[1]) and np.isfinite(y[[0, 2]]).all()    # 缺的 0.75 处断开，不跨点连线
    assert ax.get_ylim()[1] <= 100.5                                  # 准确率轴截到 100
    empty = fig.axes[1]                                               # electricity：没有数据
    assert list(empty.get_yticks()) == []
    assert "n = 3 seeds" in fig._suptitle.get_text()
    P4.plt.close(fig)
