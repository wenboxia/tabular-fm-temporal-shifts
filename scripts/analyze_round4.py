"""第四轮分析：短期占比 0.65 / 0.75 / 0.85 的探索性分析（results/phase56_prereg.md 末尾 "Exploratory short/long ratio analysis"）。

    # 第四轮 npz 在 results/archive/rog_round4/npz，0.75 列 = 第三轮 dual200 / dual200_rb（results/archive/rog_round3/npz），
    # insects 的 sw200 来自第二轮（results/archive/rog_round2/npz）
    python scripts/analyze_round4.py --out results/round4_analysis.md --csv results/round4_per_seed.csv
    # 只看 seed 0 的快速检查（结构、数值是否正常；对 seed 0 的每个第四轮文件重放核对 digest，证明 npz 实际用的容量与方法名一致）
    python scripts/analyze_round4.py --quick
    # 截断自测数据（0.75 列此时没有对应文件，报告里标"缺失"）
    python scripts/analyze_round4.py --dir <dir> --tag _smoke --n_estimators 1 --quick

不调用 TabPFN，只读 npz；缺数据时不崩溃，在报告里标"缺失"（没有对应文件）或"校验失败"（有文件，但标签重放与 npz 不一致）。
这是**探索性、只描述**的分析：做这个分析是看过第三轮结果之后才决定的，不做多重比较校正，不设门槛，不改变第三轮的结论。
报告内容（按预注册的 Reported）：缺失与一致性（含跨轮放行 commit）；每个数据集 × 占比 × {KDD, ours} 的 ADAPT、ADAPT (masked)、RET
（均值 ± SD [95% 区间]）与各阶段 RET；ours − KDD 配对差（95% 区间、正号 seed 数）；长期池构成（标签重放，与 npz digests 核对）；
可选：KDD / ours 与同规模"最近 200 行"的重合随占比变化；calib4 的 npz 与第三轮同名运行的逐位比较。
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from analyze_round3 import get_replay, parse_seeds, pp, table, write_csv  # noqa: E402
from src.eval.round3_analysis import mean_sd_ci, overlap_summary, paired, run_metrics  # noqa: E402
from src.eval.round4_analysis import (DATASETS4, EXPECTED_CAPS, R4_METHODS, SHARES, calib4_check,  # noqa: E402
                                      capacity_problems, check_consistency4, diff_points, expected_runs, load_all4,
                                      method_name, metric_vals, quick_check4)

NAN = float("nan")
ROLE_NAME = {"kdd": "KDD 双记忆", "ours": "ours（阶段均衡）", "sw": "滑窗 sw200（参照）"}
DS_NOTE = {"insects_gradual": "insects_gradual（官方变点臂）"}


# ------------------------------------------------------------------ 格式化

def cell4(v, n_exp, have):
    """mean ± SD [95% 区间]（%）。没有任何文件写"缺失"；有文件但没有有限值写"—"；有限值的 seed 少于 n_exp 时标 n。"""
    if not have:
        return "缺失"
    st = mean_sd_ci(v)
    if st["n"] == 0:
        return "—"
    if st["n"] == 1:
        return f"{pp(st['mean'])}（n=1）"
    tail = f"（n={st['n']}）" if st["n"] < n_exp else ""
    return f"{pp(st['mean'])} ± {pp(st['sd'])} [{pp(st['lo'])}, {pp(st['hi'])}]{tail}"


def pcell4(a, b, have, n_exp):
    """配对差 a − b：均值 [95% 区间]（正号 seed 数/n），单位 pp；不给 p 值（探索性，不做检验）。"""
    pr = paired(a, b)
    if pr["n"] == 0:
        return "缺失" if not have else "—"
    ci = "" if not np.isfinite(pr["lo"]) else f" [{pp(pr['lo'], True)}, {pp(pr['hi'], True)}]"
    tail = f"（n={pr['n']}）" if pr["n"] < n_exp else ""
    return f"{pp(pr['mean'], True)}{ci} ({pr['npos']}/{pr['n']}){tail}"


def have(runs, ds, m, seeds):
    return any((ds, m, s) in runs for s in seeds)


def fail_or_missing(runs, ds, m, seeds):
    """重放没有结果时的单元格词：文件在（重放与 npz 不一致）= "校验失败"，文件不在 = "缺失"。"""
    return "校验失败" if any((ds, m, s) in runs for s in seeds) else "缺失"


def src_of(origin, ds, m, seeds):
    o = next((origin[(ds, m, s)] for s in seeds if (ds, m, s) in origin), None)
    return {"r2": "第二轮", "r3": "第三轮", "r4": "第四轮", None: "—"}[o]


def caps_text(share):
    sc, lc = EXPECTED_CAPS[share]
    return f"{sc}/{lc}"


def n_stages(runs, ds):
    """评估阶段数 − 1 = 阶段边界数。"""
    for (d_, m, s), d in sorted(runs.items()):
        if d_ == ds:
            return len(d["boundaries"])
    return 0


def present_datasets(runs):
    return [ds for ds in DATASETS4 if any(k[0] == ds for k in runs)]


# ------------------------------------------------------------------ --quick

def quick(args, runs, origin, probs):
    qs = args.quick_seed
    P = print
    P(f"== 第四轮快检（seed {qs}，目录 {args.dir}，tag '{args.tag}'，n_estimators {args.n_estimators}）==")
    P("（某数据集在第四轮还没有任何文件时只标\"暂无结果\"，不算问题；0.75 列来自第三轮 "
      f"{args.round3_dir}，sw200 来自第三轮 / 第二轮，缺失只作提醒；"
      "对 seed 0 的每个第四轮文件做标签重放，用 digest 核对 npz 实际用的容量是否与方法名一致，这一步约需几十秒）\n")
    for p in probs:
        P(p if p.startswith("!!") else f"注意：{p}")
    cal = calib4_check(args.dir, runs, origin, args.n_estimators, args.tag)
    calib_bad = [c for c in cal if c["status"] == "differs"]
    if cal:
        for c in cal:
            P(f"calib4 核对 {c['key']}：{ {'identical': '逐位一致', 'differs': '**不一致**', 'no_r3': '无法比较'}[c['status']] }（{c['detail']}）")
    else:
        P("calib4：目录里没有 calib4 的 npz，报告里无法核对新代码是否复现第三轮（ROG 上的 calib4 闸门报告另见 calib4_report.txt）")
    P("")
    verdicts = {}
    nocross = []
    for ds in DATASETS4:
        r4 = [m for m in R4_METHODS if (ds, m, qs) in runs]
        P(f"[{ds}]")
        if not r4:
            P("  暂无结果文件\n")
            verdicts[ds] = None
            continue
        reasons = []
        for m in R4_METHODS:
            d = runs.get((ds, m, qs))
            if d is None:
                P(f"  ({m}) 缺失")
                reasons.append(f"缺 {m}")
                continue
            q = quick_check4(d)
            for ln in q["lines"]:
                P("  " + ln)
            for n_ in q["notes"]:
                P(f"    提醒：{n_}")
            for r in q["reasons"]:
                P(f"    !! {r}")
                reasons.append(f"{m}: {r}")
            msr = int(d["max_stream_rows"][0]) if "max_stream_rows" in d else -1
            if msr >= 0 and "_smoke" not in args.tag:
                P(f"    !! 截断运行（max_stream_rows={msr}），不是正式结果")
                reasons.append(f"{m}: 截断运行（max_stream_rows={msr}），不是正式结果")
        for m in ("dual200", "dual200_rb", "sw200"):
            d = runs.get((ds, m, qs))
            if d is None:
                P(f"  ({m}) 提醒：{'第二轮' if (ds, m) == ('insects', 'sw200') else '第三轮'}文件缺失（--round3_dir / --round2_dir 路径？），"
                  "无法做跨轮比较与 0.75 列")
                nocross.append(f"{ds}/{m}")
                continue
            mm = run_metrics(d)
            f = lambda v: "nan" if not np.isfinite(v) else f"{v * 100:.2f}"      # noqa: E731
            P(f"  [{ds}/{m} seed {qs}]（{src_of(origin, ds, m, [qs])}，commit {str(d['commit'][0])}，device {str(d['device'][0])}）"
              f"ADAPT={f(mm['ADAPT'])}  ADAPT(masked)={f(mm['ADAPT_masked'])}  RET={f(mm['RET'])}")
            cap = capacity_problems(d)
            for c in cap:
                P(f"    !! {c}")
                reasons.append(f"{m}: {c}")
        sub = {k: v for k, v in runs.items() if k[0] == ds and k[2] == qs}
        for c in check_consistency4(sub, {k: origin[k] for k in sub}):
            P(f"    !! 一致性：{c}")
            reasons.append(f"一致性：{c}")
        verdicts[ds] = reasons
        P(f"  判断：{'可以继续' if not reasons else '需要看 —— ' + '；'.join(reasons)}\n")
    done = {k: v for k, v in verdicts.items() if v is not None}
    bad = [k for k, v in done.items() if v]
    if not done:
        P("总结：还没有任何第四轮结果——先确认 --dir 指向 npz 所在目录（不是没有问题）。")
        return
    extra = []
    if nocross:
        extra.append("注意：第三轮 / 第二轮文件没有载入（检查 --round3_dir / --round2_dir），没有做跨轮核对；缺：" + "、".join(nocross))
    if calib_bad:
        extra.append("注意：calib4 与第三轮同名运行不一致，见上面的 calib4 核对")
    P(f"总结：{len(done)} 个数据集已有结果，其中 {len(done) - len(bad)} 个可以继续"
      + (f"，需要看：{', '.join(bad)}。" if bad else ("，各数据集内部没有发现问题（跨轮核对见下一行）。" if extra else "，没有发现问题。"))
      + (f" 暂无结果：{', '.join(k for k, v in verdicts.items() if v is None)}。" if len(done) < len(verdicts) else ""))
    for e in extra:
        P(e)


# ------------------------------------------------------------------ 完整分析的各段

def section_levels(P, runs, origin, S, seeds):
    P("## a. 各占比的 ADAPT、ADAPT (masked)、RET（seed 间均值 ± SD [95% t 区间]，单位 %）\n")
    P("\"占比\"是短期记忆所占的比例，括号里是 短期/长期 行数（总共 200 行）。0.75 = 第三轮的 `dual200` / `dual200_rb`；"
      "最后一行 sw200 是同规模滑窗（只用最近 200 行），作参照。单元格末尾的\"（n=k）\"表示只有 k 个 seed 有有限值；\"缺失\"= 没有对应文件；"
      "\"—\"= 有文件，但没有可用的数值（例如还没有阶段结束，没有 RET）。\n")
    for ds in present_datasets(runs):
        P(f"### {DS_NOTE.get(ds, ds)}\n")
        rows = []
        for share in SHARES:
            for role in ("kdd", "ours"):
                m = method_name(role, share)
                h = have(runs, ds, m, seeds)
                rows.append([f"{share:.2f}（{caps_text(share)}）", ROLE_NAME[role], m, src_of(origin, ds, m, seeds)]
                            + [cell4(metric_vals(S, ds, m, k, seeds), len(seeds), h) for k in ("ADAPT", "ADAPT_masked", "RET")])
        h = have(runs, ds, "sw200", seeds)
        rows.append(["—", ROLE_NAME["sw"], "sw200", src_of(origin, ds, "sw200", seeds)]
                    + [cell4(metric_vals(S, ds, "sw200", k, seeds), len(seeds), h) for k in ("ADAPT", "ADAPT_masked", "RET")])
        table(P, ["短期占比（短/长）", "角色", "方法", "来源", "ADAPT", "ADAPT (masked)", "RET"], rows)


def section_paired(P, runs, S, seeds):
    P("## b. 配对差 ours − KDD（同占比、同 seed；单位 pp）\n")
    P("单元格：差值均值 [95% t 区间] (差值为正的 seed 数/n)。**不给 p 值，不做检验**：这是探索性描述，没有多重比较校正，也没有门槛。"
      "ΔADAPT 的 −0.5 pp 是第三轮的非劣性边界，这里只作参照，不当作本分析的判据。"
      "\"缺失\"= 没有对应文件；\"—\"= 有文件，但没有可用的配对值（例如还没有阶段结束，没有 RET）。\n")
    ds_list = present_datasets(runs)
    P("### b.0 随占比变化（均值，pp）\n")
    rows = []
    for ds in ds_list:
        for key, name in (("RET", "ΔRET"), ("ADAPT", "ΔADAPT")):
            pts = {p["share"]: p for p in diff_points(S, seeds, [ds], key)}
            miss = lambda s: ("缺失" if not (have(runs, ds, method_name("ours", s), seeds)          # noqa: E731
                                            and have(runs, ds, method_name("kdd", s), seeds)) else "—")
            rows.append([DS_NOTE.get(ds, ds), name] + [(miss(s) if pts[s]["n"] == 0 else f"{pts[s]['mean']:+.2f}（{pts[s]['npos']}/{pts[s]['n']}）")
                                                       for s in SHARES])
    table(P, ["数据集", "指标"] + [f"占比 {s:.2f}（正号 seed 数/n）" for s in SHARES], rows)
    for ds in ds_list:
        P(f"### {DS_NOTE.get(ds, ds)}\n")
        rows = []
        for share in SHARES:
            o, k = method_name("ours", share), method_name("kdd", share)
            h = have(runs, ds, o, seeds) and have(runs, ds, k, seeds)
            rows.append([f"{share:.2f}（{caps_text(share)}）"]
                        + [pcell4(metric_vals(S, ds, o, key, seeds), metric_vals(S, ds, k, key, seeds), h, len(seeds))
                           for key in ("ADAPT", "ADAPT_masked", "RET")])
        table(P, ["短期占比（短/长）", "ΔADAPT", "ΔADAPT (masked)", "ΔRET"], rows)


def section_stages(P, runs, S, seeds):
    P("## c. 各阶段的 RET\n")
    P("RET_Rj = 评估阶段 j 结束之后所有检查点上该阶段留出集准确率的平均（定义同第二、三轮）。靠近流末的阶段只在结束后很短的窗口里被测量，"
      "如果那里差异小或为负，不能读成\"ours 遗忘了最后一个阶段\"。渐变段只有 R0 一个旧阶段。\"缺失\"/\"—\"的含义同 a、b 段。\n")
    for ds in present_datasets(runs):
        K = n_stages(runs, ds)
        if K < 1:
            continue
        P(f"### {DS_NOTE.get(ds, ds)}：RET_Rj（均值 ± SD [95%]，%）\n")
        rows = []
        for share in SHARES:
            for role in ("kdd", "ours"):
                m = method_name(role, share)
                h = have(runs, ds, m, seeds)
                rows.append([f"{share:.2f}", ROLE_NAME[role]]
                            + [cell4(metric_vals(S, ds, m, f"RET_R{j}", seeds), len(seeds), h) for j in range(K)])
        h = have(runs, ds, "sw200", seeds)
        rows.append(["—", ROLE_NAME["sw"]] + [cell4(metric_vals(S, ds, "sw200", f"RET_R{j}", seeds), len(seeds), h) for j in range(K)])
        table(P, ["占比", "角色"] + [f"R{j}" for j in range(K)], rows)
        P(f"{DS_NOTE.get(ds, ds)}：各阶段 RET 的 ours − KDD（pp；[95% 区间] (正号 seed 数/n)）\n")
        rows = []
        for share in SHARES:
            o, k = method_name("ours", share), method_name("kdd", share)
            h = have(runs, ds, o, seeds) and have(runs, ds, k, seeds)
            rows.append([f"{share:.2f}"] + [pcell4(metric_vals(S, ds, o, f"RET_R{j}", seeds),
                                                   metric_vals(S, ds, k, f"RET_R{j}", seeds), h, len(seeds)) for j in range(K)])
        table(P, ["占比"] + [f"R{j}" for j in range(K)], rows)


def section_pool(P, runs, seeds_replay, cache, probs):
    P("## d. 长期池构成（标签重放，不跑模型）\n")
    s0 = seeds_replay[0] if seeds_replay else 0
    P(f"用真实的 run_stream 与记忆类重放（哑预测器，只用标签），每个检查点的 `memory.digest()` 与 npz 的 digests 核对，不一致就报校验失败"
      f"（失败的运行不进入数字表格，格子里写\"校验失败\"，原因见开头的问题列表；\"缺失\"则是没有对应文件）。下面只取 seed {s0}；`dual*_rb` 的配额只依赖标签和阶段号，各 seed 的比例相同。"
      "表中数字 = 长期池里属于各评估阶段 Rj 的行数；长期池总容量：占比 0.65 → 70，0.75 → 50，0.85 → 30 行。"
      "\"R0 行\"在渐变段里含过渡区的行。\n")
    for ds in present_datasets(runs):
        reps = {}
        for share in SHARES:
            for role in ("kdd", "ours"):
                m = method_name(role, share)
                r = get_replay(cache, probs, runs, ds, m, s0)
                if r:
                    reps[(share, role)] = r
        if not reps:
            word = "校验失败" if any((ds, method_name(r_, s_), s0) in runs for s_ in SHARES for r_ in ("kdd", "ours")) else "缺失"
            P(f"### {DS_NOTE.get(ds, ds)}（seed {s0}）\n\n（{word}：没有可用的重放结果）\n")
            continue
        some = next(m_ for m_ in (method_name(r_, s_) for s_ in SHARES for r_ in ("kdd", "ours")) if (ds, m_, s0) in runs)
        d0 = runs[(ds, some, s0)]
        n_reg = next(iter(reps.values()))["n_reg"]
        msr = int(d0["max_stream_rows"][0])
        n_end = len(d0["y"]) if msr < 0 else min(len(d0["y"]), msr)
        times = [int(b) for b in d0["boundaries"] if 0 < b < n_end] + [n_end]
        P(f"### {DS_NOTE.get(ds, ds)}（seed {s0}）\n")
        rows = []
        for t in times:
            for share in SHARES:
                for role in ("kdd", "ours"):
                    r = reps.get((share, role))
                    if r is None:
                        rows.append([("终点 " if t == n_end else "边界 ") + str(t), f"{share:.2f}", method_name(role, share)]
                                    + [fail_or_missing(runs, ds, method_name(role, share), [s0])] * (n_reg + 1))
                        continue
                    i = np.flatnonzero(r["ts"] == t)
                    if not len(i):
                        continue
                    i = int(i[0])
                    rows.append([("终点 " if t == n_end else "边界 ") + str(t), f"{share:.2f}", method_name(role, share)]
                                + [int(r["comp"][j, i]) for j in range(n_reg)] + [int(r["long_len"][i])])
        table(P, ["时刻", "占比", "方法"] + [f"R{j}" for j in range(n_reg)] + ["长期池总行数"], rows)


def section_overlap(P, runs, seeds_replay, cache, probs):
    P("## e. （可选描述）记忆与\"最近 200 行\"的重合随占比变化（标签重放）\n")
    P("每批更新之后，整个记忆（长期加短期）的行集合中落在\"最近 200 个已揭示行\"（= 同规模滑窗 sw200 的上下文）里的比例；"
      "只看记忆已满（已揭示行数 ≥ 200）之后的批次，均值 / 最小值，对重放的 seed 取平均 / 取最小（单位 %）。"
      f"seed：{seeds_replay}。占比越高，短期窗口越长，与滑窗越重合。\"缺失\"= 没有对应文件，\"校验失败\"= 有文件但重放与 npz 不一致。\n")
    rows = []
    for ds in present_datasets(runs):
        for role in ("kdd", "ours"):
            cells = []
            for share in SHARES:
                m = method_name(role, share)
                sm = [overlap_summary(r) for r in (get_replay(cache, probs, runs, ds, m, s) for s in seeds_replay) if r]
                n_files = sum(1 for s in seeds_replay if (ds, m, s) in runs)
                if not sm:
                    cells.append(fail_or_missing(runs, ds, m, seeds_replay))
                else:
                    tail = f"（{len(sm)}/{n_files} 个 seed 通过重放）" if len(sm) < n_files else ""
                    cells.append(f"{pp(np.mean([x['mean_full'] for x in sm]))}（最小 {pp(np.min([x['min_full'] for x in sm]))}）{tail}")
            rows.append([DS_NOTE.get(ds, ds), ROLE_NAME[role]] + cells)
    table(P, ["数据集", "角色"] + [f"占比 {s:.2f}" for s in SHARES], rows)


def section_replay_summary(P, runs, cache):
    ok = sum(1 for r in cache.values() if r)
    bad = sum(1 for k, r in cache.items() if r is None and k in runs)
    P("### 重放校验合计\n")
    P(f"重放校验（digests / closures / 检查点时刻与 npz 比对）：通过 {ok} 个，失败 {bad} 个（失败的见开头的问题列表，已从 d、e 段的表中排除）。\n")


READING = (
    "**读法（请先读这一段）：**\n\n"
    "1. 这是**探索性**分析，只描述，不检验。**做这个分析**是看过第三轮结果**之后**才决定的（起因：Electricity 在 200 行时的适应代价，"
    "以及阶段均衡的保留增益是否依赖第二、三轮沿用的 75/25 划分，即 KDD 论文调参网格的中间值）。0.65 / 0.85 是 KDD 论文网格里的另外两个值，不是看过数据后挑的；"
    "但 KDD 论文只在 600 / 800 / 1000 行的总记忆上搜过这个网格，200 行不在它的搜索范围内。\n"
    "2. **不做多重比较校正，不设门槛**：表里的区间是各自独立的 95% t 区间（每个 (数据集, 占比, 指标) 一个），数量很多，"
    "偶然的\"全部为正\"或\"区间不含 0\"不能当作证据；第三轮的结论（Holm 家族的 8 个检验）不被本分析改变。\n"
    "3. 10 个 seed 不是 10 个独立的流：seed 只改变留出集的抽样和 TabPFN 的随机性，流（去掉各自的留出行之后）在 seed 之间大部分相同，"
    "阶段均衡长期池里选到的行也几乎相同，所以\"正号 seed 数\"主要反映留出集抽样噪声，区间偏窄。\n"
    "4. 只在 **200 行**上做；相对 KDD 的 ADAPT 差（Electricity）：1000 行 −0.51 pp，200 行 −2.10 pp，两者都没过 −0.5 pp 的非劣性边界；"
    "本分析不能推到 1000 行。\n"
    "5. 0.75 列是第三轮的 `dual200` / `dual200_rb`（commit 不同于第四轮）。`calib4` 只在 electricity / dual200_rb / seed 0 这"
    "**一个**运行上逐位复现了第三轮（本脚本读 calib4 的 npz 再核对一次，见下面的\"calib4 核对\"）；其余运行不能由它推出，"
    "靠本脚本逐 seed 核对 y / hold_idx / stream_idx / boundaries / zone / mem_boundaries 相同，以及标签重放的 digest。\n"
)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results/round4", help="第四轮 npz 目录")
    ap.add_argument("--round3_dir", default="results/round3", help="第三轮 npz 目录（0.75 列与 sw200）")
    ap.add_argument("--round2_dir", default="results/round2", help="第二轮 npz 目录（insects 的 sw200）")
    ap.add_argument("--n_estimators", type=int, default=4)
    ap.add_argument("--tag", default="", help="第四轮文件名后缀（截断自测用 _smoke）")
    ap.add_argument("--round3_tag", default="", help="第三轮文件名后缀（正式结果为空）")
    ap.add_argument("--seeds", default="0-9", help="预期的 seed（用来报告缺失），如 0-9")
    ap.add_argument("--quick", action="store_true", help="seed 0 快检，打印中文判断后退出")
    ap.add_argument("--quick_seed", type=int, default=0)
    ap.add_argument("--replay_seeds", default="0-9", help="d、e 段重放的 seed（d 段只用第一个）")
    ap.add_argument("--out", default="results/round4_analysis.md")
    ap.add_argument("--csv", default="results/round4_per_seed.csv")
    args = ap.parse_args()

    runs, origin, probs = load_all4(args.dir, args.round3_dir, args.round2_dir, args.n_estimators, args.tag, args.round3_tag)
    if args.quick:
        return quick(args, runs, origin, probs)
    if not any(v == "r4" for v in origin.values()):
        raise SystemExit("\n".join(probs) or f"{args.dir} 里没有第四轮的 npz")
    trunc = sorted(k for k, d in runs.items() if origin[k] == "r4" and "max_stream_rows" in d and int(d["max_stream_rows"][0]) >= 0)
    if trunc and "_smoke" not in args.tag:
        probs.append(f"注意：{len(trunc)} 个第四轮文件是截断运行（max_stream_rows ≥ 0，如 {trunc[0]}），不是正式结果")

    seeds = parse_seeds(args.seeds)
    probs += check_consistency4(runs, origin)
    for k, d in sorted(runs.items()):
        if origin[k] == "r4":
            probs += [f"{k}: {c}" for c in capacity_problems(d)]
    S = {k: run_metrics(d) for k, d in runs.items()}

    lines = []
    P = lambda *a: lines.append(" ".join(str(x) for x in a))      # noqa: E731
    P(f"# 第四轮分析：短期占比 0.65 / 0.75 / 0.85（200 行，探索性；n_estimators={args.n_estimators}"
      f"{'，tag ' + repr(args.tag) if args.tag else ''}）\n")
    P("按 results/phase56_prereg.md 末尾 \"Exploratory short/long ratio analysis\" 实现；不调用 TabPFN；"
      "**只描述，不检验，不设门槛**。数字单位：准确率为 %，差值为 pp。\n")
    P(READING)
    n4 = sum(1 for v in origin.values() if v == "r4")
    P(f"载入 {len(runs)} 个运行：第四轮 {n4} 个，第三轮 {sum(1 for v in origin.values() if v == 'r3')} 个，"
      f"第二轮 {sum(1 for v in origin.values() if v == 'r2')} 个。\n")
    cal = calib4_check(args.dir, runs, origin, args.n_estimators, args.tag)
    P("### calib4 核对\n")
    if not cal:
        P(f"{args.dir} 里没有 calib4 的 npz（r2_*_n{args.n_estimators}_S50_seed*_calib4{args.tag}.npz）：本报告没有核对\"新代码复现第三轮\"。"
          "ROG 上的闸门结果见 calib4_report.txt。\n")
    for c in cal:
        P(f"- `{c['name']}` 对第三轮 {c['key']}：{ {'identical': '逐位一致', 'differs': '**不一致**', 'no_r3': '无法比较'}[c['status']] }。{c['detail']}。")
        if c["status"] == "differs":
            probs.append(f"calib4 {c['key']}：{c['detail']}")
    if cal:
        P("")
    P("代码版本（npz 里的 commit 字段）：0.75 列与 sw200 的 commit 与第四轮不同是预期的，数据一致性见\"一致性与校验问题\"。\n")
    ver = {}
    for (ds_, m_, s_), d_ in sorted(runs.items()):
        ver.setdefault((ds_, src_of(origin, ds_, m_, [s_]), str(d_["commit"][0]), str(d_["device"][0])), []).append(m_)
    table(P, ["数据集", "来源", "commit", "device", "方法（个数 × seed）"],
          [[ds_, src_, c_, dv_, ", ".join(f"{m_}×{ms.count(m_)}" for m_ in sorted(set(ms)))] for (ds_, src_, c_, dv_), ms in ver.items()])
    P("### 缺失的 (数据集, 方法, seed)\n")
    miss = []
    for ds, m, org in expected_runs():
        ms = [s for s in seeds if (ds, m, s) not in runs]
        if ms:
            miss.append(f"- {ds}/{m}（应来自{ {'r2': '第二轮', 'r3': '第三轮', 'r4': '第四轮'}[org] }）: 缺 seed {ms}")
    if not miss:
        P(f"无（预期 seed {seeds}）。\n")
    else:
        for x in miss:
            P(x)
        P("")

    cache = {}
    seeds_replay = [s for s in parse_seeds(args.replay_seeds) if any(k[2] == s for k in runs)] or sorted({k[2] for k in runs})[:1]
    body = []
    Pb = lambda *a: body.append(" ".join(str(x) for x in a))      # noqa: E731
    section_levels(Pb, runs, origin, S, seeds)
    section_paired(Pb, runs, S, seeds)
    section_stages(Pb, runs, S, seeds)
    section_pool(Pb, runs, seeds_replay, cache, probs)
    section_overlap(Pb, runs, seeds_replay, cache, probs)
    section_replay_summary(Pb, runs, cache)
    P("### 一致性与校验问题\n")
    r4_keys = [k for k in runs if origin[k] == "r4"]
    ok_seeds = sorted({k[2] for k in r4_keys if cache.get(k)})
    n_ok = sum(1 for k in r4_keys if cache.get(k))
    if n_ok == len(r4_keys):
        cover = f"重放 digest 核对了全部 {n_ok} 个第四轮运行（seed {ok_seeds}），npz 实际用的容量以此为准。"
    else:
        cover = (f"重放 digest 只核对了 {n_ok}/{len(r4_keys)} 个第四轮运行（通过的 seed：{ok_seeds or '无'}）；"
                 "其余运行的容量只靠方法名和代码级检查，**不能证明** npz 实际用的容量与方法名一致。")
    if not probs:
        P("无（y / hold_idx / stream_idx / boundaries / zone / mem_boundaries 在同一 (数据集, seed) 的所有方法之间相同，含第三轮的 0.75 列；"
          "每个 (数据集, 来源) 内 commit / device 唯一；leak_count = 0；上下文不超预算 200；"
          "本地代码里的短期 / 长期容量与方法名一致（130/70、150/50、170/30）；完整运行没有 -1；已重放的运行 digest 与 npz 一致）。\n")
    else:
        for p in probs:
            P(f"- {p}")
        P("")
    P("重放核对范围：" + cover + "\n")
    lines += body

    text = "\n".join(lines)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"[written] {args.out}")
    if args.csv:
        write_csv(args.csv, runs, S)
        print(f"[written] {args.csv}")


if __name__ == "__main__":
    main()
