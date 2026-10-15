"""第三轮分析（按 results/phase56_prereg.md 的 Amendment 3 及其后的 Note；预注册之外的 p 值一律标明是描述性的）。

    # 第三轮 npz 在 results/round3（scripts/reproduce_round3.sh 的输出），
    # insects 的 1000 行与滑窗来自第二轮（results/round2，scripts/reproduce_round2.sh 的输出）
    python scripts/analyze_round3.py --out results/round3_analysis.md --csv results/round3_per_seed.csv
    # npz 在别的目录时：--dir <第三轮目录> --round2_dir <第二轮目录>
    # 只看 seed 0 的快速检查（结构与数值是否正常）
    python scripts/analyze_round3.py --quick
    # 截断自测数据
    python scripts/analyze_round3.py --dir <dir> --tag _smoke --n_estimators 1 --quick

不调用 TabPFN，只读 npz；缺数据时不崩溃，在报告里标出缺什么。第三轮没有成功门槛，本报告是描述性的。
段落对应 Amendment 3：a 可靠性、b 配对差与 Holm、c KDD 与滑窗的重合、d 机制回放、e 渐变段、
f 阶段差异（探索性，看过 Gas 的 CPU 结果之后定义）、g 不用 TabPFN 的基线（预注册口径与按批揭示口径并列）。
看过第三轮结果之后补的描述性小节：h 单类连续段掩码去掉的行比例、i 阶段结束后记忆里还剩什么（行数、类别覆盖、行年龄）、
j 各阶段流上准确率差（ours − KDD）；渐变段（e 段）另含 KDD − 滑窗、(b)(c) − (a)、最终长期池构成、逐类召回。
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eval.round3_analysis import (DATASET_METHODS, HOLM_DATASETS, NONINF_MARGIN,  # noqa: E402
                                      SIZES, adwin_alarm_info, age_summary, all_pairs, baselines_full,
                                      check_consistency, class_coverage, class_share,
                                      final_pool_by_position, find_missing, first_diff_row, holdout_class_recall,
                                      holm, last_present_time, load_runs, mask_removed_share, mean_sd_ci,
                                      noninferiority, overlap_summary, paired, parse_method, quick_check,
                                      regime_leftover, replay_run, run_metrics, spearman, stream_acc_per_regime,
                                      stage_auc)

NAN = float("nan")


def parse_seeds(s: str) -> list:
    """'0-9' 或 '0,1,5' 或 '0-3,7'。"""
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def load_all(args):
    """载入第二轮（insects 的四个方法）与第三轮的结果。返回 (runs, problems, runs3_keys)。"""
    keep = all_pairs()
    runs2, p2 = load_runs([args.round2_dir], args.n_estimators, 50, keep=keep, tags=[""])
    runs3, p3 = load_runs([args.dir], args.n_estimators, 50, keep=keep, tags=[args.tag])
    probs = p2 + p3
    for k in sorted(set(runs2) & set(runs3)):
        probs.append(f"{k} 在第二轮与第三轮目录里都有，用第三轮的")
    if not os.path.isdir(args.dir):
        probs.append(f"!! 第三轮目录 {args.dir} 不存在（先运行 scripts/reproduce_round3.sh，或用 --dir 指向 npz 所在目录）")
    elif not runs3:
        probs.append(f"!! {args.dir} 里没有 r2_*_n{args.n_estimators}_S50_seed*{args.tag}.npz（npz 是否在子目录里？）")
    runs = {**runs2, **runs3}
    return runs, probs, set(runs3)


# ------------------------------------------------------------------ 格式化

def pp(v, plus=False):
    if v is None or not np.isfinite(v):
        return "nan"
    return f"{v * 100:+.2f}" if plus else f"{v * 100:.2f}"


def cell(st, n_exp=None):
    """mean ± SD [95% CI]，单位 %。有限值的 seed 数少于 n_exp（预期 seed 数）时在末尾标 n。"""
    if st["n"] == 0:
        return "—"
    if st["n"] == 1:
        return f"{pp(st['mean'])}（n=1）"
    tail = f"（n={st['n']}）" if n_exp is not None and st["n"] < n_exp else ""
    return f"{pp(st['mean'])} ± {pp(st['sd'])} [{pp(st['lo'])}, {pp(st['hi'])}]{tail}"


def pcell(pr):
    """配对差：均值 [CI] p（正号 seed 数/n），单位 pp。"""
    if pr["n"] == 0:
        return "—"
    p = "nan" if not np.isfinite(pr["p"]) else f"{pr['p']:.3g}"
    ci = "" if not np.isfinite(pr["lo"]) else f" [{pp(pr['lo'], True)}, {pp(pr['hi'], True)}]"
    return f"{pp(pr['mean'], True)}{ci} p={p} ({pr['npos']}/{pr['n']})"


def table(P, header, rows):
    P("| " + " | ".join(header) + " |")
    P("|" + "---|" * len(header))
    for r in rows:
        P("| " + " | ".join(str(x) for x in r) + " |")
    P("")


def vals(S, ds, m, key, seeds):
    """按 seeds 取一列指标，缺的是 nan（配对时按 seed 对齐）。"""
    return np.array([S.get((ds, m, s), {}).get(key, NAN) for s in seeds], float)


def methods_of(ds, size):
    """该数据集在某规模下的方法，顺序 sw、KDD、ours、ours_adwin。"""
    ms = [m for m in DATASET_METHODS[ds] if parse_method(m)[1] == size]
    rank = lambda m: {"sw": 0, "kdd": 1, "ours": 2}[parse_method(m)[0]] + (0.5 if parse_method(m)[2] else 0)   # noqa: E731
    return sorted(ms, key=rank)


def role_name(m):
    role, size, adw = parse_method(m)
    return {"sw": "滑窗", "kdd": "KDD 双记忆", "ours": "ours（阶段均衡）"}[role] + ("，ADWIN" if adw else "")


def rng(v, nd=1, pct=False):
    """一组数的范围文本：全相同时只写一个数，否则 '均值（最小–最大）'。pct=True 时乘 100。"""
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if not len(v):
        return "—"
    if pct:
        v = v * 100
    if v.min() == v.max():
        return f"{v[0]:.{nd}f}"
    return f"{v.mean():.{nd}f}（{v.min():.{nd}f}–{v.max():.{nd}f}）"


def lohi(v, nd=1):
    """'最小–最大'（相同则只写一个）。"""
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if not len(v):
        return "—"
    return f"{v.min():.{nd}f}" if v.min() == v.max() else f"{v.min():.{nd}f}–{v.max():.{nd}f}"


def n_regimes(runs, ds):
    for (d_, m, s), d in sorted(runs.items()):
        if d_ == ds:
            return len(d["boundaries"]) + 1
    return 0


# ------------------------------------------------------------------ --quick

def quick(args, runs, runs3_keys, probs):
    qs = args.quick_seed
    P = print
    P(f"== 第三轮快检（seed {qs}，目录 {args.dir}，tag '{args.tag}'，n_estimators {args.n_estimators}）==")
    P("（某数据集在第三轮还没有任何文件时，缺的方法只标\"尚未运行\"，不算问题）\n")
    for p in probs:
        P(f"注意：{p}")
    verdicts = {}
    for ds, ms in DATASET_METHODS.items():
        have = [m for m in ms if (ds, m, qs) in runs]
        P(f"[{ds}]")
        if not have:
            P("  暂无结果文件\n")
            verdicts[ds] = None
            continue
        has3 = any((ds, m, qs) in runs3_keys for m in ms)
        reasons = []
        for m in ms:
            d = runs.get((ds, m, qs))
            if d is None:
                if has3:
                    P(f"  ({m}) 缺失")
                    reasons.append(f"缺 {m}")
                else:
                    P(f"  ({m}) 尚未运行")
                continue
            q = quick_check(d)
            origin = "" if (ds, m, qs) in runs3_keys else "（第二轮）"
            P("  " + q["lines"][0].replace("]", f"]{origin}", 1))
            for ln in q["lines"][1:]:
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
        sub = {k: v for k, v in runs.items() if k[0] == ds and k[2] == qs}
        for c in check_consistency(sub):
            P(f"    !! 一致性：{c}")
            reasons.append(f"一致性：{c}")
        if not has3:
            verdicts[ds] = None
            P("  判断：第三轮还没有该数据集的结果（上面只有第二轮的文件），等第三轮运行完成\n")
            continue
        verdicts[ds] = reasons
        P(f"  判断：{'可以继续' if not reasons else '需要看 —— ' + '；'.join(reasons)}\n")
    done = {k: v for k, v in verdicts.items() if v is not None}
    bad = [k for k, v in done.items() if v]
    if not done:
        P("总结：还没有任何第三轮结果——先确认 --dir 指向 npz 所在目录（不是没有问题）。")
        return
    P(f"总结：{len(done)} 个数据集已有结果，其中 {len(done) - len(bad)} 个可以继续"
      + (f"，需要看：{', '.join(bad)}。" if bad else "，没有发现问题。")
      + (f" 暂无结果：{', '.join(k for k, v in verdicts.items() if v is None)}。" if len(done) < len(verdicts) else ""))


# ------------------------------------------------------------------ 完整分析的各段

def section_reliability(P, runs, S, seeds):
    P("## a. 可靠性（seed 间均值 ± SD [95% t 区间]，单位 %）\n")
    P("FGT = 各已结束阶段 FGT_j（最佳 − 最终留出准确率，定义同第二轮）在每个文件内的平均，再跨 seed 统计；"
      "各阶段 FGT_Rj 见 CSV。单元格末尾的\"（n=k）\"表示该方法只有 k 个 seed 有有限值（少于预期的 "
      f"{len(seeds)} 个）。RET_Rj = 阶段 j 结束之后所有检查点上留出准确率的平均，所以靠近流末的阶段（如 electricity R8、"
      "insects R4、gas R6）只在结束后很短的窗口里被测量，那时 KDD 的记忆刚被这些阶段填满；那里增益小或为负，"
      "不能读成\"ours 遗忘了最后一个阶段\"。\n")
    for ds in DATASET_METHODS:
        if not any(k[0] == ds for k in runs):
            P(f"### {ds}\n\n（没有结果）\n")
            continue
        K = n_regimes(runs, ds) - 1
        for size in SIZES:
            ms = [m for m in methods_of(ds, size) if any((ds, m, s) in runs for s in seeds)]
            if not ms:
                continue
            P(f"### {ds}，{size} 行\n")
            rows = []
            for m in ms:
                rows.append([m, role_name(m)] + [cell(mean_sd_ci(vals(S, ds, m, k, seeds)), len(seeds))
                                                  for k in ("ADAPT", "ADAPT_masked", "RET", "FGT")])
            table(P, ["方法", "角色", "ADAPT", "ADAPT (masked)", "RET", "FGT"], rows)
            if K >= 1:
                rows = [[m] + [cell(mean_sd_ci(vals(S, ds, m, f"RET_R{j}", seeds)), len(seeds)) for j in range(K)] for m in ms]
                P("各阶段 RET（RET_Rj）：\n")
                table(P, ["方法"] + [f"R{j}" for j in range(K)], rows)


def section_paired(P, runs, S, seeds):
    P("## b. 配对差（ours − KDD、ours − 滑窗，同规模；双侧配对 t 检验；单位 pp）\n")
    P("单元格：差值均值 [95% CI] p=双侧 p (差值为正的 seed 数/n)。**第三轮没有成功门槛**，这里只描述。"
      "\"KDD − 滑窗\"一行用来区分双记忆结构本身的效果与阶段均衡的效果：该行接近 0 时，ours − KDD 就等于 ours − 只保留最近窗口。\n")
    for ds in DATASET_METHODS:
        if ds == "insects_gradual_start":
            continue
        for size in SIZES:
            ours, kdd, sw = f"dual{size}_rb", f"dual{size}", f"sw{size}"
            if not any((ds, ours, s) in runs for s in seeds):
                continue
            P(f"### {ds}，{size} 行\n")
            rows = []
            for name, a_, base in (("ours − KDD", ours, kdd), ("ours − 滑窗", ours, sw), ("KDD − 滑窗", kdd, sw)):
                rows.append([name] + [pcell(paired(vals(S, ds, a_, k, seeds), vals(S, ds, base, k, seeds)))
                                      for k in ("ADAPT", "ADAPT_masked", "RET")])
            table(P, ["比较", "ADAPT", "ADAPT (masked)", "RET"], rows)
            K = n_regimes(runs, ds) - 1
            if K >= 1:
                ks = [f"RET_R{j}" for j in range(K)] + ["FGT"]
                rows = [[name] + [pcell(paired(vals(S, ds, a_, k, seeds), vals(S, ds, base, k, seeds))) for k in ks]
                        for name, a_, base in (("ours − KDD", ours, kdd), ("ours − 滑窗", ours, sw), ("KDD − 滑窗", kdd, sw))]
                P("各阶段 RET 与 FGT 的配对差（描述性，不进 Holm）。**FGT 越低越好**：FGT 差为负 = ours 遗忘更少，"
                  "FGT 一列括号里的\"正号 seed 数\"是 ours 遗忘更多的 seed 数，含义与 RET 列相反。\n")
                table(P, ["比较"] + [f"R{j}" for j in range(K)] + ["FGT"], rows)

    P("### ADAPT 非劣性（边界 −0.5 pp；单侧检验 d > −0.5 pp；描述性，不是门槛）\n")
    P("单侧 t 检验与第二轮 H2 相同，但原始版（ADAPT）与 masked 版（ADAPT (masked)）分别报告，不合并取较大的 p，"
      "也不用\"至少 9/10 个 seed\"的符号规则。\n")
    rows = []
    for ds in HOLM_DATASETS:
        for size in SIZES:
            ours, kdd, sw = f"dual{size}_rb", f"dual{size}", f"sw{size}"
            for name, base in (("ours − KDD", kdd), ("ours − 滑窗", sw)):
                cells = []
                for k in ("ADAPT", "ADAPT_masked"):
                    ni = noninferiority(vals(S, ds, ours, k, seeds), vals(S, ds, base, k, seeds), NONINF_MARGIN)
                    if ni["n"] == 0:
                        cells.append("—")
                        continue
                    ok = "无法判断" if not np.isfinite(ni["lower"]) else ("是" if ni["lower"] > -NONINF_MARGIN else "否")
                    p1 = "nan" if not np.isfinite(ni["p"]) else format(ni["p"], ".3g")
                    cells.append(f"{pp(ni['mean'], True)}；单侧 p={p1}；下界 {pp(ni['lower'], True)}；下界 > −0.50：{ok}")
                rows.append([ds, size, name] + cells)
    table(P, ["数据集", "规模", "比较", "ADAPT", "ADAPT (masked)"], rows)

    P("### RET(ours) − RET(KDD)：Holm 校正（家族 = 4 数据集 × 2 规模 = 8 个双侧配对 t 检验；渐变用 (a) 臂 insects_gradual）\n")
    tests = [(ds, size) for ds in HOLM_DATASETS for size in SIZES]
    prs = [paired(vals(S, ds, f"dual{size}_rb", "RET", seeds), vals(S, ds, f"dual{size}", "RET", seeds))
           for ds, size in tests]
    adj = holm([pr["p"] for pr in prs])
    rows = [[ds, size, pcell(pr), "nan" if not np.isfinite(pr["p"]) else f"{pr['p']:.3g}",
             "nan" if not np.isfinite(a) else f"{a:.3g}"] for (ds, size), pr, a in zip(tests, prs, adj)]
    table(P, ["数据集", "规模", "RET 差", "p（未校正）", "p（Holm）"], rows)
    P("没有出现 nan 才是完整的 8 个检验；nan 的检验在 Holm 排序里按 p=1 处理（家族大小仍为 8）。\n")


def get_replay(cache, probs, runs, ds, m, s):
    key = (ds, m, s)
    if key in cache:
        return cache[key]
    d = runs.get(key)
    rep = None
    if d is not None:
        try:
            rep = replay_run(d, m, verify=True)
        except AssertionError as e:
            probs.append(f"重放校验失败 {key}：{e}")
    cache[key] = rep
    return rep


def section_overlap(P, runs, S, seeds_replay, cache, probs):
    P("## c. KDD 双记忆与同规模滑窗的重合（只用标签重放，不跑模型）\n")
    P("每批更新之后，记忆行集合中落在\"最近 N 个已揭示行\"（N = 记忆预算 = 同规模滑窗）里的比例；\"记忆已满\"指已揭示行数 ≥ N。"
      "重放用真实的 run_stream 与记忆类，每个检查点的 digest 都与 npz 比对；通过/失败计数见 i 段之后的\"重放校验合计\"，"
      "失败的运行不进入表格。"
      f"seed：{seeds_replay}，多个 seed 时均值取平均、最小值取最小。\n")
    rows = []
    for ds in HOLM_DATASETS:
        for size in SIZES:
            cells = []
            for m in (f"dual{size}", f"dual{size}_rb"):
                sm = [overlap_summary(r) for r in (get_replay(cache, probs, runs, ds, m, s) for s in seeds_replay) if r]
                if not sm:
                    cells += ["—", "—", "—"]
                else:
                    cells += [pp(np.mean([x["mean_all"] for x in sm])), pp(np.mean([x["mean_full"] for x in sm])),
                              pp(np.min([x["min_full"] for x in sm]))]
            rows.append([ds, size] + cells)
    table(P, ["数据集", "规模", "KDD 均值（全部批）", "KDD 均值（记忆已满）", "KDD 最小值（已满）",
              "ours 均值（全部批）", "ours 均值（已满）", "ours 最小值（已满）"], rows)


def section_pool(P, runs, seeds_replay, cache, probs):
    P("## d. 机制回放：长期池里各评估阶段的行数（dual* 与 dual*_rb；批次按 boundaries ∪ mem_boundaries 切，"
      "official 在 mem_boundaries 处、ADWIN 在 npz 的 closures 处 close_regime）\n")
    P("下面各表只取一个 seed 作示例；dual*_rb 的配额只依赖标签和阶段号，是确定性的，各 seed 的比例相同（见 i、e 段的多 seed 汇总）。"
      "\"R0 行\"指评估阶段 R0 的行（渐变段里含过渡区的行）。\n")
    s0 = seeds_replay[0] if seeds_replay else 0
    for ds in DATASET_METHODS:
        for size in SIZES:
            ms = [m for m in methods_of(ds, size) if parse_method(m)[0] in ("kdd", "ours")]
            reps = {m: get_replay(cache, probs, runs, ds, m, s0) for m in ms}
            reps = {m: r for m, r in reps.items() if r}
            if not reps:
                continue
            d0 = runs[(ds, next(iter(reps)), s0)]
            n_reg = next(iter(reps.values()))["n_reg"]
            n_end = len(d0["y"]) if int(d0["max_stream_rows"][0]) < 0 else min(len(d0["y"]), int(d0["max_stream_rows"][0]))
            times = [int(b) for b in d0["boundaries"] if 0 < b < n_end] + [n_end]
            P(f"### {ds}，{size} 行（seed {s0}）\n")
            rows = []
            for t in times:
                for m, r in reps.items():
                    i = np.flatnonzero(r["ts"] == t)
                    if not len(i):
                        continue
                    i = int(i[0])
                    rows.append([("终点 " if t == n_end else "边界 ") + str(t), m]
                                + [int(r["comp"][j, i]) for j in range(n_reg)] + [int(r["long_len"][i])])
            table(P, ["时刻", "方法"] + [f"R{j}" for j in range(n_reg)] + ["长期池总行数"], rows)


def section_replay_summary(P, runs, cache):
    """放在所有重放之后（d、e 段都算进来）。"""
    ok = sum(1 for r in cache.values() if r)
    bad = sum(1 for k, r in cache.items() if r is None and k in runs)
    P("### 重放校验合计\n")
    P(f"重放校验（digests / closures / 检查点时刻与 npz 比对）：通过 {ok} 个，失败 {bad} 个"
      "（失败的见报告开头，已从 c/d/e/i 的重放表中排除）。\n")


def section_gradual(P, runs, S, seeds, cache, probs):
    P("## e. 渐变段（insects_gradual 切片；评估阶段 R0 = 35°C 与过渡区，R1 = 纯 23°C）\n")
    if not any(k[0] == "insects_gradual" for k in runs):
        P("（没有 insects_gradual 的结果）\n")
        return
    P("三个记忆边界臂：(a) 官方变点 58,158（`insects_gradual`）；(b) 过渡区起点 55,158（`insects_gradual_start`，"
      "用了事后才知道的过渡区起点的参照臂，不可部署，它不一定比 (a) 好）；(c) ADWIN 报警（`*_rb_adwin`）。"
      "评估阶段与留出集三臂相同。过渡区 = npz 的 zone（流内坐标）；T1–T3 是过渡区三等分各段，ZONE_500 是过渡区开始后 500 行。"
      "**注意**：这里和下面各段的\"R0 行\"都指评估阶段 R0（35°C + 过渡区）的行，含过渡区的行；"
      "但 R0 的留出集只含纯 35°C 的行（过渡区的行不进留出集）。\n")
    shares = [class_share(runs[("insects_gradual", "sw200", s_)]) for s_ in seeds
              if runs.get(("insects_gradual", "sw200", s_)) is not None]
    if shares:
        P(f"整条流各类所占比例（%）：每个 seed 的最小类 {lohi([100 * x.min() for x in shares])}，"
          f"最大类 {lohi([100 * x.max() for x in shares])}（这是类别不平衡的切片）；seed {seeds[0]}："
          f"{', '.join(f'{100 * v:.1f}' for v in shares[0])}。\n")
    keys = ["RET", "ADAPT", "ADAPT_masked", "ZONE", "ZONE_T1", "ZONE_T2", "ZONE_T3", "ZONE_500"]
    z0 = None
    for size in SIZES:
        arms = [("滑窗", "insects_gradual", f"sw{size}"), ("KDD 双记忆", "insects_gradual", f"dual{size}"),
                ("ours (a) 官方变点", "insects_gradual", f"dual{size}_rb"),
                ("ours (b) 过渡区起点", "insects_gradual_start", f"dual{size}_rb"),
                ("ours (c) ADWIN", "insects_gradual", f"dual{size}_rb_adwin")]
        P(f"### {size} 行\n")
        rows = [[name, f"{ds}/{m}"] + [cell(mean_sd_ci(vals(S, ds, m, k, seeds)), len(seeds)) for k in keys] for name, ds, m in arms]
        table(P, ["臂", "来源"] + keys, rows)
        P("各臂相对 KDD / 滑窗的配对差（pp）。只有 (a) 臂的 RET / ADAPT 属于预注册的配对检验；ZONE / ZONE_500 的 p "
          "和 (b)(c) 臂的检验是描述性的，不属于预注册检验，也不进 Holm。(b) − (a)、(c) − (a) 两行是看过第三轮结果之后补的描述性比较"
          "（(a) − (c) 的符号与之相反）：\n")
        rows = []
        for name, ds, m in arms[2:]:
            for bname, bm in (("KDD", f"dual{size}"), ("滑窗", f"sw{size}")):
                rows.append([f"{name} − {bname}"] + [pcell(paired(vals(S, ds, m, k, seeds),
                                                                   vals(S, "insects_gradual", bm, k, seeds)))
                                                     for k in ("RET", "ADAPT", "ZONE", "ZONE_500")])
        for name, ds, m in arms[3:]:
            rows.append([f"{name} − (a)"] + [pcell(paired(vals(S, ds, m, k, seeds),
                                                          vals(S, "insects_gradual", f"dual{size}_rb", k, seeds)))
                                             for k in ("RET", "ADAPT", "ZONE", "ZONE_500")])
        table(P, ["比较", "RET", "ADAPT", "ZONE", "ZONE_500"], rows)

        # 与 KDD 的预测第一次不同的位置（结构性相同的检验）
        rows = []
        for name, ds, m in arms[2:]:
            fds, zb = [], None
            for s_ in seeds:
                da, db = runs.get((ds, m, s_)), runs.get(("insects_gradual", f"dual{size}", s_))
                if da is None or db is None:
                    continue
                fd = first_diff_row(da, db)
                fds.append(np.nan if fd is None else fd)
                z = np.asarray(da["zone"]).reshape(-1, 2)[0]
                z0, zb = int(z[0]), int(z[1])
            if fds:
                n_in = int(sum(1 for f in fds if np.isfinite(f) and f < zb))
                n_500 = int(sum(1 for f in fds if np.isfinite(f) and f < z0 + 500))
                rows.append([name, lohi(fds, 0), f"{n_500}/{len(fds)}", f"{n_in}/{len(fds)}"])
        if rows:
            P(f"与同规模 KDD 的预测第一次不同的流内行号（各 seed 的范围；过渡区 = [{z0}, {zb})，ZONE_500 = [{z0}, {z0 + 500})）。"
              "(a) 臂的记忆在官方变点之前不划分阶段，所以在过渡区里与 KDD 逐行相同，(a) − KDD 的 ZONE 恒为 0、p=nan，"
              "这是结构决定的，不是实验发现，也不能读成\"0/10 个 seed 为正 = ours 输了\"；同理 (b)(c) 臂在第一次关闭阶段之前也与 KDD 相同：\n")
            table(P, ["臂", "第一处不同的行号", "在 ZONE_500 内就不同的 seed 数", "在过渡区内就不同的 seed 数"], rows)

        m = f"dual{size}_rb_adwin"
        rows = []
        for s in seeds:
            d = runs.get(("insects_gradual", m, s))
            if d is None:
                rows.append([s, "缺失", "", "", "", ""])
                continue
            rep = get_replay(cache, probs, runs, "insects_gradual", m, s)
            a = adwin_alarm_info(d, rep)
            sg = lambda v: "—" if v is None else f"{v:+d}"      # noqa: E731
            rows.append([s, a["alarms"] if a["alarms"] else "无报警", sg(a["off_zone"]), sg(a["off_cp"]), a["further"],
                         pp(a["r0_share"])])
        P(f"ADWIN 每个 seed（{m}）。\"报警位置\"= 新阶段第一行（报警行 + 1）；偏移 = 首次报警 − 参照点（负 = 在其之前），"
          "首次报警取流中第一次，不管落在哪里；\"又报警\"= 首次之后的报警次数；R0 占比 = 最终长期记忆里评估阶段 0 的行所占 %。\n")
        table(P, ["seed", "报警位置", "相对过渡区起点", "相对文档变点", "又报警", "最终长期记忆 R0 占比 (%)"], rows)
        infos = [adwin_alarm_info(runs[("insects_gradual", m, s)]) for s in seeds if ("insects_gradual", m, s) in runs]
        firsts = [a["first"] for a in infos if a["first"] is not None]
        if firsts:
            z0_ = int(np.asarray(runs[("insects_gradual", m, seeds[0])]["zone"]).reshape(-1, 2)[0, 0])
            cp_ = int(runs[("insects_gradual", m, seeds[0])]["boundaries"][0])
            f_zone = np.array(firsts) - z0_
            f_cp = np.array(firsts) - cp_
            sec = np.array([a["alarms"][1] - cp_ for a in infos if len(a["alarms"]) > 1])
            med = lambda v: f"{np.median(v):+g}"      # noqa: E731
            txt = (f"汇总（{len(firsts)} 个有报警的 seed；偏移单位：行；中位数不取整）：首次报警相对过渡区起点 "
                   f"{int(f_zone.min()):+d} 到 {int(f_zone.max()):+d}（中位数 {med(f_zone)}）；"
                   f"首次报警相对文档变点 {int(f_cp.min()):+d} 到 {int(f_cp.max()):+d}（中位数 {med(f_cp)}）")
            if len(sec):
                txt += (f"；第二次报警相对文档变点 {int(sec.min()):+d} 到 {int(sec.max()):+d}（中位数 {med(sec)}，"
                        f"{len(sec)} 个 seed 有第二次报警）")
            P(txt + "。\n")

        # 最终长期池构成
        rows = []
        for name, ds, m_ in arms[1:]:
            comp_ = []
            for s_ in seeds:
                d, rep = runs.get((ds, m_, s_)), get_replay(cache, probs, runs, ds, m_, s_)
                if d is not None and rep:
                    comp_.append(final_pool_by_position(rep, d))
            if not comp_:
                continue
            cell_ = lambda k: rng([c[k] for c in comp_], 1)      # noqa: E731
            rows.append([name, f"{ds}/{m_}", comp_[0]["total"], cell_("pure_first"), cell_("t1"), cell_("t2"), cell_("t3"),
                         cell_("pure_last"), lohi([c["classes_first"] for c in comp_], 0), len(comp_)])
        P("流末长期池的构成（按流内位置分类，行数；多个 seed 时写 均值（最小–最大），全相同只写一个数）。"
          "\"纯 35°C\" = 过渡区之前的行，T1–T3 = 过渡区三等分，\"纯 23°C\" = 过渡区之后的行。"
          "**R0 的留出集是纯 35°C**；(a) 臂的\"R0 行\"若全在 T1–T3，就是说它在没有保留任何纯 35°C 行的情况下对纯 35°C 留出集有增益：\n")
        table(P, ["臂", "来源", "池大小", "纯 35°C", "T1", "T2", "T3", "纯 23°C", "纯 35°C 行覆盖的类数", "seed 数"], rows)

        # KDD 长期池里 R0 最后出现的时刻
        for name, ds, m_ in arms[1:2]:
            last = []
            for s_ in seeds:
                rep = get_replay(cache, probs, runs, ds, m_, s_)
                if rep:
                    last.append(last_present_time(rep, 0))
            last = [x for x in last if x is not None]
            if last:
                P(f"KDD 双记忆（{size} 行）长期池里最后一次含 R0 的行的时刻（已揭示行数，各 seed 的范围）：{lohi(last, 0)}"
                  "（文档变点 12,750 对应的流内位置）。\n")

        # R0 留出集上逐类召回
        rows = []
        for name, ds, m_ in arms:
            rc = np.array([holdout_class_recall(runs[(ds, m_, s_)], 0) for s_ in seeds
                           if runs.get((ds, m_, s_)) is not None])
            if len(rc):
                rows.append([name] + [("—" if not np.isfinite(v) else f"{100 * v:.1f}") for v in np.nanmean(rc, axis=0)])
        if rows:
            ncls = len(rows[0]) - 1
            P("R0 留出集上逐类召回（%，R0 结束之后各检查点的平均，再对 seed 平均；R0 的留出集只含纯 35°C 的行，"
              "留出集里没有的类写 —）：\n")
            table(P, ["臂"] + [f"类 {q}" for q in range(ncls)], rows)


def section_stage(P, runs, S, seeds, args):
    P("## f. 阶段差异与阶段增益（探索性，看过 Gas 的 CPU 结果之后定义）\n")
    P("阶段 j 的\"差异\" = 逻辑回归区分阶段 j 的标准化流特征与之后所有阶段的行，5 折分层交叉验证 ROC-AUC 的均值"
      "（prepare(0, dataset)，最后一个阶段除外）。增益 = RET_Rj(ours) − RET_Rj(KDD)（同规模、seed 均值）。"
      "此度量是在看过 gas 的各阶段保留之后选定的，所以只作探索性描述，不是检验。\n")
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from run_round2 import prepare
    except Exception as e:      # noqa: BLE001
        P(f"（无法导入 prepare：{e}；跳过）\n")
        return
    aucs, pts = {}, []
    for ds in HOLM_DATASETS:
        if not any(k[0] == ds for k in runs):
            continue
        try:
            dd = prepare(0, ds)
            aucs[ds] = stage_auc(dd["X"], dd["regime"])
        except Exception as e:      # noqa: BLE001
            P(f"（{ds}：无法载入数据流，{type(e).__name__}: {e}；跳过）\n")
            continue
        for size in SIZES:
            for j, auc in aucs[ds].items():
                o = vals(S, ds, f"dual{size}_rb", f"RET_R{j}", seeds)
                k = vals(S, ds, f"dual{size}", f"RET_R{j}", seeds)
                g = np.nanmean(o - k) if np.isfinite(o - k).any() else NAN
                pts.append((ds, size, j, auc, g))
    if not pts:
        return
    table(P, ["数据集", "规模", "阶段 j", "AUC（j vs 之后）", "增益 (pp)"],
          [[ds, size, f"R{j}", f"{auc:.3f}", pp(g, True)] for ds, size, j, auc, g in pts])
    rows = []
    for ds in list(aucs) + ["全部数据集"]:
        for size in list(SIZES) + ["两种规模合并"]:
            sel = [p for p in pts if (ds == "全部数据集" or p[0] == ds) and (size == "两种规模合并" or p[1] == size)]
            rho, p, n = spearman([x[3] for x in sel], [x[4] for x in sel])
            rows.append([ds, size, n, "nan" if not np.isfinite(rho) else f"{rho:+.2f}",
                         "nan" if not np.isfinite(p) else f"{p:.3g}"])
    P("Spearman 相关（AUC 与增益；n 是点数，n<3 为 nan；合并行里同一数据集的点不独立，p 仅供参考）：\n")
    table(P, ["数据集", "规模", "n", "rho", "p"], rows)
    P("**读法（探索性）：**\"全部数据集\"行的 rho 主要来自数据集之间的差异，数据集内部（gas 的 AUC 已到上限，rho 为负）"
      "并不支持\"阶段越不同、增益越大\"；两种规模合并的行把同一个 AUC 算了两次。这些行的 p 不能当作证据，"
      "不要用\"预测\"\"解释\"这类词。\n")


def _ref_method(runs, ds, seeds):
    return next((m for m in ("sw200", "sw1000", "dual200") if any((ds, m, s) in runs for s in seeds)), None)


def section_baselines(P, runs, S, seeds):
    P("## g. 不用 TabPFN 的基线（流上已预测行的平衡准确率 %，各 seed 的流不同，先逐 seed 算再平均；均值 ± SD）\n")
    P("预注册写的是 \"No-Change (previous label)\" 和 \"majority class of the last 200 rows\"，实现时按行逐个计算"
      "（上一行的标签；被预测行之前 200 行的多数类）。但协议是每批 50 行预测完才一起揭示标签，所以**预注册口径用到了批内尚未揭示的标签，"
      "偏乐观**，TabPFN 并没有这些信息。因此并列加两个按批揭示的参照（看过第三轮结果之后补的描述性指标，不是预注册指标，与 TabPFN 信息相同）："
      "No-Change_batch = 当前批开始前最后一个已揭示的标签；Majority_batch = 当前批开始前 200 个已揭示行的多数类"
      "（平票取类别编号最小者）。批次按 run_stream 的规则重建（从第 start 行起每 stride 行一批，不跨越阶段边界与记忆边界）。\n")
    rows, rows2 = [], []
    for ds in DATASET_METHODS:
        if ds == "insects_gradual_start":
            continue
        ref = _ref_method(runs, ds, seeds)
        if ref is None:
            continue
        bl = [baselines_full(runs[(ds, ref, s)]) for s in seeds if (ds, ref, s) in runs]
        col = lambda k: np.array([b[k] for b in bl], float)      # noqa: E731
        ms = lambda k: (lambda st: "—" if st["n"] == 0 else f"{pp(st['mean'])} ± {pp(st['sd'])}")(mean_sd_ci(col(k)))      # noqa: E731
        adapt_means = {m: np.nanmean(vals(S, ds, m, "ADAPT", seeds)) for m in DATASET_METHODS[ds]
                       if any((ds, m, s) in runs for s in seeds)}
        worst_m = min(adapt_means, key=adapt_means.get)
        best_batch = max(np.mean(col("nc_batch")), np.mean(col("maj_batch")))
        rows.append([ds, len(bl), pp(col("nc_row").mean()), pp(col("maj_row").mean()), ms("nc_batch"), ms("maj_batch"),
                     f"{pp(adapt_means[worst_m])}（{worst_m}）", pp(adapt_means[worst_m] - best_batch, True)])
        rows2.append([ds, pp(col("nc_row_plain").mean()), pp(col("nc_batch_plain").mean()), int(bl[0]["n_pred"]),
                      int(bl[0]["n_batches"])])
    table(P, ["数据集", "seed 数", "No-Change（预注册，偏乐观）", "最近 200 行多数类（预注册，偏乐观）", "No-Change_batch",
              "Majority_batch", "TabPFN 方法中最低的 ADAPT（均值）", "它与较好的按批基线之差 (pp)"], rows)
    P("普通（未平衡）准确率：逐行 No-Change 的普通准确率 = 已预测行里\"标签与上一行相同\"的比例（标签自相关的度量；"
      "Electricity 约 85 % 说明相邻标签高度自相关）；按批揭示后降到右边一列。n_pred / 批数取自第一个 seed。\n")
    table(P, ["数据集", "No-Change 逐行（普通准确率 %）", "No-Change_batch（普通准确率 %）", "已预测行数 (n_pred)", "批数"], rows2)
    P("**读法：**不能写\"No-Change 胜过 TabPFN\"。逐行口径的 No-Change 在 Electricity 上的高值来自相邻行标签的自相关加上批内泄漏；"
      "按批揭示的口径与 TabPFN 信息相同，在所有数据集上 TabPFN 都远高于平凡基线（上表最后一列）。\n")


def section_mask(P, runs, seeds):
    P("## h. ADAPT (masked) 的掩码去掉了多少行（单类连续段 ≥ 30 行；描述性）\n")
    P("掩码为 Insects 设计；\"占已预测行\"是 ADAPT (masked) 实际的分母，\"占整条流\"含前 start 行没有预测的行。"
      "各 seed 的流不同，先逐 seed 算再汇总。\"两种分母之差\" = 占已预测行 − 占整条流，逐 seed 的范围（pp）。"
      "掩码比例为 0 的数据集，其 ADAPT 与 ADAPT (masked) 两列完全相同。\n")
    rows = []
    for ds in DATASET_METHODS:
        if ds == "insects_gradual_start":
            continue
        ref = _ref_method(runs, ds, seeds)
        if ref is None:
            continue
        ds_runs = [runs[(ds, ref, s)] for s in seeds if (ds, ref, s) in runs]
        a_ = np.array([mask_removed_share(d) for d in ds_runs])
        b_ = np.array([mask_removed_share(d, "stream") for d in ds_runs])
        s0 = next((i for i, s in enumerate(s_ for s_ in seeds if (ds, ref, s_) in runs) if s == seeds[0]), 0)
        dif = 100 * (a_ - b_)
        rows.append([ds, len(a_), f"{100 * a_.mean():.2f}", f"{100 * a_.min():.2f}–{100 * a_.max():.2f}",
                     f"{100 * a_[s0]:.2f}", f"{100 * b_.mean():.2f}", f"{100 * b_.min():.2f}–{100 * b_.max():.2f}",
                     f"{100 * b_[s0]:.2f}", f"{dif.min():+.3f} 到 {dif.max():+.3f}"])
    table(P, ["数据集", "seed 数", "占已预测行 均值 (%)", "范围", f"seed {seeds[0]}", "占整条流 均值 (%)", "范围",
              f"seed {seeds[0]}（整条流）", "两种分母之差 (pp)"], rows)


def _per_regime_stat(items, key):
    """items：每个 seed 一个 regime_leftover 结果（j -> dict）；返回按阶段 j 对 seed 取均值的列表。"""
    js = sorted({j for it in items for j in it})
    return [np.mean([it[j][key] for it in items if j in it]) for j in js]


def section_leftover(P, runs, seeds_replay, cache, probs):
    P("## i. 阶段结束之后记忆里还剩什么（只用标签重放，不跑模型；10 个 seed）\n")
    P("\"旧阶段\" = 已结束的评估阶段（boundaries 之前的阶段）。表中\"逐阶段\"的数是各阶段在 seed 间的均值；"
      "渐变段的 ours 是 (a) 臂，\"R0 行\"含过渡区的行。\n")
    combos = [(ds, size) for ds in HOLM_DATASETS for size in SIZES if any(k[0] == ds for k in runs)]

    def collect(ds, m):
        out = []
        for s in seeds_replay:
            d, rep = runs.get((ds, m, s)), get_replay(cache, probs, runs, ds, m, s)
            if d is not None and rep:
                out.append((d, rep))
        return out

    # 1. 少于 5 行、刚结束时的占比、平均行数
    rows_a, rows_b, rows_c = [], [], []
    for ds, size in combos:
        r5, rs, rc = [ds, size], [ds, size], [ds, size]
        for m in (f"dual{size}", f"dual{size}_rb"):
            items = [regime_leftover(rep, d) for d, rep in collect(ds, m)]
            if not items:
                r5 += ["—"]; rs += ["—"]; rc += ["—", "—"]
                continue
            lt5 = _per_regime_stat(items, "lt5")
            r5.append(f"{lohi([100 * v for v in lt5])}（逐阶段：{', '.join(f'{100 * v:.1f}' for v in lt5)}）")
            rs.append(f"{100 * np.nanmean([it[j]['share_end'] for it in items for j in it]):.1f}")
            avg = _per_regime_stat(items, "avg_long")
            rc += [lohi(avg), f"{min(it[j]['min_long'] for it in items for j in it)}"]
        rows_a.append(r5); rows_b.append(rs); rows_c.append(rc)
    P("**阶段结束之后，整个记忆（长期加短期）里含该阶段的行不足 5 行的比例**（占阶段结束之后各次更新的 %，范围 = 各旧阶段的最小–最大）：\n")
    table(P, ["数据集", "规模", "KDD", "ours"], rows_a)
    P("**阶段刚结束那次更新时，长期池里属于刚结束阶段的行所占比例**（%，对所有旧阶段与 seed 取平均）：\n")
    table(P, ["数据集", "规模", "KDD", "ours"], rows_b)
    P("**阶段结束之后，长期池里该阶段的行数**（对结束后各次更新取平均，范围 = 各旧阶段的最小–最大；最少 = 所有 seed、所有更新里的最小值）：\n")
    table(P, ["数据集", "规模", "KDD 平均行数", "KDD 最少", "ours 平均行数", "ours 最少"], rows_c)

    # 2. 流末覆盖
    rows = []
    for ds, size in combos:
        for role, m in (("KDD", f"dual{size}"), ("ours", f"dual{size}_rb")):
            cov = [class_coverage(rep, d) for d, rep in collect(ds, m)]
            if not cov:
                continue
            nrows = [c[j]["n_rows"] for c in cov for j in c]
            percls = [v for c in cov for j in c for v in c[j]["per_class"]]
            cov_n = sum(c[j]["covered"] for c in cov for j in c)
            tot_n = sum(c[j]["n_classes"] for c in cov for j in c)
            full = sum(1 for c in cov for j in c if c[j]["covered"] == c[j]["n_classes"])
            n_old = sum(len(c) for c in cov)
            rows.append([ds, size, role, lohi(nrows, 0), f"{np.mean(percls):.1f}", lohi(percls, 0), f"{cov_n}/{tot_n}",
                         f"{full}/{n_old}"])
    P("**流末长期池对旧阶段的类别覆盖**（每个旧阶段、每个 seed 一个点；\"留出集的类\"= 该阶段留出集里出现的类）：\n")
    table(P, ["数据集", "规模", "方法", "每个旧阶段在池里的行数（最小–最大）", "每个留出集的类平均行数", "每个留出集的类的行数（最小–最大）",
              "被覆盖的 (阶段, 类, seed) 数/总数", "类全被覆盖的 (阶段, seed) 数/总数"], rows)

    # 3. 行年龄
    rows = []
    for ds, size in combos:
        r = [ds, size]
        for m in (f"dual{size}", f"dual{size}_rb"):
            ages = [age_summary(rep) for _, rep in collect(ds, m)]
            ages = [a for a in ages if np.isfinite(a)]
            if not ages:
                r.append("—")
                continue
            r.append(f"{np.mean(ages):.0f}" + (f"（约 {np.mean(ages) / 48:.1f} 天）" if ds == "electricity" else ""))
        rows.append(r)
    P("**长期池里行的年龄的中位数**（年龄 = 已揭示行数 − 该行的流内下标；记忆已满之后各次更新的中位数取平均，再对 seed 平均，单位：行；"
      "Electricity 每天 48 行，括号里是天数）：\n")
    table(P, ["数据集", "规模", "KDD", "ours"], rows)


def section_stream_regime(P, runs, seeds):
    P("## j. 各阶段流上的平衡准确率差（ours − KDD，pp；描述性）\n")
    P("阶段 j 的流上准确率 = 流里评估阶段 j 的已预测行的平衡准确率。括号里是差值为正的 seed 数；R0 在第一次关闭阶段之前两者相同，"
      "所以差为 0。用来看 ADAPT 的差是集中在阶段切换之后，还是分散在整条流上。\n")
    for ds in HOLM_DATASETS:
        for size in SIZES:
            ours, kdd = f"dual{size}_rb", f"dual{size}"
            a_ = {s: stream_acc_per_regime(runs[(ds, ours, s)]) for s in seeds if (ds, ours, s) in runs}
            b_ = {s: stream_acc_per_regime(runs[(ds, kdd, s)]) for s in seeds if (ds, kdd, s) in runs}
            ss = sorted(set(a_) & set(b_))
            if not ss:
                continue
            n_reg = len(a_[ss[0]])
            cells, means = [], []
            for j in range(n_reg):
                pr = paired([a_[s][j] for s in ss], [b_[s][j] for s in ss])
                means.append(pr["mean"])
                cells.append(f"{pp(pr['mean'], True)} ({pr['npos']}/{pr['n']})")
            lo_hi = "—" if n_reg < 2 else (pp(means[1], True) if n_reg == 2
                                           else f"{pp(min(means[1:]), True)} 到 {pp(max(means[1:]), True)}")
            P(f"### {ds}，{size} 行（R1 起各阶段差的范围：{lo_hi}）\n")
            table(P, [f"R{j}" for j in range(n_reg)], [cells])


def write_csv(path, runs, S):
    n_reg = max([len(d["boundaries"]) for d in runs.values()] + [1])
    keys = (["ADAPT", "ADAPT_masked", "RET", "FGT"] + [f"RET_R{j}" for j in range(n_reg)]
            + [f"FGT_R{j}" for j in range(n_reg)] + ["ZONE", "ZONE_T1", "ZONE_T2", "ZONE_T3", "ZONE_500"])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write("dataset,method,seed," + ",".join(keys) + "\n")
        for (ds, m, s) in sorted(runs):
            row = S[(ds, m, s)]
            fh.write(f"{ds},{m},{s}," + ",".join(f"{row.get(k, NAN) * 100:.6f}" for k in keys) + "\n")


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results/round3", help="第三轮 npz 目录")
    ap.add_argument("--round2_dir", default="results/round2",
                    help="第二轮 npz 目录（提供 insects 的 sw200/sw1000/dual1000/dual1000_rb）")
    ap.add_argument("--n_estimators", type=int, default=4)
    ap.add_argument("--tag", default="", help="第三轮文件名后缀（截断自测用 _smoke）")
    ap.add_argument("--seeds", default="0-9", help="预期的 seed（用来报告缺失），如 0-9")
    ap.add_argument("--quick", action="store_true", help="seed 0 快检，打印中文判断后退出")
    ap.add_argument("--quick_seed", type=int, default=0)
    ap.add_argument("--replay_seeds", default="0-9", help="c/d 段重放的 seed（ADWIN 臂总是重放全部 seed）")
    ap.add_argument("--out", default="results/round3_analysis.md")
    ap.add_argument("--csv", default="results/round3_per_seed.csv")
    args = ap.parse_args()

    runs, probs, runs3_keys = load_all(args)
    if args.quick:
        return quick(args, runs, runs3_keys, probs)
    if not runs3_keys:
        raise SystemExit("\n".join(probs) or f"{args.dir} 里没有第三轮的 npz")
    trunc = sorted({(k[0], k[1], k[2]) for k, d in runs.items() if k in runs3_keys and "max_stream_rows" in d
                    and int(d["max_stream_rows"][0]) >= 0})
    if trunc and "_smoke" not in args.tag:
        probs.append(f"注意：{len(trunc)} 个第三轮文件是截断运行（max_stream_rows ≥ 0，如 {trunc[0]}），不是正式结果")

    seeds = parse_seeds(args.seeds)
    probs += check_consistency(runs)
    S = {k: run_metrics(d) for k, d in runs.items()}
    lines = []
    P = lambda *a: lines.append(" ".join(str(x) for x in a))      # noqa: E731

    P(f"# 第三轮分析（n_estimators={args.n_estimators}{'，tag ' + repr(args.tag) if args.tag else ''}）\n")
    P("按 results/phase56_prereg.md 的 Amendment 3 及其后的 Note 实现；不调用 TabPFN；**第三轮没有成功门槛**，"
      "本报告描述各数据集上阶段均衡是否、何时有帮助。数字单位：准确率为 %，差值为 pp。\n")
    present = sorted({(k[0], k[1]) for k in runs})
    P(f"载入 {len(runs)} 个运行，{len(present)} 个 (数据集, 方法)。第二轮来源（insects）：{sum(1 for k in runs if k not in runs3_keys)} 个。\n")
    P("代码版本（npz 里的 commit 字段；同一数据集内的比较若跨两个 commit，这里看得出来）：\n")
    ver = {}
    for (ds_, m_, s_), d_ in sorted(runs.items()):
        ver.setdefault((ds_, str(d_["commit"][0]), str(d_["device"][0]), "第二轮" if (ds_, m_, s_) not in runs3_keys else "第三轮"),
                       []).append(m_)
    table(P, ["数据集", "commit", "device", "来源", "方法（个数 × seed）"],
          [[ds_, c_, dv_, src_, ", ".join(f"{m_}×{ms.count(m_)}" for m_ in sorted(set(ms)))]
           for (ds_, c_, dv_, src_), ms in ver.items()])
    miss = find_missing(runs, seeds)
    P("### 缺失的 (数据集, 方法, seed)\n")
    if not miss:
        P(f"无（预期 seed {seeds}）。\n")
    else:
        for (ds, m), ms in sorted(miss.items()):
            P(f"- {ds}/{m}: 缺 seed {ms}")
        P("")
    cache = {}
    # 重放放在前面，这样一致性与校验问题能写进文件开头
    seeds_replay = [s for s in parse_seeds(args.replay_seeds) if any(k[2] == s for k in runs)] or \
        sorted({k[2] for k in runs})[:1]
    body = []
    Pb = lambda *a: body.append(" ".join(str(x) for x in a))      # noqa: E731
    section_reliability(Pb, runs, S, seeds)
    section_paired(Pb, runs, S, seeds)
    section_overlap(Pb, runs, S, seeds_replay, cache, probs)
    section_pool(Pb, runs, seeds_replay, cache, probs)
    section_gradual(Pb, runs, S, seeds, cache, probs)
    section_stage(Pb, runs, S, seeds, args)
    section_baselines(Pb, runs, S, seeds)
    section_mask(Pb, runs, seeds)
    section_leftover(Pb, runs, seeds_replay, cache, probs)
    section_replay_summary(Pb, runs, cache)
    section_stream_regime(Pb, runs, seeds)
    P("### 一致性与校验问题\n")
    if not probs:
        P("无（y / hold_idx / stream_idx / boundaries 在同一 (数据集, seed) 内相同；leak_count = 0；上下文不超预算；"
          "完整运行没有 -1；重放 digest 与 npz 一致）。\n")
    else:
        for p in probs:
            P(f"- {p}")
        P("")
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
