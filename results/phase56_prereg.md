# Round 2 pre-registration: context memory for adaptation vs forgetting (frozen TabPFN, Insects)

Committed **before** the Mac pilot and before any round-2 result exists (2026-10-03).
Any later change is added at the bottom as a dated amendment with a mechanistic reason, never because of a score.

## Question

A frozen TabPFN that predicts from a sliding window of recent rows adapts quickly but drops everything older than the window.
Does a **regime archive** — a few class-balanced representative rows stored per finished regime and recalled only when a query resembles that regime — keep its adaptation while forgetting much less?

Motivation from a pilot cross-regime transfer matrix (6 classes, TabPFN n_estimators = 1, 3 seeds, 200-row contexts): a context from another regime often drops balanced accuracy from 46–68 % to 7–29 %, and two regimes recur (R5 resembles R0, R4 resembles R2).

## Data

- Insects `abrupt_balanced` (Souza et al. 2020, Table 2): 52,848 rows, 33 features, 6 classes {2,3,4,5,11,12} mapped to 0–5. Native labels, no grouping.
- Official change points 14352 / 19500 / 33240 / 38682 / 39510 define regimes R0–R5. Per the paper (§5) R0 is 30 °C, R1 20 °C, R2 about 35 °C; later temperatures are not stated, similarities between regimes are our measurement.
- **Holdouts** (per seed): from each regime, 50 rows per class (R4: 20 per class) sampled uniformly at random across the **whole** regime, removed from the stream, never in any context. Drift points are remapped to stream coordinates.
- Features are z-scored with the mean and standard deviation of the first 200 stream rows.
- Known data property: each official change point is immediately preceded by a single-class run (1754 / 74 / 97 / 537 / 34 rows) and the stream ends with one (822 rows). Every stream metric is therefore also reported with these runs masked (rows inside identical-label runs of length ≥ 30).

## Protocol

- Batch-incremental prequential: from stream row 200, predict the next 50 rows with the context built from earlier rows, then reveal their labels and update the memory. Batches never straddle a regime boundary.
- Retention checkpoints every 500 stream rows and at the end of each regime: predict the holdouts of every regime seen so far with the method's **current** memory (the same routing rule as for stream rows). Checkpoints never change the memory.
- Model: TabPFN (tabpfn 6.4.1, v2.5 weights), frozen, `n_estimators = 4` on GPU (pilot: 1 on CPU), `random_state = seed`.
- Seeds 0–9. A seed sets the holdout sample, the archive sample and TabPFN's random state; the stream order is fixed.

## Methods (all run for all 10 seeds)

| Name | Context |
|---|---|
| `sw{100,200,300,400,600,1000,1500,2000}` | the most recent N rows (`sw200` = TabPFN alone, the main baseline) |
| `dual1000` | dual memory of Lourenço et al. (KDD 2026) with the paper's setting: M = 1000, 75 % short-term FIFO, long-term pool evicts the oldest row of the most frequent class, no age limit |
| `dual400` | the same with M = 400 |
| `cbfifo400` | the most recent ≤ 66 rows of each class |
| `arch_union` | `sw200` plus the archives of all finished regimes (25 rows per class each, class-balanced random sample of the whole regime) |
| **`arch_routed`** (ours) | `sw200` plus the archive of the finished regime a query resembles most, only if it resembles that regime more than the current window; otherwise `sw200` alone |
| `arch_routed_adwin` | as `arch_routed`, but regimes are closed by river ADWIN (δ = 0.002, cooldown 300 rows) on the method's own per-row errors instead of the official points; minimum segment 500 rows, at most 8 archives (oldest evicted) |

Router (fixed now): z-score the pool (current window + archives) with pool statistics; take the k = 10 nearest neighbours of the query (Euclidean, ties by index); score each source by its neighbour count divided by its size; route to the best archive only if its score is higher than the window's.

Baselines without TabPFN: No-Change (previous label) and the majority class of the last 200 rows.

## Metrics

- **ADAPT** — balanced accuracy over all predicted stream rows (also masked).
- **RET** — for each finished regime j < 5, balanced accuracy on its holdout averaged over all checkpoints after regime j ended; then averaged over j.
- **POST_d** — balanced accuracy over the first 500 stream rows after change point d (also masked).
- **FGT_j** — best minus final holdout accuracy of regime j.

## Hypotheses

Primary (one-sided paired t-test over the 10 seeds, Holm correction across H1–H3, α = 0.05; in addition the sign must hold in at least 9 of 10 seeds):

- **H1** RET(`arch_routed`) − RET(`sw200`) ≥ 5 pp, and the difference is significantly > 0.
- **H2** non-inferior adaptation: the lower bound of the one-sided 95 % confidence interval of ADAPT(`arch_routed`) − ADAPT(`sw200`) is above −0.5 pp, in both the unmasked and the masked version.
- **H3** RET(`arch_routed`) − RET(`dual1000`) ≥ 2 pp, and the difference is significantly > 0.

Secondary (reported descriptively):

- **H4** POST gains of `arch_routed` over `sw200` ≥ 5 pp after 19500 and 38682 (old archives match the new regime), and within ±2 pp after 14352 and 39510 (negative controls).
- **H5** no sliding window size is at least as good as `arch_routed` on both ADAPT and RET.
- **H6** RET(`arch_routed`) − RET(`cbfifo400`) ≥ 2 pp.
- **H7** ADAPT(`arch_routed`) ≥ ADAPT(`arch_union`).
- **H8** `arch_routed_adwin` keeps at least half of the H1 gain and passes H2.

If a primary hypothesis fails, the result is reported as failed, together with the matching fallback statement: above the window-size frontier (H5); retention of the KDD dual memory with about a third of its context (H3 within 2 pp); archive alone suffices and routing saves context (H7 tie); official boundaries are an upper bound and segmentation is the bottleneck (H8); otherwise descriptive results (transfer matrix, window frontier, first retention evaluation of the dual memory).

## Pilot go/no-go (Mac, seed 0, n_estimators = 1, methods `sw200`, `sw1000`, `dual1000`, `arch_routed`)

Go if: no leakage or budget violation; RET(`arch_routed`) − RET(`sw200`) ≥ 3 pp; ADAPT difference ≥ −2 pp. The pilot also serves as the reference for the GPU reproduction gate on the ROG.

## Disclosed development runs

Before this file was committed, one pipeline check ran `sw200` and `arch_routed` on the first 20,000 stream rows (seed 0, n_estimators = 1, CPU). It covered R0, R1 and the start of R2 only. RET was 28.3 vs 42.6 %, ADAPT 74.7 vs 74.0 %. No design parameter was changed after it.

## Amendments

### Amendment 1 (2026-10-03, after the Mac pilot)

**Pilot result** (seed 0, n_estimators = 1, CPU, full stream; balanced accuracy %):

| method | ADAPT | RET |
|---|---|---|
| `sw200` | 75.56 | 34.79 |
| `sw1000` | 76.71 | 39.18 |
| `dual1000` | 76.44 | 45.53 |
| `arch_routed` | 72.85 | 43.72 |

`arch_routed` failed the pre-registered go criterion (ADAPT −2.71 pp, limit −2 pp; the RET criterion passed with +8.93 pp).

**Mechanism.** The router sent 13–72 % of current-regime stream rows (by regime) to an old archive, and on those rows the prediction was worse than the window alone for almost every (regime, archive) pair, e.g. R2 queries routed to the R1 archive: 63.8 % vs 76.9 %. The window already holds the most relevant data; per-query routing adds conflicting old rows. Meanwhile `dual1000` (750 recent + 250 class-balanced old rows, always in context) kept adaptation and retained more, but its long-term pool drifts towards the most recent regimes (RET on R1: 23.3 %).

**Change.** The proposed method becomes **`dual1000_rb`**, a regime-balanced dual memory: identical to `dual1000` (M = 1000, 75 % short-term FIFO, overflow into the long-term pool) except for the long-term eviction rule. When the pool is full, it evicts the oldest row of the most frequent class **within the regime that holds the most rows in the pool** (ties: the newer regime; then the smaller class id). Regimes come from the official change points; `dual1000_rb_adwin` uses ADWIN alarms instead (same settings as `arch_routed_adwin`). Without any regime boundary the rule reduces exactly to `dual1000` (unit test).

**Hypotheses.** H1–H6 now refer to `dual1000_rb` in place of `arch_routed`. H7 becomes: ADAPT(`dual1000_rb`) − ADAPT(`dual1000`) ≥ −0.5 pp (the regime balancing must not cost adaptation). H8 refers to `dual1000_rb_adwin`. `arch_routed`, `arch_routed_adwin` and `arch_union` stay in the main run and are reported as pre-registered (expected: `arch_routed` fails H2). The `calib5` GPU gate and the S = 10 sensitivity run use `sw200` and `dual1000_rb`.

**Pilot go/no-go for the new method** (same thresholds as before, seed 0, n_estimators = 1): RET(`dual1000_rb`) − RET(`sw200`) ≥ 3 pp and ADAPT(`dual1000_rb`) − ADAPT(`sw200`) ≥ −2 pp. The comparison with `dual1000` is reported but is not a go criterion.

### Amendment 2 (2026-10-03, before the main run; prompted by an independent code review, no new results seen)

- **Holm family.** H2 is a non-inferiority test and is now part of the Holm family with H1 and H3, as the original text says ("Holm correction across H1–H3"). Its p-value is the one-sided one-sample t-test of (ADAPT difference + 0.5 pp) > 0, taking the larger p of the unmasked and masked versions; its sign rule is "difference > −0.5 pp in at least 9 of 10 seeds". The one-sided 95 % lower bound is still reported.
- **Seed count.** H1–H3 are only given a formal verdict with all 10 seeds; any earlier analysis (e.g. after `main_a` with 5 seeds) is labelled informal.
- **ADWIN details** (implementation as committed before the pilot, now stated explicitly): an alarm that would close a segment shorter than 500 rows is ignored, and the detector's own reset and 300-row cooldown still apply, so that change point is not re-detected. `dual1000_rb_adwin` has no cap on the number of segments; the 8-archive cap applies only to the archive methods.
- **Reporting.** FGT_j, the masked POST values and the two non-TabPFN baselines (No-Change, majority of the last 200 rows) are reported for every method.

## Note on version history (added 2026-10-06)

The public repository's history was restarted on 2026-10-06 so that it contains only the files needed to run the project. The original history is kept by the author and can be shown on request:

| Version | Commit (original history) | Committed | Git blob of this file |
|---|---|---|---|
| Pre-registration | `0303b71` | 2026-10-02 21:14 CEST (the header line above gives the planned date, 2026-10-03) | `30dcc72f06b86d50ffe2030b77c18257001d927a` |
| + Amendment 1 | `685e653` | 2026-10-03 09:03 CEST | `3c485d3378df6cac62bb8c8d7bf036258456a025` |
| + Amendment 2 | `3c7c19f` | 2026-10-03 10:10 CEST, before the main runs | `165a8e198f7b721fef647fae055cf60f3df3f402` |

The text above this note is unchanged since Amendment 2. Amendment 2 was written after the go/no-go pilot that Amendment 1 requires had finished (`dual1000_rb`, seed 0, n_estimators = 1: retention +12.6 pp and adaptation +0.93 pp against `sw200`, so "go") and before any main-run result; "no new results seen" in its heading refers to the main runs. Two names in it refer to the author's setup: "the ROG" is the GPU laptop used for all main runs (NVIDIA RTX 2060, 6 GB), and `calib5`, `core`, `main_a`, `main_b` and `sens` are the names of its run batches (`calib5` = the GPU reproduction gate against the Mac pilot).

## Amendment 3 (2026-10-13, round 3: more datasets; written before any full round-3 run)

**Why.** Round 2 used a single stream. Round 3 asks whether its result holds on further datasets with different drift characteristics, compares the sliding window, the KDD dual memory and our method at the same context size (200 and 1,000 rows), and analyses the reliability of the results and when regime balancing helps. Implementation: `src/data/streams.py`. Before this amendment was fixed, only truncated pipeline checks had been run on the new streams (CPU, one ensemble member, seed 0, at most the first 11,000 rows); they served to test the code and were not used to choose any setting.

**Dataset selection.** Candidates were the real streams of the USP DS repository (Souza et al. 2020) plus INSECTS variants. A dataset qualified if (1) regimes can be defined from documented or natural time boundaries, (2) every regime contains enough rows of most classes for a stratified holdout, (3) later regimes differ from earlier ones, (4) the stream is at most about 100k rows of numeric features within TabPFN's limits (at most 10 classes), and (5) it has no unfixable artefacts (leaking columns, label-sorted order, long single-class runs). Three streams qualify and are all used; none will be dropped after seeing results.

**Datasets.**

| Name | Source | Drift | Regimes |
|---|---|---|---|
| `insects` | INSECTS abrupt_balanced (round 2, unchanged) | abrupt, temperature | 5 official change points |
| `gas` | Gas Sensor Array Drift, UCI 224 = OpenML 1476 (content SHA-256 `2025486c…a66a5`) | incremental sensor ageing over 36 months, not recurring | 10 monthly batches; batches with fewer than 300 rows merged with the next (B4+B5, B8+B9) → 8 regimes |
| `electricity` | NSW electricity market, OpenML 151 (content SHA-256 `dc4c9a3b…04cd0`) | seasonal, partly recurring | southern-hemisphere meteorological seasons; partial seasons shorter than 30 days merged with the neighbour → 10 regimes |
| `insects_gradual` | INSECTS incremental-gradual, imbalanced (file SHA-256 `e380c13d…c8dc`), slice | gradual: 35 °C and 23 °C instances interleaved | 2 regimes |

Preprocessing:
- **Gas.** The source file lists batch 10 first; batches are put back in chronological order (sizes 445/1244/1586/161/197/2300/3613/294/470/3600 from the UCI documentation, checked against its per-batch class counts). Inside each batch the source order is label-structured and undocumented, so rows are shuffled once with a fixed seed (0); every run uses the same stream. 128 features, 6 classes. Some classes are missing in some batches (toluene in batches 3–5).
- **Electricity.** Day 363 (rows 17,424–17,471) is a duplicate of day 362 (4 May 1997) and is removed; the weekday column after it is shifted by one and is corrected. Features: `day`, `period`, `nswprice`, `nswdemand`. `date` is dropped (with batch-wise label reveal it acts as a time index into autocorrelated labels) and `vicprice`, `vicdemand`, `transfer` are dropped (constant placeholders for the first 17,424 rows). 943 days × 48 half-hours = 45,264 rows, 2 classes.
- **INSECTS gradual.** Table 2 gives one change point, 58,159; for the imbalanced INSECTS files the change points are 1-based (checked on all five points of abrupt_imbalanced), so the first row of the pure 23 °C concept is 58,158 (0-based). Matching every row to the temperature-ordered pool (incremental_imbalanced) shows that the interleaved transition is the 3,000 rows before it, [55,158, 58,158) (first 23 °C row 55,160, last 35 °C row 58,157), with about 30 %, 50 % and 70 % cold rows in its three thirds. Slice: [45,158, 68,158) = 10,000 pure 35 °C rows, the transition, and 10,000 pure 23 °C rows (margins = 10 × the largest memory). Evaluation regimes: R0 = [45,158, 58,158), R1 = [58,158, 68,158).

**Holdouts (new datasets).** Per (regime, class): min(50, ⌊eligible rows / 4⌋) rows, drawn uniformly at random from the whole regime; a class with fewer than 10 such rows gets no holdout in that regime, and that regime's retention is the balanced accuracy over its remaining classes. In the gradual slice the transition rows are not eligible. Resulting sizes (every seed): gas 1,635 rows, electricity 1,000, gradual slice 550 (R0 without class 5, which has only 10 pure-35 °C rows). `insects` keeps the round-2 holdouts.

**Memory boundary in the gradual slice.** Where the old regime ends is ambiguous under gradual drift, so three arms are run and all are reported: (a) `insects_gradual`: the memory closes R0 at 58,158, the documented change point (same rule as for `insects`); (b) `insects_gradual_start`: at 55,158, the start of the transition (known from the analysis above, so an upper reference rather than a deployable rule); (c) ADWIN on the method's own errors (`*_rb_adwin`, settings as in Amendment 2). Evaluation regimes and holdouts are the same in all arms.

**Protocol.** Unchanged: batches of S = 50, first prediction at row 200, checkpoints every 500 rows and at evaluation-regime ends, z-scores from the first 200 stream rows, frozen TabPFN 6.4.1 (v2.5 weights), `n_estimators` = 4, seeds 0–9.

**Methods.** On `gas`, `electricity` and `insects_gradual`: `sw200`, `sw1000`, `dual200` (150 + 50), `dual1000` (750 + 250), `dual200_rb`, `dual1000_rb`; on `insects_gradual` also `dual200_rb_adwin`, `dual1000_rb_adwin`, and on `insects_gradual_start` also `dual200_rb`, `dual1000_rb`. On `insects`: `dual200` and `dual200_rb` are added.

**Analysis.**
- Reliability: for every dataset and size, mean, SD and 95 % t-interval over seeds of ADAPT, ADAPT (masked), RET, RET per regime and FGT; paired differences of our method against the KDD dual memory and against the sliding window of the same size, with two-sided paired t-tests and the number of seeds with a positive difference. The comparison RET(ours) − RET(KDD dual) is Holm-corrected over the 4 datasets × 2 sizes. ADAPT is reported against a non-inferiority margin of −0.5 pp. No new success thresholds: round 3 characterises when regime balancing helps.
- Mechanism: a labels-only replay of the long-term memory (share of each regime over time), the overlap between the KDD dual memory and the sliding window of the same size, and the per-regime gain against how much the regimes differ.
- Gradual slice: the three boundary arms side by side; balanced accuracy on the transition rows (overall and per third) and on the first 500 rows after the transition starts.
- Other context sizes and short/long ratios are run only if time allows and are reported as exploratory hyperparameter analysis.

### Note added after a pre-flight review (2026-10-13, before any GPU run of round 3)

The amendment above was committed on 2026-10-13 at 16:51 CEST (16:54 in the public repository, the commit before this note). Its sentence "Before this amendment was fixed, only truncated pipeline checks had been run" refers to that time. Afterwards, and before any GPU run, the code, data and run scripts were reviewed. During the review TabPFN was also run on CPU; all such runs are listed here:
- `gas`, to see whether retention is near its ceiling and separates the methods, which could have led to changing the role of `gas` or the Holm family: holdout predictions at every checkpoint over the whole stream, one ensemble member (`sw200` seeds 0–2; `dual200` and `dual200_rb` seeds 0–1; `dual1000` and `dual1000_rb` seed 0; `sw1000` seed 0, 25 of 31 checkpoints; stream rows were not predicted). They showed that the KDD dual memory retains about as much as the sliding window of the same size and the regime-balanced memory clearly more; retention per regime was seen as well.
- `insects_gradual`, to see whether ADWIN raises alarms on real TabPFN errors, without which the ADWIN arm would add nothing: predictions over the whole stream for `dual200_rb_adwin` and `dual1000_rb_adwin`, seed 0, one ensemble member. The first alarm came about 300–500 rows after the start of the transition; a second alarm fell inside R1.
- `electricity`, `gas` and `insects_gradual`, to check that the accuracy floors of the run checks would not be crossed: stream balanced accuracy of a sliding-window context on a sample of batches over the whole stream, seed 0, one ensemble member. About 81 % (200 rows) and 76 % (1,000 rows) on `electricity`, 98 % (both sizes) on `gas`, 67 % (200 rows) on the gradual slice.
- Truncated pipeline checks of the kind described in the amendment (one ensemble member, at most the first 3,000 rows, seeds 0, 1 and 3, all five stream configurations, including the end-to-end test of the run bundle), and two short runs on `insects` (first 1,500 rows) to check that the new code reproduces round 2.
- To check numerical precision, a few single calls with four ensemble members (fp32 and emulated fp16) on `gas` and `electricity` context windows of 200 or 1,000 rows; only finiteness and activation size were inspected.

None of these runs is part of the reported results. Nothing listed in the amendment was changed because of them: datasets, preprocessing, regimes, holdouts, memory-boundary arms, methods, sizes, seeds, metrics, the Holm family and the −0.5 pp margin are as committed at 16:51. Besides operational changes to the run scripts and checks (gas runs first, one automatic retry of a failed run, file checks against the pre-registered boundaries and holdout sizes, the gradual transition zone saved in each result file; none changes what is computed), the review led only to the clarifications and descriptive additions below. The additions were written after the results above had been seen; they describe what will be reported and are not used as tests.

Clarifications (no change in data or code):
- Holdout rule: a (regime, class) gets no holdout when its holdout size min(50, ⌊eligible rows / 4⌋) would be below 10, i.e. when it has fewer than 40 eligible rows. The sizes given above (gas 1,635, electricity 1,000, gradual slice 550) follow this reading; class 5 in R0 of the gradual slice has 10 eligible rows (⌊10/4⌋ = 2). On `gas` this reading also leaves out three (regime, class) cells with 29–32 eligible rows (in R0, R3 and R4); the literal reading would give 1,657 holdout rows instead of 1,635.
- Gas: the 10 batches are in time order, each covering 1–5 recording months (months 1–36); they are not monthly batches.
- Electricity: day 363 repeats day 362 in date code, `period` and `nswdemand`, but one `nswprice` value and two labels differ, as do the dropped `vicprice`/`vicdemand`/`transfer` columns (placeholders on day 362, real from day 363 on) and the weekday column (the shift corrected above); one of the two days had to go, and the first is kept. The `vicprice`/`vicdemand`/`transfer` placeholders are constant or nearly so for the first 17,424 rows. The label is defined by the data source as whether `nswprice` is above its mean over the previous 24 hours (48 rows); this matches all 45,216 rows that have 48 previous rows. No retained feature contains this mean, so no column leaks the label, but the task rewards tracking the recent price level.

Descriptive additions (written after the review, not tests):
- ADAPT (masked): the single-class-run mask was designed for the INSECTS artefact before its change points. It removes 0 % of the stream rows on `gas` and the gradual slice, and 14.2 % on `electricity`, where runs of ≥ 30 equal labels are natural autocorrelation. Both versions are reported as planned.
- KDD dual memory on `gas` and `electricity`: a labels-only replay of the memories (no model) shows that, because classes arrive roughly evenly, the KDD memory overlaps the sliding window of the same size by about 95 % on average. On these two streams the comparison with the KDD dual memory is therefore in effect a comparison with keeping no older regimes; it is reported next to the comparison with the sliding window and the measured overlap.
- ADWIN arm of the gradual slice: for every seed, the alarm positions, the signed offset in rows of the first alarm (the first in the stream, wherever it falls) from the start of the transition and from the documented change point (negative = before it), the number of further alarms, and the share of R0 rows in the final long-term memory. ADWIN reacts to falling as well as rising error, so an alarm inside R1 can split it; a seed without alarms is kept (its memory then equals `dual*`).
- "How much the regimes differ" in the mechanism analysis: for each regime j, the mean 5-fold cross-validated ROC-AUC of a logistic regression that separates regime j's z-scored features from the rows of all later regimes; related to the per-regime gain RET_Rj(ours) − RET_Rj(KDD dual) by Spearman correlation. This measure was chosen after the per-regime retention on `gas` from the CPU runs above had been seen; exploratory.
