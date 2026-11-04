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

### Exploratory short/long ratio analysis (2026-10-17, written before any of its runs)

Amendment 3 states that other context sizes and short/long ratios "are run only if time allows and are reported as exploratory hyperparameter analysis". This analysis was chosen **after the round-3 results had been seen**, to examine two of them: the adaptation cost on `electricity` at 200 rows, and whether the retention gain over the KDD dual memory depends on the 75 / 25 split taken from the KDD paper. It is exploratory: no tests are corrected, no thresholds are set, and the round-3 conclusions are not changed by it.

- **Runs.** Short-term share 0.65 and 0.85 (the two other values of the KDD paper's grid, which that paper searches only for total memories of 600, 800 and 1,000 rows; 0.75 is the round-3 `dual200` / `dual200_rb`), 200 context rows, for the KDD dual memory and the regime-balanced dual memory (`dual200_s65`, `dual200_s85`, `dual200_rb_s65`, `dual200_rb_s85`), on `insects`, `gas`, `electricity` and `insects_gradual` (memory boundary at the documented change point), seeds 0–9, `n_estimators` = 4, the same protocol, streams and holdouts as round 3. The 1,000-row size is not run (GPU time): there the adaptation cost on `electricity` relative to the KDD dual memory was smaller (−0.51 pp against −2.10 pp at 200 rows; both outside the −0.5 pp margin). Whether the retention gain depends on the split is therefore examined at 200 rows only and is not claimed for 1,000 rows.
- **Gate.** Before these runs, the new code reruns `electricity` / `dual200_rb` / seed 0 and must reproduce the round-3 result.
- **Checks before the runs.** After this note was committed, truncated pipeline checks of the run bundle were made (CPU, one ensemble member, first 3,000 rows, seeds 0–1); they are not part of the reported results.
- **Reported.** For each stream and share (0.65, 0.75, 0.85): ADAPT, ADAPT (masked), RET and RET per regime of both memories (mean, SD, 95 % interval over seeds), the paired differences ours − KDD with 95 % intervals and the number of positive seeds, and the long-term pool composition (labels-only replay). Descriptive only.

## Amendment 4 (post-review additions, registered 2026-10-29, before any round-5 run)

**Why.** A review of the thesis draft pointed out that the regime-balanced dual memory (`dual*_rb`, ours) allocates its long-term slots by the official regime boundaries, and that RET scores the same official regimes with equal weight. The retention gain could therefore come from allocating by regime as such, or from knowing the boundaries and from the scoring unit being aligned with them. Round 5 separates these explanations with four analyses: (1) the ADWIN variant on the whole grid, which uses no official boundary in the memory; (2) a class-balanced reservoir long-term store (CBRS), which keeps old rows without any regime information; (3) RET on fixed 2,000-row segments, a scoring unit that no method uses; (4) the class-balanced dual memory with the official regime ID as an extra feature, which gives the model the boundary information without changing the memory.

**Disclosure.**
- Analysis (3) needs no new runs. On 2026-10-29, **before this amendment was written**, it was computed with a read-only script on the stored `ckpt_pred` of the round-2 and round-3 runs, so its results are known; they are reported in `results/phase56_round5.md` §1. Key numbers (RET_seg2000, paired difference in pp at 200 / 1,000 rows, positive seeds): ours − class-balanced dual memory: `gas` +13.39 / +11.71 (10/10, 10/10), `electricity` +3.32 / +2.50 (10/10, 10/10), `insects_gradual` +2.19 / +4.11 (10/10, 10/10), `insects` −0.20 / +0.25 (2/10, 9/10), against +18.00 / +15.74, +3.91 / +3.21, +11.15 / +15.63 and +1.45 / +1.42 for the official RET; ours − sliding window of the same size: `insects` +8.90 / +6.87, `gas` +13.64 / +12.23, `electricity` +3.32 / +2.24, `insects_gradual` +4.75 / +6.89 (all 10/10). For the two existing ADWIN cells the difference to the class-balanced dual memory is also known: `insects` 1,000 rows −0.61 (1/10), `insects_gradual` +2.24 / +2.83 (9/10, 10/10). That first computation did not filter on the batch size and used the S = 10 sensitivity runs for three seeds of `sw200` and `dual1000_rb` on `insects`; the official RET moved by at most 0.04 pp, RET_seg2000 by at most 0.30 pp. All numbers here and in the report use only S = 50, `n_estimators` = 4, complete runs.
- The results of analyses (1), (2) and (4) are unknown. Before this amendment only code checks were run: unit tests of the CBRS store on synthetic label sequences (no model), and truncated pipeline checks on CPU (one ensemble member, seed 0, first 1,500 rows) of `dual200_rb_adwin` on `gas` and `electricity`, `dual200_cbrs` and `dual200_rid` on `gas`. They are not part of the reported results and no setting was chosen from them.
- The CBRS rule was checked against the original paper (Chrysakis & Moens 2020, ICML, PMLR v119, Algorithm 1), in particular its definition of a *full* class (a class that is, or has been at any earlier step, the largest class; once full, always full). A first implementation that treated only the currently largest class(es) as full was corrected before this amendment; one of the truncated checks above used it.

**Methods.** All four streams use the round-3 data, preprocessing, regimes and holdouts; on the gradual slice only arm (a) (`insects_gradual`, memory boundary at the documented change point) and, for the ADWIN variant, arm (c) are used. Sizes 200 and 1,000 rows, seeds 0–9.
- **ADWIN variant** `dual200_rb_adwin`, `dual1000_rb_adwin`: as defined in Amendments 1–2 (river ADWIN, δ = 0.002, 300-row cooldown, on the method's own per-row errors; alarms that would close a segment shorter than 500 rows are ignored; no cap on the number of segments). New runs: `insects` 200 rows, `gas` 200 and 1,000 rows, `electricity` 200 and 1,000 rows. Existing runs complete the grid: `insects` 1,000 rows (round 2) and `insects_gradual` 200 and 1,000 rows (round 3, arm (c)).
- **CBRS dual memory** `dual200_cbrs`, `dual1000_cbrs`: the same structure as the class-balanced dual memory (short-term FIFO of round(0.75 M) rows, i.e. 150 / 750; the oldest short-term row overflows into a long-term pool of 50 / 250 rows; context = both). The long-term pool is maintained by class-balancing reservoir sampling, where the "stream" is the sequence of rows entering the long-term pool. For each such row of class c, first n_c ← n_c + 1 (n_c = rows of class c that have entered the pool so far, this one included; m_c = rows of class c in the pool); then: (i) pool not full: store the row; (ii) pool full and c not full: choose the largest class in the pool (ties uniformly at random) and overwrite a uniformly random row of it; (iii) pool full and c full: with probability m_c / n_c overwrite a uniformly random class-c row, otherwise discard the new row. After every store or overwrite, including while the pool fills, the class(es) with the most rows in the pool are marked full, permanently. No regime information is used (closing a regime does nothing). Random numbers: `numpy.random.default_rng(20000 + seed)`, used only for these draws.
- **Regime-ID ablation** `dual200_rid`, `dual1000_rid`: the memory is exactly `dual200` / `dual1000`. After z-scoring, one column is appended to every row's features: the integer index of the row's official evaluation regime (stream rows: their own regime; holdout rows: the regime they were drawn from), not z-scored and passed to TabPFN as an ordinary feature (no feature types are declared). Values: `insects` 0–5, `gas` 0–7, `electricity` 0–9, `insects_gradual` 0–1 (the evaluation regimes, R0 = [45,158, 58,158) including the transition, R1 from the documented change point). The column thus carries the same boundary information that ours uses to close regimes, as an input to the model instead of a rule of the memory.
- **Protocol.** Unchanged from round 3: batches of S = 50, first prediction at row 200, checkpoints every 500 rows and at evaluation-regime ends, z-scores from the first 200 stream rows, frozen TabPFN 6.4.1 (v2.5 weights), `n_estimators` = 4, `random_state` = seed, run on the ROG. 210 new runs (ADWIN 50, CBRS 80, regime ID 80). Analysis: `scripts/analyze_round5.py`.

**Metrics.** All reported side by side for every method, stream and size:
- ADAPT and the official RET, as before.
- **RET_seg2000 (primary for round 5).** The stream (stream coordinates) is cut into complete segments [2000 s, 2000 (s + 1)); a remainder shorter than 2,000 rows at the end is dropped. Segment s is scored on the holdout rows whose stream position falls in it: their balanced accuracy is averaged over all checkpoints after the segment's end, and the segment scores are averaged with equal weight. Segments without holdout rows are skipped.
- Sensitivity: the same with L = 1,000 and L = 5,000 (`RET_seg1000`, `RET_seg5000`), and L = 2,000 using only checkpoints at least 5,000 rows after the segment's end (`RET_seg2000_lag5000`). On `gas` the L = 5,000 and lag versions use only 2–3 segments and are for reference.

**Comparisons.** Paired differences over the 10 seeds with two-sided paired t-tests, 95 % t-intervals and the number of positive seeds. Holm correction within each family over its 4 streams × 2 sizes, separately for each metric. Descriptive: no pass threshold.
- **A** ours_adwin − class-balanced dual memory (`dual*_rb_adwin` − `dual*`).
- **B** ours − CBRS (`dual*_rb` − `dual*_cbrs`).
- **C** two families, each corrected on its own: **C1** regime ID − class-balanced dual memory (`dual*_rid` − `dual*`), **C2** ours − regime ID (`dual*_rb` − `dual*_rid`).
- Also reported, outside the families (D1–D3, with Holm over the same 8 cells for reference only): ours − class-balanced dual memory, ours − sliding window, CBRS − class-balanced dual memory.

**Pre-specified readings** (fixed now; they label the results and are not tests):
- **A.** If on `gas` the official-RET gain of ours_adwin over the class-balanced dual memory is ≥ +10 pp at a size, dependence on the official boundaries is regarded as largely ruled out on `gas` at that size.
- **B.** If on a stream ours − CBRS on RET_seg2000 is ≤ 0.5 pp at a size, the gain of ours on that stream and size is attributed to keeping old rows rather than to allocating by regime.
- **C.** If regime ID − class-balanced dual memory is at least half of ours − class-balanced dual memory (both RET_seg2000, same stream and size), a substantial part of the gain is regarded as explainable by the boundary information itself. Applied only where ours − class-balanced dual memory on RET_seg2000 is positive (known from §1; so not on `insects` at 200 rows, where it is −0.20 pp).

Whatever the results, all four analyses are reported in the thesis.

### Clarifications to Amendment 4 (2026-10-29, after a check of the registered text and code, still before any round-5 run)

These notes correct or qualify the wording above. No method, run, setting or comparison changes; the only code change in the same commit is a comment.

1. **How TabPFN reads the regime-ID column (affects reading C).** "Passed to TabPFN as an ordinary feature" means that no feature types are declared; it does not mean the column is treated as numerical. TabPFN 6.4.1 infers each column's type from the context rows of every fit (`tabpfn/preprocessing/modality_detection.py`; `MIN_UNIQUE_FOR_NUMERICAL_FEATURES` = 4, `MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE` = 100): if every context row has the same regime ID the column is constant and is removed; with 2–3 distinct IDs (and more than 100 context rows, always the case here) it is categorical, and a query row whose ID does not occur in the context is encoded as an unknown category (code −1 at the first encoding, a missing value in the ensemble members that re-encode categories), so it shares no ID value with any context row; only with 4 or more distinct IDs is it numerical. A fit of `TabPFNClassifier` on 200 rows confirmed this (1 ID: constant; 2–3: categorical; 4–5: numerical). The memory of `dual*_rid` is exactly that of `dual*` and does not depend on the model's predictions, so the context at every checkpoint can be replayed without the model. Replay of seed 0, checkpoints at which at least one earlier regime is scored (count of checkpoints with a constant / categorical / numerical ID column; mean share of the scored earlier-regime holdout rows whose ID is absent from the context): `insects` 200 rows 62 / 17 / 0, 88 %; 1,000 rows 42 / 37 / 0, 81 %; `gas` 19 / 10 / 1, 72 %; 11 / 17 / 2, 63 %; `electricity` 82 / 3 / 0, 99 %; 67 / 18 / 0, 93 %; `insects_gradual` 19 / 1 / 0, 95 %; 14 / 6 / 0, 70 %. Because the class-balanced dual memory mostly holds rows of the current regime, the column usually cannot tell the model which old regime a holdout row comes from. **Reading C is therefore one-sided:** a C1 difference that reaches the threshold still labels the gain as substantially explainable by the boundary information; a C1 difference near zero shows only that the regime ID given as an input, with a memory that does not keep old rows, does not replace keeping old rows. It does not rule out that the boundary information explains part of the gain of ours, and it is not reported as such. The same replay counts (all seeds) are reported with the round-5 results; they need no rerun.
2. **Lag wording.** "Only checkpoints at least 5,000 rows after the segment's end" should read "only checkpoints **more than** 5,000 rows after the segment's end" (checkpoint time t > segment end + 5,000), which is what `ret_segments_per_segment` computes and what the numbers disclosed above use. With lag 0 the rule is t > segment end, the same strict inequality as the official RET.
3. **RET_seg2000 is regime-free in its scoring unit, not in its holdout rows.** The holdouts are drawn per (regime, class) with a fixed count, so holdout density along the stream follows the regimes, and a short regime weighs heavily in the segment that contains it (seed 0: on `gas` segment [2,000, 4,000) has 329 holdout rows, 80 of them from regime 3, which spans 278 stream rows; on `insects` most segments have 30–51 holdout rows, but segments 7–8 and 16–18 have 108–181 because regimes 1, 3 and 4 are short; on `electricity` 35–52 rows in every segment). RET_seg2000 removes the equal weight per official regime, but not the regime-aligned density of the holdout rows. This is reported with the round-5 results and in the thesis limitations.
4. **Batches are cut at the official change points in every run, including the ADWIN variant (affects reading A).** The runner never lets a batch of S = 50 span an official change point, whatever the segmentation (`run_stream`: cuts = boundaries ∪ memory boundaries); checkpoints are also taken at official regime ends, but they do not change the memory (digest check). In `*_adwin` runs the memory closes regimes only at ADWIN alarms and never sees the official change points, but up to 49 rows after each change point are predicted with a memory updated exactly up to it. This small use of the official boundaries is pre-existing (rounds 2–3) and is disclosed with reading A; it is not changed, so that round-5 runs stay comparable with the earlier ADWIN runs.

## Amendment 5 (round 6: regime-balanced reservoir dual memory, registered 2026-11-01, before any round-6 run)

**Why.** Round 5 (Amendment 4) found that a class-balanced reservoir long-term store (CBRS), which uses no regime information, retains old regimes about as well as ours or better: ours − CBRS on RET_seg2000 (pp, 200 / 1,000 rows) is `insects` −0.30 / −1.02, `gas` +2.17 / +0.67, `electricity` −0.06 / −2.47, `insects_gradual` −2.74 / −1.37 (reading B applies in 6 of the 8 cells; ours is significantly better only on `gas` at 200 rows). A replay of the memories from the labels alone (no model) then showed that the long-term eviction acts at two levels. Ours gives every regime an equal share, but inside each (regime, class) it keeps only the newest rows that entered the pool (mean position of the kept rows inside their regime 0.78–1.00, 10 seeds). CBRS keeps a time-uniform sample inside each class (0.41–0.57) but allocates in proportion to how many rows of each regime entered the pool, so with 200 rows a short regime can lose all its rows (no row of `insects` R4 in the memory in 53 % of the updates after it ends, `gas` R3 29 %; ours 0 %). Round 6 tests a memory that combines the two: equal shares between regimes, class balance and reservoir sampling inside each regime.

**Disclosure.**
- The design was made after seeing the round-5 results; it is post hoc, and it is run on the same four streams, so it may suit these streams better than others. This is stated with the results.
- Exploratory analyses of stored runs (rounds 2, 3 and 5), not pre-registered, were done before this amendment and motivated the design: per-regime RET_j of ours, CBRS and the class-balanced dual memory (e.g. `gas` R0 at 200 rows: 95.4 / 80.3 / 59.8 %); the worst-regime retention RET_worst defined below (ours − CBRS, pp, 200 / 1,000 rows: `insects` +0.62 / +2.12, `gas` +14.56 / +2.76, `electricity` +4.31 / +0.30, `insects_gradual` −11.27 / −8.05, where only one regime is scored); the share of updates after a regime ends in which the memory holds none of its rows; and the position of the kept rows inside their regime. They are reported as exploratory with the round-5 results.
- Before this amendment the new memory was run only without the model: unit tests on synthetic labels, and a labels-only replay on the four streams × 2 sizes × seeds 0–9 (label and regime source: the round-5 CBRS runs). Replay results: per-regime counts after each regime's end equal ours' within 1 row on average; mean position of the final pool's rows inside their regime 0.47–0.58 (ours 0.88–1.00, CBRS 0.49–0.53); no ended regime is ever without rows (CBRS: 5.0 % of the post-end updates on `gas` and 10.6 % on `insects` at 200 rows, averaged over regimes and seeds). Its RET and ADAPT are unknown.
- The expectations below were written before any run, partly from an independent review of the mechanism.

**Method** `dual200_rbr`, `dual1000_rbr` (regime-balanced reservoir dual memory). Same structure as the class-balanced dual memory: short-term FIFO of round(0.75 M) rows (150 / 750), the oldest short-term row overflows into a long-term pool of 50 / 250 rows, context = both. Regimes are closed at the official memory boundaries, as for ours (arm (a) on the gradual slice); each row gets the current regime index when it is revealed. For each row entering the long-term pool, with regime g and class c, first n[g,c] ← n[g,c] + 1 (n[g,c] = rows of (g, c) that have entered the pool so far, this one included; m[g,c] = rows of (g, c) in the pool); then:
1. pool not full: store the row;
2. g not a full regime: in the regime with the most rows in the pool (ties: the most recent regime), take the class with the most rows (ties uniformly at random) and overwrite a uniformly random row of that (regime, class);
3. g full, c not full within g: overwrite a uniformly random row of the class with the most rows within g (ties uniformly at random);
4. otherwise: with probability m[g,c] / n[g,c] overwrite a uniformly random row of (g, c), else discard the new row.
After every store or overwrite, including while the pool fills, the regime(s) with the most rows in the pool are marked full, and within every regime the class(es) with the most rows are marked full in that regime; both marks are permanent. This is the CBRS rule of Amendment 4 applied twice, to regimes and then to classes within a regime, with a reservoir inside each (regime, class) cell; with a single regime it makes exactly the decisions of the CBRS store (unit test with a shared random-number stream). Random numbers: `numpy.random.default_rng(30000 + seed)`, used only for these draws. The memory depends only on the labels and the official boundaries, not on the model's predictions, so the digest at every checkpoint can be checked by a labels-only replay.

**Runs.** 4 streams × 200 / 1,000 rows × seeds 0–9 = 80 runs, protocol unchanged from round 5 (S = 50, first prediction at row 200, checkpoints every 500 rows and at evaluation-regime ends, z-scores from the first 200 stream rows, frozen TabPFN 6.4.1 with the v2.5 weights, `n_estimators` = 4, `random_state` = seed, ROG GPU). Calibration gate `calib7`: `gas` `dual200_cbrs` seed 0 is rerun with the round-6 code and must be bit-identical to its round-5 result. Not run, and stated as limitations: an ADWIN variant of the new memory (ADWIN closes 37–46 segments on `electricity` against 10 regimes, about one long-term row per segment at 200 rows, so the per-regime class balance and reservoir would need an extra fallback rule that is not designed here), and a start-of-transition arm on the gradual slice.

**Metrics.** As in Amendment 4 (ADAPT, official RET, RET_seg2000 as primary, and its sensitivity versions), plus **RET_worst**, registered now for round 6: for each run, the minimum over the scored regimes j of RET_j (the per-regime retention of Amendment 3); reported as the mean over seeds. On `insects_gradual` only R0 is scored, so there RET_worst = RET. Per-regime RET_j are reported descriptively.

**Comparisons.** Paired differences over the 10 seeds, two-sided paired t-tests, 95 % t-intervals and the number of positive seeds; Holm correction within each family over its 4 streams × 2 sizes, separately for each metric.
- **E1** new memory − CBRS (`dual*_rbr` − `dual*_cbrs`).
- **E2** new memory − ours (`dual*_rbr` − `dual*_rb`).
- Also reported, outside the families: new memory − class-balanced dual memory, new memory − sliding window of the same size.

**Expectations** (from the mechanism; not tests). E2: the new memory retains more than ours on `electricity` (mainly at 1,000 rows), on the long regimes of `insects`, and on R0 of `insects_gradual` at 1,000 rows; about equal on `gas`. E1: above CBRS on `gas` at 200 rows; about equal on `gas` at 1,000 rows, on `electricity` and on `insects` (RET_seg2000); below CBRS on the old regime of `insects_gradual`, where CBRS keeps more old rows of the classes that become rare later; RET_worst at least that of CBRS on the multi-regime streams. ADAPT within about 1 pp of ours.

**Pre-specified readings** (fixed now; they label each cell and are not tests). Holm-adjusted p < 0.05 in the family and metric concerned:
- **E1 on RET_seg2000:** positive and significant → allocating by regime adds retention beyond keeping old rows on that stream and size; negative and significant → it costs retention there; otherwise no difference shown.
- **E1 on RET_worst** (the six multi-regime cells): positive and significant → the new memory protects the worst-retained regime better than CBRS there.
- **E2 on RET_seg2000:** positive and significant → time-uniform sampling inside a regime retains more than keeping its newest rows there.

Whatever the results, round 6 is reported in the thesis together with rounds 2–5. Analysis: `scripts/analyze_round5.py` (extended with the new memory, families E1–E2 and RET_worst).

## Post-hoc checks (2026-11-03, not registered)

Two checks were added after all round-5 and round-6 results had been seen. They are not part of Amendments 4 or 5, are not tests and change no reported number; both are computed by `scripts/analyze_round5.py`.
- **Chance baseline for the ADWIN alarms.** For each run of the ADWIN variant (`dual*_rb_adwin`, 80 runs), all its alarm positions are shifted cyclically along the stream by 200 equally spaced offsets, which keeps their number and spacing but not their timing; the share of official change points followed by an alarm within 1,000 rows, the share followed by an alarm before the next change point, and the median delay to that alarm are recomputed and averaged over the offsets (the median for the delay). Result: on `insects` and `gas` the alarms are much closer to the change points than the shifted ones (an alarm within 1,000 rows after 86 / 88 % and 49 / 41 % of the change points at 200 / 1,000 rows, against 20–24 % shifted; median delay 28–110 rows against 936–2,751), although on `gas` only 36–39 % of the change points are followed by any alarm before the next one; on `electricity` they are not (86 / 64 % against 81 / 77 %; median delay 554 / 698 against 518 / 564 rows), so there the alarms do not track the season changes.
- **Labels-only replay of all round-5 and round-6 runs.** All 320 dual-memory runs compared in rounds 5 and 6 (`dual*_cbrs`, `dual*_rb_adwin` including its 30 runs from rounds 2 and 3, `dual*_rid` and `dual*_rbr`, 80 each) are replayed from the labels alone, the ADWIN variant with its stored alarm positions. Result: in all 320 runs the memory digest, the regime closures and the checkpoint times match the stored results at every checkpoint.

## Note on file references (2026-11-04)

The round-5 report named in the disclosure of Amendment 4 is not part of the public repository. The per-seed adaptation and retention values of rounds 5 and 6, and of the round-2 and round-3 runs they reuse, are in `results/round5_per_seed.csv`; these values and the alarm statistics above are computed by `scripts/analyze_round5.py` from the raw run files. The registered text above is unchanged.
