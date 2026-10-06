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
