# docs/ — English index

The five documents in this folder are written in **Chinese**. This index tells an English
reader what each one contains and where the key numbers live, so nothing here is a dead end.

The primary English document is the repository **[README.md](../README.md)**, which covers
every experiment, result and limitation. These notes are the longer working material behind it.

| File | What it is | Length |
|---|---|---|
| [结果摘要.md](结果摘要.md) | One-page summary: what was done, the key results, the three observations about the paper, open questions | ~74 lines |
| [原理讲解.md](原理讲解.md) | Algorithm walkthrough: EKF vs. FGO theory, the cost function, Gauss-Newton, factor types, LC vs. TC, noise calibration, solver details, experiment design and discussion | ~520 lines |
| [代码逐行分析.md](代码逐行分析.md) | Function-by-function analysis of every module, plus the debugging log (10 bugs found and how) | ~520 lines |
| [论文讲解_对照复现版.md](论文讲解_对照复现版.md) | Section-by-section comparison of the paper's claims against what this reproduction found | ~310 lines |
| [缩写与术语总表.md](缩写与术语总表.md) | Glossary of symbols, abbreviations and parameter names | ~280 lines |

---

## 结果摘要.md — one-page summary

The shortest entry point. Covers: the setup (300 epochs, GNSS at 1 Hz with σ = 6 m and 12%
NLOS epochs, IMU at an effective 100 Hz); the three headline results (iteration-count ablation,
window sweep, parameter sensitivity); the three observations about the paper; and the open
questions put to the authors.

## 原理讲解.md — algorithm walkthrough

Derives both estimators from scratch. Contains the Kalman gain intuition (measurement noise
`R` small → gain near 1 → trust the measurement; prediction covariance `P⁻` small → gain near 0
→ ignore it), the full cost function with its three factor types, the Gauss-Newton iteration,
and the factor-linearity table (motion model, IMU factor and GNSS position factor are all
**linear**; the tightly-coupled pseudorange factor is the only **nonlinear** one).

It also documents the noise calibration that matters most:

| Parameter | Paper (Eq. 22 / 27) | Measured here | Note |
|---|---|---|---|
| Motion-model position σ | 0.3 m | 0.085 m | paper is 3.5× looser |
| Motion-model velocity σ | 0.01 m/s | 0.265 m/s | paper is ~27× tighter |
| IMU factor velocity σ | 0.15 m/s | 0.265 m/s | paper is ~1.8× tighter |

and the solver details (relative pivot threshold `1e-10 × max element`; whitening leaves the
condition number unchanged — measured cond ratio 1.000000).

## 代码逐行分析.md — function-by-function analysis

Module by module (`la.py`, `sim.py`, `estimators.py`, `experiment.py`, `plot.py`,
`make_figures.py`, `_selftest.py`, `_calib*.py`, `run_all.py`, `verify_vs_numpy.py`), with line
counts and the role of each. The most useful part for a reader is the **debugging log**: ten
bugs found during the reproduction, each with symptom, cause and the lesson drawn — including
the PDF text-extraction kerning bug (`Factorgraphoptimization`), an unbounded random walk that
drove velocity to 437 m/s, an absolute solver threshold that falsely reported singularity on a
full-rank matrix, over-constraining the motion model (which reversed the window trend), and
noise parameters copied from the paper (which inverted the trend entirely).

## 论文讲解_对照复现版.md — paper vs. reproduction, claim by claim

Each claim in the paper is listed with what this reproduction measured and a verdict:

| Claim in the paper | What was measured here | Verdict |
|---|---|---|
| Larger window → better accuracy (Fig. 8) | K = 1→150: 9.67 → 4.03 m, monotone | **reproduced** |
| ~30 s window is close to full batch optimization (§4.3) | K=30 gives 5.67 m vs. 4.03 m at K=150 (41% apart) | **not reproduced** (saturation is later than claimed) |
| FGO beats EKF (Table 2) | K=150 4.03 m vs. EKF 6.47 m (38% better) | **same direction, smaller than the paper's 53%** |
| "Multiple iterations" is a source of FGO's advantage (§4.3) | iteration 1→20: spread 1e-13 m | **not reproduced** (the paper's attribution looks wrong) |
| More iterations mitigate linearization error (§4.3) | the paper itself calls the TC nonlinearity "trivial" | **internally inconsistent** |
| Long windows get worse under an abrupt noise change (§4.4) | long windows stayed better throughout | **not reproduced** (honest negative result) |
| Incorrect error modelling → long windows get worse (§4.4) | under mismatch K=30 degrades 2.09× and the ordering reverses; EKF only 1.06× | **reproduced, with a quantitative characterization** |

## 缩写与术语总表.md — glossary

Symbols, abbreviations and parameter names used across the code and notes, including the Latin
and Greek letter conventions (σ scalar vs. Σ matrix vs. ρ correlation), the experiment tags
(`Q1`–`Q6`, `N_SEEDS`, `a_rw`), and the "mismatch factor" defined as true residual over the
noise the estimator assumes.

---

## Note on the tables in these files

The Markdown tables in these five documents lost their pipe characters during an earlier
conversion step, so GitHub renders them as plain paragraphs rather than grids. The information
is all present and readable; only the tabular layout is missing. This is a known cosmetic
defect, tracked for a later fix.
