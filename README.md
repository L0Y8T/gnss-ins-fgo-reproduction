# GNSS/INS Factor Graph Optimization vs. Extended Kalman Filter — An Ablation-Study Reproduction

> **The core reproduction uses only the Python standard library — zero third-party dependencies.**
> It reproduces the ablation experiments of Sections 4.3 / 4.4 of Wen et al. (2021), *NAVIGATION*,
> and independently tests several of the paper's arguments.
>
> Chinese version: **[README.zh.md](README.zh.md)**
>
> The one exception: the tightly-coupled pseudorange experiment `expC2_tc_pseudorange.py` uses numpy (see §4).

---

## TL;DR

> **The paper attributes FGO's advantage over EKF to "multiple iterations."**
> **In the system studied here, however, the Gauss-Newton iteration count affects the result at the 10⁻¹³ level — i.e. not at all.**
> **Swapping in the paper's actual tightly-coupled pseudorange model (nonlinear in the state) leaves this conclusion intact** (Experiment C2).
> The reason is therefore not that "the model is affine" but that **the cost function is quadratic**: an L2 Newton step reaches the exact minimum in two steps, so further iterations are no-ops.
> FGO's real benefit comes from batch smoothing inside the window, not from the iteration count.
>
> **There is one exception**: once a robust kernel (a non-quadratic loss) is enabled, the iteration count becomes a first-class variable — and **more iterations make the result worse**. The minimum of the robust cost that IRLS converges to is not the minimum-error point (Experiment C3).

---

## 1. What this repository does

Wen et al. (2021) compare two GNSS/INS fusion estimators:

| | EKF | FGO |
|---|---|---|
| Approach | Per-epoch recursion: predict → update | On each epoch, **re-optimize the last K epochs jointly** |
| History | Compressed into a covariance matrix (recursive) | History in the window is **kept explicitly** and solved jointly |
| Solve size per step | 4 unknowns | 4K unknowns |

The paper concludes that FGO is significantly more accurate than EKF, and that **longer windows and more iterations both improve accuracy**.

This repository reproduces that ablation study in **dependency-free pure Python** and tests each of those claims.

---

## 2. The four original experiments and their main findings

### Experiment 1 · Iteration-count ablation — **"multiple iterations" has no effect**

Window fixed; only the Gauss-Newton iteration count is varied (1 / 2 / 3 / 5 / 10 / 20):

| Window K | 1 iteration | 20 iterations | **Spread** |
|---|---|---|---|
| 2 | 8.6155 m | 8.6155 m | **1.315e-13 m** |
| 5 | 6.7743 m | 6.7743 m | **2.487e-14 m** |
| 10 | 5.3180 m | 5.3180 m | **8.882e-16 m** |
| 30 | 4.7159 m | 4.7159 m | **5.329e-15 m** |

**A spread of order 1e-13 = floating-point rounding = the iteration count does not affect the result.**

**Why**: every model in this experiment is **linear** (motion model, IMU factor, GNSS observation), so the objective is a **quadratic** function of the state, and Newton / Gauss-Newton reaches the exact minimum of a quadratic in **one step**. The second iteration merely marks time.

> **This does not mean "iterations are useless," but "model nonlinearity" is not what makes them useful.**
> Experiment C2 replaces the observation model with the paper's actual tightly-coupled pseudorange `ρ = ‖SV − X‖ + b` (nonlinear in the state); the L2 spread is still **0.000 m**, and perturbing the window's initial guess to **500 m** costs only one extra step while the solution is completely unchanged. The paper's description of this nonlinearity as "trivial" is correct.
>
> What actually makes the iteration count matter is **the cost function ceasing to be quadratic** — i.e. enabling a robust kernel. See §2B.

### Experiment 2 · Window-size sweep — gains start only at **K=3**

| Window K | 1 | 2 | 3 | 5 | 10 | 30 | 150 | EKF |
|---|---|---|---|---|---|---|---|---|
| Mean (m) | **9.67** | **9.67** | 9.13 | 7.88 | 6.39 | 5.67 | **4.03** | 6.47 |

**K=1 and K=2 give bit-identical results** (9.6717 / 0.6292 / 7.2808 / 26.4329).

#### A structural observation: at K=1 and K=2, FGO degenerates to raw GNSS

Count the **unknowns and constraints**:

| Window K | Unknowns 4K | Constraints 6K−4 | Status |
|---|---|---|---|
| 1 | 4 | 4 (including 2 rows of velocity prior) | **exactly determined** |
| 2 | 8 | 8 | **exactly determined** |
| 3 | 12 | 14 | **overdetermined** ← real estimation starts here |
| 30 | 120 | 176 | overdetermined |

**"Exactly determined" = the system has a unique solution = every weight becomes irrelevant.**

Measured: at K=1 and K=2, FGO's **position estimates are identical to the raw GNSS measurements** epoch by epoch — the deviation is exactly 0.000e+00 m on most epochs, with a maximum of 1.8e-15 m at K=1 and 4.5e-13 m at K=2, i.e. last-bit rounding only. In other words **the position estimate *is* the raw GNSS measurement, with no smoothing whatsoever**.

> This bears on the paper's choice to use K=1 FGO to represent "single-iteration FGO" — under that configuration FGO performs no least-squares estimation at all. Window accuracy has a definite starting point: **K=3**.

### Experiment 3 · Abrupt noise change — the Section 4.4 counterexample did not reproduce

GNSS noise is switched from 6 m to 12 m at epoch 180. Section 4.4 argues that a long window degrades in this situation by "carrying the pre-change noise characteristics over to the current epoch"; **this experiment did not observe that**.

### Experiment 4 · Parameter mismatch — **EKF is robust, FGO is fragile** (the most valuable set)

Ground-truth motion is fixed; only the noise parameters **assumed** by the estimator are changed:

| | Matched | Mismatched | **Degradation** |
|---|---|---|---|
| **EKF** | 5.293 m | 5.010 m | **1.06×** |
| FGO K=10 | 6.036 m | 6.388 m | 1.06× |
| **FGO K=30** | 5.781 m | **12.089 m** | **2.09×** |
| FGO K=80 | 4.339 m | 8.613 m | 1.99× |

**Mechanism**:

- **EKF** maintains a covariance matrix online and uses it to weight model and observations **automatically** → it adapts when parameters are set wrong.
- **FGO** takes the inverse covariance as a **fixed weight** and puts every factor into one least-squares problem on equal footing → wrong parameters stay wrong.

**And under mismatch the window-size ordering reverses**: with matched parameters long windows win, but with mismatched parameters K=30 is worse than a small window.

---

## 2B. Extension experiments: A / B / C

Three extension groups were run after the original reproduction; all scripts are under `code/`.

| Group | Scripts | Question |
|---|---|---|
| **A** | `robust.py`, `expA_noise.py`, `expA3_*`, `expA4_*` | Does FGO's advantage hold when the noise is not Gaussian? Do robust kernels help? |
| **B** | `expB_yaw.py` | How does the solver behave once orientation (rotation) is added to the state? |
| **C** | `expC1_delta_sweep.py`, `expC2_tc_pseudorange.py`, `expC3_iteration_multiseed.py` | Can the paper's tightly-coupled model explain the "iterations" claim? |

### A · Non-Gaussian noise and robust kernels

The stock `sim.py` noise **was never Gaussian to begin with**: 88% of epochs are N(0, 6²), and the remaining **12% carry a deterministic bias of (15, 20) m**. The true σ is **8.57 m** while the estimator keeps weighting as if it were 6.0 m — **underestimating the noise by a factor of 1.4**.

| Noise | Mean improvement of Cauchy over L2 (K=20) |
|---|---|
| Pure Gaussian (homoscedastic) | −5.1% |
| Laplace (heavy-tailed) | larger |
| Student-t(ν=3) | larger |
| **Stock (with 12% outliers)** | **−25.1%** |

Share of epochs with error above 15 m: **4.4% → 0.8%**.

### B · Once rotation is added, the solver no longer lands in one step

| Model | Magnitude spanned by step 1 | Steps to converge |
|---|---|---|
| 2D affine `[p, v]` | **14 orders of magnitude** (= mathematically exact in one step) | 2 |
| 5D with yaw `[p, v, θ]` | 4.0 – 5.9, then ≈1.8 per step | 4 (K=3) → 10 (K=20) |

**But the accuracy gain is limited.** This only shows that "once the model is nonlinear, the solver's behaviour changes" — not that "more iterations significantly improve accuracy."

### C1 · δ sensitivity of the robust kernel

Group A used δ = 2.0 throughout. Sweeping δ from 1.0 to 8.0:

| Kernel | Improvement over L2 at K=20 |
|---|---|
| **Cauchy** | **+4.0% … +33.8% (positive throughout)** |
| Huber | +0.0% … +23.5%, **vanishing for δ > 5** |

**Conclusion**: Cauchy's gain is **not a lucky δ tuning** (positive across the whole range), but its **magnitude varies by 30 percentage points**; Huber's gain **depends on δ being small enough**. So "robust kernels help" **must always be qualified by δ**.

A sanity check: at δ = 8.0 the weights approach 1 and the result falls back to L2 — which is what the δ → ∞ limit demands, and what was measured.

### C2 · Tightly-coupled pseudorange: does nonlinearity make iterations matter?

The observation is rebuilt from the paper's Eq. (32): `ρ = ‖SV − X‖ + b`, 8 satellites, horizontal GDOP **1.019**, per-epoch clock bias as an extra state, σ_ρ calibrated to **5.889 m** so that single-epoch horizontal accuracy matches the 6 m of the loosely-coupled setup. On 12% of epochs, 1–2 satellites carry an NLOS positive bias of **+25 m**.

| Check | Result |
|---|---|
| L2 spread over n_iter 1→30 on the nonlinear model | **0.000 m** (bit-identical for K = 2 / 3 / 10 / 20) |
| Window initial-guess perturbation 0 / 10 / 100 / **500 m** | iterations 2.42 / 3.00 / 3.00 / 3.00; final error **constant at 3.7029 m** |
| Cauchy, n_iter 1→30, K=3 | 4.931 → 5.279 m (**+7.1%**) |
| Cauchy, n_iter 1→30, K=10 | 3.105 → 3.367 m (**+8.4%**) |
| Cauchy, n_iter 1→30, K=20 | 2.904 → 3.131 m (**+7.8%**) |

**Two conclusions**:

1. **Model nonlinearity is not enough to make the iteration count affect accuracy.** The paper is right that the nonlinearity is "trivial" — and it is even more trivial than the paper says.
2. **"More iterations is worse" belongs to the robust kernel, not to the model.** The same phenomenon appears in both the loosely- and tightly-coupled setups.

**The real contribution of tight coupling is redundancy at short windows:**

| Setup | Degrees of freedom at K=2 | Robust kernel vs. L2 |
|---|---|---|
| Loosely coupled | 8 unknowns / 8 equations (**exactly determined**) | **1.7e-13 m** — provably useless |
| Tightly coupled | 10 unknowns / 21 equations (redundancy 11) | huber **+1.8%**, cauchy **+1.7%** |

### C3 · 30 seeds + mechanism

| Condition | Cauchy, n_iter 1 → 30 | Seeds favouring n_iter=1 | Sign test |
|---|---|---|---|
| Stock (with outliers), K=10 | 5.079 → 5.681 m (**+11.8%**) | **30/30** | **p = 1.9e-09** |
| Stock (with outliers), K=20 | 4.471 → 4.844 m (+8.3%) | **30/30** | **p = 1.9e-09** |
| **Pure Gaussian, outlier mechanism disabled**, K=10 | 4.301 → 4.608 m (**+7.1%**) | **30/30** | **p = 1.9e-09** |
| L2, any K | difference 0 (rounding) | — | — |

**Mechanism** (stock, K=10, averaged over 30 seeds):

| Quantity | n_iter = 1 | n_iter = 30 |
|---|---|---|
| True mean error | 5.079 m | **5.681 m (worse)** |
| Robust cost (what IRLS minimizes) | 0.994 | **0.950 (down 4.4%)** |
| Mean down-weighting of GNSS factors | 0.752 | **0.762 (relaxed)** |

**IRLS is converging correctly — it is the minimizer of the robust cost that is not the minimum-error point.** Later iterations partly undo the suppression the first step applied. And since **this happens with pure Gaussian noise and no outliers too**, the first IRLS step's gain looks more like **shrinkage estimation** than "outlier rejection."

---

## 3. Quick start

The core experiments need **only Python 3.8+ and no third-party packages**.

```bash
cd code

# (1) linear algebra self-check
python la.py
#   expected: solve A x = b -> [1.0, 3.0]; weighted least squares in one step -> 4.8

# (2) inspect the simulated data
python sim.py

# (3) estimator self-check (~10 s)
python _selftest.py

# (4) run the four original experiments (~3-10 min)
python experiment.py

# (5) figures
python make_figures.py
#   then open ../results/fig1_window_vs_accuracy.svg in a browser

# (6) cross-validation against LAPACK (needs numpy; reference judge only)
python verify_vs_numpy.py
```

Extension experiments (groups A and B are also dependency-free):

```bash
python robust.py                        noise-model self-check (variance alignment)
python expA_noise.py                    experiments A0 / A1 / A2
python expA3_robust_iteration_seeds.py  experiment A3 (multi-seed)
python expA4_robust_best.py             experiment A4 (full grid at the best iteration count)
python expB_yaw.py                      experiments B1 / B2 / B3
python expC1_delta_sweep.py             experiment C1 (delta sweep)
python expC3_iteration_multiseed.py     experiment C3 (30 seeds + mechanism diagnosis)
python _selftest_c2.py                  grouped self-check for experiment C2
```

Exactly one script **needs numpy**:

```bash
python expC2_tc_pseudorange.py          experiment C2 (tightly-coupled pseudorange, see §4)
```

**Performance note**: `N_SEEDS = 5` in `experiment.py` makes Experiment 2 repeat 5 times and average. Setting it to `1` is faster (slightly noisier; the trend is unchanged). `expC3` uses 30 seeds and takes about an hour at full length.

---

## 4. Why numpy is deliberately avoided

**This is a design choice, not an environment constraint.**

Hand-written linear algebra makes every numerical step traceable — which is precisely what allowed this reproduction to pin down the dimensional inconsistency in the paper's Eq. (22). It also means the whole repository runs immediately after cloning, so anyone can verify every number without setting up an environment first.

### And this is **verifiable**, not merely asserted

| Check | Result |
|---|---|
| Cross-validation against LAPACK (a real sliding-window problem) | relative error **≤ 2.26e-14** |
| Cross-validation against LAPACK (random well-conditioned systems) | relative error **≤ 6.77e-16** |
| Independent optimality check (drop in ‖HᵀWr‖ at the solution) | **≈1e12×**, confirming a true minimum |
| **Main pipeline with every third-party import blocked** | **passes** |
| **Re-run with fixed seeds: the four CSVs are byte-identical** | passes |

To reproduce:

```bash
cd code && python verify_vs_numpy.py
```

> numpy is used here **only as a judge** and takes no part in the main pipeline. With numpy removed, `la` / `sim` / `estimators` / `experiment` all run as before.

### The single exception: `expC2`

The loosely-coupled state is only 4K-dimensional, so a hand-written Gaussian elimination is entirely sufficient. The tightly-coupled window is **5K-dimensional** and adds 8 pseudorange factors per epoch — at K=20 that is a 255 × 100 system, solved several thousand times within one experiment (a 432-run grid × 150 epochs × up to 30 iterations). **That scale is not feasible with a hand-written solver**, so `expC2` uses numpy for the dense solve.

Only three files in the package use numpy, and each is explicitly marked:

| File | Why numpy |
|---|---|
| `expC2_tc_pseudorange.py` | solve scale, see above |
| `_selftest_c2.py` | it is `expC2`'s grouped self-check and follows it |
| `verify_vs_numpy.py` | numpy **only as a judge**, not in the main pipeline |

**Every other experiment (groups A and B, C1, C3) goes through the hand-written path in `la.py`.**

**On performance**: this reproduction **claims no performance advantage**. A hand-written solver is necessarily far slower than LAPACK under repeated calls (measured ≈112 ms in pure Python at n=120). It was written for **traceability and zero dependencies, not speed**.

---

## 5. Repository layout

```
.
├── README.md                    English (this file)
├── README.zh.md                 Chinese
├── LICENSE
├── .gitignore
├── code/
│   ├── la.py                    linear algebra (matrix ops + Gaussian elimination + WLS)
│   ├── verify_vs_numpy.py       cross-validation against LAPACK
│   ├── sim.py                   synthetic data generation (trajectory + GNSS + IMU)
│   ├── estimators.py            EKF and FGO implementations (core)
│   ├── experiment.py            the four original experiments
│   ├── plot.py                  hand-written SVG plotting (dependency-free)
│   ├── make_figures.py          CSV -> SVG
│   ├── run_all.py               run everything and log the output
│   ├── _selftest.py             quick self-check
│   ├── _selftest_c2.py          tightly-coupled grouped self-check
│   ├── _calib.py / _calib2.py   noise-covariance calibration tools
│   ├── robust.py                non-Gaussian noise models + robust kernels (group A)
│   ├── expA_noise.py            experiments A0 / A1 / A2
│   ├── expA3_robust_iteration_seeds.py
│   ├── expA4_robust_best.py
│   ├── expB_yaw.py              experiment B (5D with yaw)
│   ├── expC1_delta_sweep.py     experiment C1
│   ├── expC2_tc_pseudorange.py  experiment C2 (uses numpy)
│   ├── expC3_iteration_multiseed.py
│   ├── make_figures_ext.py      figures for groups A / B (SVG, dependency-free)
│   └── requirements.txt         dependency note (numpy needed for expC2 only)
├── docs/                        (Chinese working notes; see docs/README.md for an English index)
│   ├── README.md                English index of the five Chinese documents
│   ├── 结果摘要.md               one-page summary
│   ├── 原理讲解.md               algorithm walkthrough
│   ├── 代码逐行分析.md            function-by-function analysis
│   ├── 论文讲解_对照复现版.md      section-by-section paper vs. reproduction
│   └── 缩写与术语总表.md          symbols and abbreviations
└── results/
    ├── exp1..exp4_*.csv         the four original experiments
    ├── expA1..expA4_*.csv       experiment group A
    ├── expB1..expB3_*.csv       experiment group B
    ├── expC1_delta_sweep.csv    experiment C1
    ├── expC2a/b/c_*.csv         experiment C2
    ├── expC3_*.csv              experiment C3 (incl. per-seed paired results)
    ├── *_console.txt            full console output of each run
    ├── fig1_window_vs_accuracy.png / .svg
    └── fig2..figB3_*.svg        remaining figures
```

> The three PNG figures for experiment group C are generated by a separate script and are not in this repository (that step uses matplotlib, which conflicts with the zero-dependency principle, so it stays outside).

---

## 6. Implementation notes

### State and models (2D loosely coupled)

The state is 4-dimensional, `x = [E, N, vE, vN]ᵀ`; in batch optimization the K epochs in the window are stacked into a 4K vector.

Three factor types:

| Factor | Constrains | Definition |
|---|---|---|
| **Motion-model factor** | **position only** | `p_{g+1} = p_g + v_g·Δt` |
| **IMU factor** | **velocity only** | `v_{g+1} = v_g + a_g·Δt` |
| **GNSS factor** | position observation | `p_g = z_g` |

> **The motion-model factor must not constrain velocity.** Adding a row `v_{g+1} = v_g` to it implies "acceleration is always zero," which directly contradicts the IMU factor and makes **long windows perform worse** (the trend reverses). In the paper, `h_MM` in Eq. (21) writes position only, and the velocity recursion is handled entirely by the INS factor (Eq. 25) — which matches this implementation.

### Solver

Gauss-Newton normal equations `(HᵀWH)ΔX = −HᵀWr`, solved by **Gaussian elimination with partial pivoting**.

Two engineering details worth noting:

1. **Relative pivot threshold**: the criterion is `1e-10 × largest matrix element`, not a fixed value. The reason is not "values may be small" but **scale invariance** — multiplying both sides of `Ax=b` by a constant leaves the solution unchanged, so the threshold must follow the problem's scale. Measured: scaling the same system by 1e-12 makes an absolute threshold report singularity while the relative threshold behaves normally.
2. **Whitening**: scale `H` and `r` by `√W` so that `HᵀWH = (√W H)ᵀ(√W H)`. This saves memory and generalizes (use a Cholesky factor when Σ is non-diagonal). **It does not change the condition number** — the two are mathematically the same matrix (measured cond ratio 1.000000).

### Two observations about the paper

Two points worth confirming with the authors came up during the reproduction (**these may stem from how this implementation calibrates its parameters; a misunderstanding is not ruled out**):

1. **Dimensions in Eqs. (21)/(22)**: `h_MM` in Eq. (21) writes position only, yet the covariance in Eq. (22) gives standard deviations for both position and velocity components. If the motion-model factor does not constrain velocity, the velocity variance should not appear in that factor.
2. **Values in Eq. (22)**: calibrating with the values as printed (position 0.3 m, velocity 0.01 m/s) reverses the window trend relative to the paper's conclusion (longer windows are worse). Calibrating on the ground-truth trajectory instead (position 0.6 m, velocity 0.7 m/s) restores the expected trend. The measured IMU-factor residual RMS is 0.265 m/s, whereas Eq. (27) assumes 0.15 m/s (about 1.8× tighter).

> Noise covariances must be calibrated on one's own data — which is itself consistent with the spirit of the paper's Section 4.4.

---

## 7. Limitations (stated honestly)

| # | Limitation | Impact |
|---|---|---|
| 1 | **Synthetic data, not a real dataset** | **Absolute metre figures are meaningless**; only **trends** and **relative differences between configurations** are meaningful |
| 2 | ~~TC (tight coupling) not implemented~~ **now added (Experiment C2)** | see below |
| 3 | Accelerometer bias states omitted | simplification; may make results optimistic |
| 4 | **2D planar model** (Experiment B adds yaw) | no ECEF/ENU transforms, no full attitude estimation |
| 5 | Simulated IMU at 1 Hz | real IMUs are typically 100 Hz |
| 6 | Experiment C2's constellation is idealized | 8 satellites with fixed azimuth/elevation, static geometry (GDOP 1.019 is optimistic) |
| 7 | Experiment C2 implements no tightly-coupled EKF | the point is a property of FGO itself; a second estimator is unnecessary |
| 8 | Robust-kernel gains depend on δ | C1: Cauchy positive throughout but varying by 30 percentage points; Huber vanishes for δ > 5 |

### On limitation 2 (resolved)

This originally read: "what is implemented is LC, not the paper's TC, and all models are linear; **this may be exactly why FGO's large advantage did not reproduce**." Experiment C2 added TC, with the conclusion:

**The real contribution of tight coupling is redundancy at short windows, not line-of-sight nonlinearity.**
At K=2 the loosely-coupled setup has 8 unknowns / 8 equations (exactly determined, robust kernels provably useless), while the tightly-coupled setup has 10 unknowns / 21 equations (redundancy 11) and robust kernels immediately gain 1.7–1.8%.

### On limitation 1

The advantage of synthetic data is that the **ground truth is 100% accurate and the noise characteristics are fully controllable**, so the causal chain "window size → accuracy" is clean. In real datasets the ground truth itself carries 5–10 cm of error and the noise characteristics drift over time, masking the effect one wants to observe.

**Understanding the mechanism in simulation first and moving to real data afterwards is the usual research order.** The next step is a real-data version on the [UrbanNav](https://github.com/weisongwen/UrbanNavDataset) dataset.

### Which conclusions do **not** depend on the data being real

| Conclusion | Data-dependent? |
|---|---|
| Experiment 1: iteration count has no effect | No — it is a **mathematical property** (quadratic cost ⇒ Newton lands in one step) |
| C2: still no effect after switching to a nonlinear model | No — same as above, and corroborated by two independent observation models |
| K=1/K=2 exactly determined → position = raw GNSS | No — this is **constraint counting**, pure algebra |
| C2b: tight coupling has redundancy at K=2 → robust kernels revive | No — this is **constraint counting** |
| C3: more iterations is worse | Not for the **direction** (30/30) — but the **magnitude** is data-dependent |
| Experiment 4's mechanism: EKF adapts via covariance, FGO uses fixed weights | Largely no — this is a **structural** difference |
| The dimensional observation on Eq. (22) | No — it concerns the paper's **internal** consistency |
| **The specific metre figures** | **Yes, entirely** — they change with the data |

---

## 8. The paper reproduced

> Wen, W., Pfeifer, T., Bai, X., & Hsu, L.-T. (2021).
> **Factor graph optimization for GNSS/INS integration: A comparison with the extended Kalman filter.**
> *NAVIGATION: Journal of the Institute of Navigation*, 68(2), 315–331.
> https://doi.org/10.1002/navi.421

Related work:

- Hsu, L.-T., Kubo, N., Wen, W., Chen, W., Liu, Z., Suzuki, T., & Meguro, J. (2021).
  UrbanNav: An open-sourced multisensory dataset for benchmarking positioning algorithms designed for urban areas.
  *ION GNSS+ 2021*, 226–256.

**This repository is an independent reproduction and is not affiliated with the paper's authors.** Observations about the original text are technical discussion only; corrections are welcome.

---

## 9. License

MIT License, see [LICENSE](LICENSE).
