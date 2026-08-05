# Code and data

**Every script here runs on its own.** Open a Colab notebook, paste one file
into a cell, press run. Nothing to install, nothing to configure, no files to
upload — the scripts that need data download it from this repository
automatically.

Each one prints a table and, where relevant, saves a figure. Every number and
every figure in the paper comes from exactly one of these.

---

## The three figures in the paper

| | Paste this file | Prints | Takes |
|---|---|---|---|
| **Fig. 1** the coupled exponent | `make_paper_figures.py` | measured vs predicted θ across 24 cells | seconds |
| **Fig. 2** the condensation transition | `make_transition_figure.py` | the crossing value at α=β against the exact 0.5000 | seconds |
| **Fig. 3** the Lévy gas | `make_levy_figure.py` | residuals against β = γ+α−2 | seconds |

Two more figures live in the Supplemental Material:

| | Paste this file | Takes |
|---|---|---|
| **Fig. S1** validation of Eq. (1) | `make_paper_figures.py` (same run as Fig. 1) | seconds |
| **Fig. S2** recovery of the Brownian limit | `make_alpha2_figure.py` | seconds |

These are fast because they plot measurements that are already in hand. The
simulations behind them are the scripts in the next section, and those take
minutes to an hour. `make_transition_figure.py` reads a cached
`transition_data.npz`; delete it and the script re-simulates from scratch in
about ten minutes.

## Reproducing the measurements

Same instructions: paste, run, read the table. None of these needs any input
file.

| Paste this file | What it establishes | Takes |
|---|---|---|
| `collapse_figure.py` | the exponents of Table I, 24 cells spanning θ from −2 to +1.5 | ~20 min |
| `alpha2_consistency.py` | that the Brownian limit is recovered: flat to 1% at α=2 | ~40 min |
| `factor_check.py` | the prefactor of Eq. (1) against gaps where **H** is computable exactly | ~5 min |
| `test_subharmonic.py` | the stationary tail index for γ<2 | ~50 min |
| `rerun_steep.py` | the same for γ≥2, calibrated against the exact γ=4, α=1 density | ~30 min |
| `nested_window_check.py` | that the γ=4, α=1.5 point is window-limited, not discrepant | ~2 min |
| `crossover_check.py` | the crossover function, and 2B/(A+2B) on the critical line | ~10 min |
| `fixed_eta_scan.py` | that m/**H** → 1/2 at fixed η as N grows | ~25 min |

`make_levy_figure.py` plots values produced by `test_subharmonic.py`,
`rerun_steep.py` and `nested_window_check.py`; each is annotated with its
source inside that file.

## What decided the paper's claims

Not needed to reproduce anything. These are the checks that determined what
the paper says, including several that ruled out things earlier drafts
claimed. They are kept so the reasoning can be audited rather than taken on
trust.

| Paste this file | What it settled |
|---|---|
| `calibrate_pd.py` | that the estimator returns w = 1 on laws that genuinely are Poisson–Dirichlet — so w < 1 under stable noise is the model, not the tool |
| `pw_distribution.py` | the distribution of the winner's own winning probability, showing the atom and the dust directly |
| `global_scan.py` | that skewness, value–noise coupling and heterogeneous amplitudes all act only through N·C_α |
| `dust_fraction.py` | that the atom/dust separation fails as α → 2 — which is why an earlier claim, m ~ **H**^α, was withdrawn |
| `test_spike_formula.py` | the accidental-winner mass against C_α σ^α T_α |

## Utilities

| File | Use |
|---|---|
| `CHECKLIST.md` | seven steps to run before shipping a figure; each exists because skipping it let a defect reach the manuscript |
| `_figstyle.py` | PRL single-column geometry and font sizes; figures are built at final size so the point sizes in the code are what the reader sees |
| `_place.py` | rasterises a panel and reports which anchors are free of ink, so panel labels are placed rather than guessed |
| `_collide.py` | flags text over text, over tick labels, and over data; lines are densely resampled so a legend lying on a curve is caught |
| `_clip.py` | measures ink at the edges of a *saved* figure, which is how content cut off by fixed margins is found |
| `check_consistency.py` | run after editing the manuscript or Supplemental Material: titles, labels, citations, figure files, and the SM's hardcoded pointers into the main text |
| `wordcount.py` | PRL length estimate, broken down by environment |

The four figure tools are used together, in the order set out in
`CHECKLIST.md`. Two of them exist because an earlier check gave the wrong
answer rather than no answer: `_collide.py` originally sampled only marker
vertices and passed figures whose legends sat on the curves, and the clipping
test was originally run on the in-memory figure, which hands back the wrong
object when a script builds more than one.

`wordcount.py` is an estimate, not a verdict. It strips inline math with a
regex that will not cross a blank line -- an earlier version could, and
silently swallowed whole paragraphs, under-counting by about a thousand words
for several revisions. Its figure allowance also assumes a fixed aspect ratio
and needs recomputing when a figure of a different shape is added. Confirm the
length by compiling.

## Data

Downloaded automatically when needed; here for completeness.

| File | Produced by | Used by |
|---|---|---|
| `collapse_data.csv`, `collapse_n2.csv`, `exponents.csv` | `collapse_figure.py` | Figs. 1 and S1 |
| `alpha2_results.csv`, `alpha2_slopes.csv` | `alpha2_consistency.py` | Fig. S2 |
| `transition_data.npz` | `make_transition_figure.py` | Fig. 2 |

## Requirements

`numpy`, `scipy`, `pandas`, `matplotlib` — all preinstalled in Colab. Nothing
else.
