# Code and data

Everything needed to regenerate every figure and every number quoted in the
manuscript. Python 3 with `numpy`, `scipy`, `pandas`, `matplotlib`.

Scripts write into the working directory, so run them from a directory that
also holds the CSVs in `../data/` (or copy the CSVs alongside).

---

## Figures

| Figure | File | Generator | Input | Time |
|---|---|---|---|---|
| Fig. 1 | `fig1.png` | `make_paper_figures.py` | `collapse_data.csv`, `collapse_n2.csv`, `exponents.csv` | seconds |
| Fig. 2 | `fig2.png` | `make_transition_figure.py` | `transition_data.npz` (cached) | seconds, or ~10 min if the cache is deleted |
| Fig. 3 | `fig3.png` | `make_levy_figure.py` | hardcoded measurements, see below | seconds |
| Fig. S1 | `figS1.png` | `make_paper_figures.py` | same as Fig. 1 | seconds |
| Fig. S2 | `figS2.png` | `make_alpha2_figure.py` | `alpha2_results.csv` | seconds |

All five have been checked to regenerate from the files shipped here.

Two generators plot values that are hardcoded rather than read from a file,
because the simulations behind them take about eighty minutes in total and
the figures are otherwise redrawn in seconds. Those values, and the scripts
that produce them, are listed under *Measurements* below. Deleting
`transition_data.npz` forces Fig. 2 to re-simulate.

## Data

| File | Produced by | Feeds |
|---|---|---|
| `collapse_data.csv`, `collapse_n2.csv`, `exponents.csv` | `collapse_figure.py` | Fig. 1, Fig. S1 |
| `alpha2_results.csv`, `alpha2_slopes.csv` | `alpha2_consistency.py` | Fig. S2 |
| `transition_data.npz` | `make_transition_figure.py` | Fig. 2 |

## Measurements

Each script prints a table; the numbers quoted in the manuscript come from
the runs described here. None of them takes input files.

| Script | Establishes | Time |
|---|---|---|
| `collapse_figure.py` | the exponents of Table I across twenty-four cells | ~20 min |
| `alpha2_consistency.py` | recovery of the Brownian limit at $\alpha=2$ | ~40 min |
| `factor_check.py` | the prefactor of Eq. (1) against prescribed gaps, all three classes | ~5 min |
| `test_subharmonic.py` | the stationary tail index for $\gamma<2$ | ~50 min |
| `rerun_steep.py` | the same for $\gamma\ge2$, with the exact-density calibration | ~30 min |
| `nested_window_check.py` | that the $\gamma=4$, $\alpha=1.5$ point is window-limited | ~2 min |
| `crossover_check.py` | the crossover function and the constant on the critical line | ~10 min |
| `fixed_eta_scan.py` | $m/\mathcal{H}\to1/2$ at fixed $\eta$ as $N$ grows | ~25 min |
| `make_transition_figure.py` | the crossing of $Y_2$ at $\alpha=\beta$ | ~10 min |

The measurements hardcoded into `make_levy_figure.py` come from
`test_subharmonic.py`, `rerun_steep.py` and `nested_window_check.py`; their
provenance is annotated in that file.

## Diagnostics kept for the record

These are not needed to reproduce the paper. They are the checks that
decided what the paper says, and several of them decided against an earlier
claim; they are kept so the reasoning can be audited.

| Script | What it settled |
|---|---|
| `calibrate_pd.py` | that the estimator returns $w=1$ on laws that really are Poisson--Dirichlet, so $w<1$ under stable noise is a property of the model |
| `pw_distribution.py` | the distribution of the winner's own winning probability, showing the atom and the dust directly |
| `global_scan.py` | that skewness, value--noise coupling and heterogeneous amplitudes act only through $NC_\alpha$ |
| `dust_fraction.py` | that the atom/dust separation fails as $\alpha\to2$, which is why the earlier $m\sim\mathcal{H}^{\alpha}$ claim was withdrawn |
| `test_spike_formula.py` | the accidental-winner mass against $2C_\alpha\sigma^\alpha T_\alpha$ |
| `wordcount.py` | the PRL length estimate, by environment |

## A caution about `wordcount.py`

It strips inline math with a regex that will not cross a blank line. An
earlier version could, and silently swallowed whole paragraphs, under-counting
the manuscript by about a thousand words for several revisions. The figure
allowance also assumes a fixed aspect ratio and must be recomputed by hand
when a figure of a different shape is added. Treat its output as an estimate
and confirm the length by compiling.
