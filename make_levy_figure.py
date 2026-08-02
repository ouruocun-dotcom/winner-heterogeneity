"""
fig3: the Levy gas in a potential U(x) = |x|^gamma / gamma.

(a) Measured stationary tail index against the prediction beta = gamma+alpha-2.
(b) theta = (gamma-2)/(gamma+alpha-2) versus gamma for three alpha, all
    crossing zero at gamma=2: the harmonic well is critical whatever the noise.

Measurements come from code/test_subharmonic.py (gamma<2), code/rerun_steep.py
(gamma>=2) and code/nested_window_check.py (the gamma=4, alpha=1.5 point).
They are hardcoded so the figure redraws in seconds instead of ~80 minutes.

WHAT CHANGED FROM THE FIRST VERSION
The sharpest test in the paper was invisible in the old figure.  At gamma=4,
alpha=1 the stationary density is known exactly, and fitting that exact
density over the same window used for the simulation gives 3.110, not the
asymptotic 3.000: at large beta the tail is reachable only at moderate x,
where subleading corrections survive, so ANY finite-window fit reads high.
The measurement, 3.111, reproduces 3.110 -- agreement to 0.03% -- but plotted
against the asymptote alone it looks like a 4% discrepancy.  Panel (a) now
marks that window-matched value and joins it to the measurement, with an
inset on its own scale, so the comparison the text makes is the one the eye
makes.  Points still descending as the fitting window is pushed outward keep
their arrows.

Output: fig4.png
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# gamma, alpha, measured tail index, status
#   'ok'      converged
#   'window'  window-limited, still descending as the window is pushed out
#   'exact'   closed-form stationary density available (see EXACT_WINDOW)
DATA = [
    (1.4, 1.0, 0.396, 'ok'),
    (1.5, 1.0, 0.494, 'ok'),
    (1.6, 1.0, 0.596, 'ok'),
    (1.8, 1.0, 0.791, 'ok'),
    (1.2, 1.5, 0.745, 'ok'),
    (1.6, 1.5, 1.124, 'ok'),
    (2.0, 1.0, 1.010, 'exact'),
    (2.0, 1.5, 1.554, 'exact'),
    (3.0, 0.5, 1.527, 'ok'),
    (3.0, 1.0, 2.109, 'ok'),
    (4.0, 1.0, 3.111, 'exact'),
    (4.0, 1.5, 3.722, 'window'),
]

# (gamma, alpha) -> value from fitting the EXACT stationary density over the
# same window as the simulation.  At gamma=2 the stationary law is stable of
# index alpha with no window correction, so the reference is beta itself.
EXACT_WINDOW = {(4.0, 1.0): 3.110, (2.0, 1.0): 1.000, (2.0, 1.5): 1.500}

ALPHAS = [0.5, 1.0, 1.5]
cm = plt.get_cmap('viridis')
col = {a: cm(i/(len(ALPHAS) - 1)) for i, a in enumerate(ALPHAS)}
mk = {0.5: 's', 1.0: 'o', 1.5: '^'}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

# ---------------------------------------------------------------- panel (a)
ax = axes[0]
lim = (0.2, 4.2)
ax.plot(lim, lim, 'k--', lw=1.0, zorder=1, label=r'$y=x$')

for a in ALPHAS:
    pts = [(g + a - 2, b, st) for g, aa, b, st in DATA if aa == a]
    if not pts:
        continue
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=52, marker=mk[a],
               facecolors='none', edgecolors=col[a], linewidths=1.7, zorder=4,
               label=rf'$\alpha={a}$')
    for x, y, st in pts:
        if st == 'window':
            ax.annotate('', xy=(x, x + 0.06), xytext=(x, y),
                        arrowprops=dict(arrowstyle='->', color=col[a],
                                        lw=1.3, shrinkA=2, shrinkB=0), zorder=2)

# window-matched exact reference: what a finite-window fit of the EXACT
# density returns, i.e. the number the measurement should reproduce
ref_x, ref_y = [], []
for (g, a), v in EXACT_WINDOW.items():
    if abs(v - (g + a - 2)) < 1e-9:
        continue                                  # no window correction
    ref_x.append(g + a - 2); ref_y.append(v)
if ref_x:
    ax.scatter(ref_x, ref_y, s=95, marker='+', color='k', linewidths=1.4,
               zorder=5, label='exact law, same window')

ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel(r'predicted  $\beta=\gamma+\alpha-2$', fontsize=11)
ax.set_ylabel('measured tail index', fontsize=11)
ax.set_title('(a) stationary tail of the Lévy gas',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper left', frameon=False)
ax.grid(alpha=0.25, lw=0.4)
ax.set_aspect('equal', adjustable='box')

# inset: the one case with a closed form, on its own scale
axin = ax.inset_axes([0.60, 0.12, 0.36, 0.30])
axin.axhline(3.000, color='0.6', lw=1.0, ls='--')
axin.axhline(3.110, color='k', lw=1.2)
axin.scatter([0], [3.111], s=60, marker='o', facecolors='none',
             edgecolors=col[1.0], linewidths=1.8, zorder=4)
axin.set_xlim(-1, 1); axin.set_ylim(2.95, 3.20)
axin.set_xticks([])
axin.set_yticks([3.00, 3.11])
axin.set_yticklabels(['3.000 asymptote', '3.110 same window'], fontsize=6)
axin.tick_params(length=2, pad=1)
axin.set_title(r'$\gamma=4,\ \alpha=1$', fontsize=7, pad=2)
for sp in axin.spines.values():
    sp.set_linewidth(0.6)

# ---------------------------------------------------------------- panel (b)
ax = axes[1]
for a in ALPHAS:
    g = np.linspace(2 - a + 1e-3, 4.2, 800)
    ax.plot(g, (g - 2)/(g + a - 2), lw=1.9, color=col[a], label=rf'$\alpha={a}$')
    ax.axvline(2 - a, ls=':', lw=1.0, color=col[a], alpha=0.8)
for g, a, b, st in DATA:
    ax.scatter([g], [(g - 2)/b], s=52, marker=mk[a], facecolors='none',
               edgecolors=col[a], linewidths=1.7, zorder=4)
ax.axhline(0, color='k', lw=0.9, alpha=0.6)
ax.axvline(2, color='k', lw=0.9, alpha=0.6)
ax.plot([2], [0], 'k*', ms=13, zorder=5)
ax.annotate('harmonic well\n' + r'$\gamma=2$: $\theta=0$ for every $\alpha$',
            xy=(2, 0), xytext=(2.35, -1.15), fontsize=8.5,
            arrowprops=dict(arrowstyle='->', lw=1.0, color='k'))
ax.text(0.62, 0.62, 'stationary state\nceases to exist\n' + r'($\beta<0$)',
        fontsize=7.6, color='0.35', ha='center')
ax.set_xlim(0.35, 4.2); ax.set_ylim(-2.6, 0.75)
ax.set_xlabel(r'potential exponent  $\gamma$', fontsize=11)
ax.set_ylabel(r'$\theta=(\gamma-2)/(\gamma+\alpha-2)$', fontsize=11)
ax.set_title(r'(b) harmonic well is critical for every $\alpha$',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=8.5, loc='lower right', frameon=False)
ax.grid(alpha=0.25, lw=0.4)

plt.tight_layout()
plt.savefig('fig3.png', dpi=300, bbox_inches='tight')
print('saved fig3.png')

print('\nresiduals (measured - predicted):')
for g, a, b, st in DATA:
    p = g + a - 2
    tag = {'ok': '', 'window': '   window-limited',
           'exact': '   exact law available'}[st]
    print(f'  gamma={g:4.1f} alpha={a:4.1f}  pred {p:6.3f}  meas {b:6.3f}'
          f'  dev {b - p:+.3f}{tag}')
conv = [b - (g + a - 2) for g, a, b, st in DATA if st != 'window']
print(f'\nmean |dev| over the {len(conv)} converged points: '
      f'{np.mean(np.abs(conv)):.3f}')
print('against the window-matched exact values:')
for (g, a), v in EXACT_WINDOW.items():
    meas = [b for gg, aa, b, _ in DATA if gg == g and aa == a][0]
    print(f'  gamma={g}, alpha={a}: measured {meas:.3f} vs {v:.3f}'
          f'  ({100*abs(meas - v)/v:.2f}%)')
