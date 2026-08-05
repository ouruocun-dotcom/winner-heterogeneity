"""
fig5: the condensation transition of the winner measure.

(a) Y_2 = Pr[two replicas pick the same winner] = weight of the condensate,
    against alpha/beta for four population sizes.  Curves for different N
    cross at alpha = beta: above it the condensate fills the measure as N
    grows, below it evaporates.  The crossing value is exact.

(b) The same data against the scaling variable eta_N, with the closed-form
    scaling function.  That function depends on alpha/beta as well as eta,
    so the band between its values at the two extreme ratios is drawn; the
    spread of the points is that dependence, not scatter.

Noise is Cauchy (alpha = 1) and beta is varied, so alpha/beta = 1/beta and
the crossing sits at beta = 1.  sigma is chosen so the crossing lands near
Y_2 = 1/2, which fixes where the curves meet, not that they meet.

LAYOUT NOTE
An earlier version put (b) as an inset inside (a).  The star annotation was
then drawn underneath it and the two phase labels sat on top of the curves,
making three separate collisions.  Side-by-side panels remove all of them,
and cost fewer words under the PRL figure allowance as well.  All text is
placed in headroom created for it, never over data.

Runtime ~10 min, or seconds if transition_data.npz already exists
(delete it to force a re-run).
"""
# ---------------------------------------------------------------- data access
# Looks for the file locally first, then downloads it from the repository, so
# the script runs as-is in a fresh Colab cell with nothing uploaded.
REPO = ('https://raw.githubusercontent.com/'
        'ouruocun-dotcom/winner-heterogeneity/{branch}/data/')


def fetch(name):
    import os
    import urllib.request
    for p in (name, f'data/{name}', f'../data/{name}', f'/content/{name}'):
        if os.path.exists(p):
            return p
    err = None
    for branch in ('main', 'master'):
        try:
            urllib.request.urlretrieve(REPO.format(branch=branch) + name, name)
            return name
        except Exception as e:
            err = e
    raise FileNotFoundError(
        f"{name} is not here and could not be downloaded ({err}).\n"
        f"Fetch it from {REPO.format(branch='main') + name} and put it in "
        f"this directory, or upload it with the folder icon in the Colab "
        f"sidebar.")

import os
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma as G
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ALPHA = 1.0
SIGMA = 0.5*np.pi
BETAS = np.array([0.65, 0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3, 1.5, 1.7])
NS    = (50, 200, 800, 3200)
CONF  = 24000
SEED  = 20260804
CACHE = 'transition_data.npz'
B_AMP = G(ALPHA)*np.sin(np.pi*ALPHA/2)/np.pi * SIGMA**ALPHA


def cauchy(size, rng):
    return np.tan(rng.uniform(-np.pi/2, np.pi/2, size))


def Y2(N, be, rng, conf=CONF):
    blk = max(1, min(400, 3_000_000//N))
    nrep = max(1, conf//blk)
    agree = 0
    for _ in range(nrep):
        S = (1.0/rng.uniform(0, 1, (blk, N)))**(1.0/be)
        w1 = np.argmax(S + SIGMA*cauchy((blk, N), rng), axis=1)
        w2 = np.argmax(S + SIGMA*cauchy((blk, N), rng), axis=1)
        agree += np.count_nonzero(w1 == w2)
    n = nrep*blk
    p = agree/n
    return p, np.sqrt(max(p*(1 - p), 1e-9)/n)


def eta_of(N, be):
    return B_AMP*N**(1 - ALPHA/be)


def Y2_pred(eta, r):
    return quad(lambda u: np.exp(-u - 2*eta*u**r), 0, np.inf, limit=200)[0]


# ------------------------------------------------------------------ data
try:
    CACHE = fetch(CACHE)
except Exception:
    pass
if os.path.exists(CACHE):
    d = np.load(CACHE)
    Y, E = d['Y'], d['E']
    print(f'loaded {CACHE}')
else:
    rng = np.random.default_rng(SEED)
    Y = np.zeros((len(NS), len(BETAS)))
    E = np.zeros_like(Y)
    print(f'alpha={ALPHA}  sigma={SIGMA:.4f}  B={B_AMP:.4f}  (crossing at beta=1)')
    print(f"{'beta':>6}{'r':>6}" + ''.join(f"{'N='+str(n):>10}" for n in NS))
    for j, be in enumerate(BETAS):
        for i, N in enumerate(NS):
            Y[i, j], E[i, j] = Y2(N, be, rng)
        print(f'{be:6.2f}{ALPHA/be:6.2f}'
              + ''.join(f'{Y[i, j]:10.4f}' for i in range(len(NS))), flush=True)
    np.savez(CACHE, Y=Y, E=E, BETAS=BETAS, NS=np.array(NS))

R = ALPHA/BETAS
o = np.argsort(R)
cross = 1.0/(1.0 + 2*B_AMP)
cm = plt.get_cmap('viridis')
col = [cm(i/(len(NS) - 1)) for i in range(len(NS))]

import _figstyle
_figstyle.use()
fig, (axA, axB) = plt.subplots(1, 2, figsize=(_figstyle.COL, 1.55))

# ---- (a) the crossing
for i, N in enumerate(NS):
    axA.errorbar(R[o], Y[i][o], yerr=E[i][o], marker='o', ms=2.6, lw=1.0,
                 color=col[i], capsize=0, zorder=3, label=rf'${N}$')
axA.axvline(1.0, color='0.55', lw=0.7, zorder=1)
axA.plot([1.0], [cross], marker='*', ms=8, color='k', zorder=6)
axA.set_ylim(-0.03, 1.05)
axA.text(0.30, 0.88, 'fluid', fontsize=6.2, ha='center', color='0.35',
         transform=axA.transAxes)
axA.text(0.86, 0.30, 'condensed', fontsize=6.2, ha='center', color='0.35',
         transform=axA.transAxes)
axA.set_xlabel(r'$\alpha/\beta$')
axA.set_ylabel(r'$Y_2$  (condensate)')
axA.set_yticks([0, 0.5, 1.0])
axA.set_xticks([0.6, 1.0, 1.4])
axA.legend(title=r'$N$', title_fontsize=6.0, ncol=4, loc='lower left',
           bbox_to_anchor=(0.0, 1.02), borderpad=0.1, labelspacing=0.1,
           columnspacing=0.6, handlelength=0.9, handletextpad=0.35)
axA.grid(alpha=0.2, lw=0.3, zorder=0)
axA.text(0.80, 0.22, '(a)', transform=axA.transAxes, fontsize=7.5, fontweight='bold', va='top', ha='left',
        bbox=dict(fc='white', ec='none', alpha=0.75, pad=1.0), zorder=9)

# ---- (b) the collapse
gg = np.logspace(-2.4, 1.8, 80)
axB.fill_between(gg, [Y2_pred(g, R.max()) for g in gg],
                 [Y2_pred(g, R.min()) for g in gg], color='0.82', zorder=1)
axB.plot(gg, [Y2_pred(g, 1.0) for g in gg], 'k-', lw=1.1, zorder=2)
for i, N in enumerate(NS):
    axB.plot([eta_of(N, b) for b in BETAS], Y[i], 'o', ms=2.6,
             color=col[i], zorder=3)
axB.set_xscale('log')
axB.set_xlim(4e-3, 6e1); axB.set_ylim(-0.03, 1.05)
axB.set_yticks([0, 0.5, 1.0]); axB.set_yticklabels([])
axB.set_xticks([1e-2, 1e0])
axB.set_xlabel(r'$\eta_N$')
axB.grid(alpha=0.2, lw=0.3, zorder=0)
axB.text(0.80, 0.60, '(b)', transform=axB.transAxes, fontsize=7.5, fontweight='bold', va='top', ha='left',
        bbox=dict(fc='white', ec='none', alpha=0.75, pad=1.0), zorder=9)

fig.subplots_adjust(left=0.155, right=0.985, bottom=0.235, top=0.80, wspace=0.10)
plt.savefig('fig2.png')
print('saved fig2.png')
j1 = int(np.argmin(np.abs(BETAS - 1.0)))
print(f'crossing predicted {cross:.4f}; measured '
      + ', '.join(f'{Y[i][j1]:.4f}' for i in range(len(NS))))
