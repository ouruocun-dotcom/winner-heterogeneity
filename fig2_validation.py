"""
Figure 2: Universal experimental validation of d = sqrt(2*H).

(a) Observed distance d vs winner heterogeneity H for 102 microbial
    communities from two independent experimental systems:
    - Soil bacteria pairs and trios (Meroz et al., Nat Commun 2021)
    - Budding yeast ~300-strain pooled competition (Khristich et al., bioRxiv 2025)
(b) Temporal evolution of mean replicate distance for UNANIMOUS vs
    CONTESTED bacterial communities over ~400 generations.

Requires data files in ./data/:
    meroz2021_counts.xlsx
    khristich2025_barcode_counts.csv

Output: fig2_validation.png
Runtime: ~30 seconds
Dependencies: numpy, matplotlib, pandas, scipy, openpyxl (pip install openpyxl)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter
from scipy.stats import spearmanr

# ================================================================
# LOAD AND PROCESS BACTERIAL DATA (Meroz et al. 2021)
# ================================================================
print("Loading bacterial data...")
df_bac = pd.read_excel('data/meroz2021_counts.xlsx')
sp_all = ['Ea','Pa','Pch','Pci','Pv','Pf','Sm','Ab',
          'H77','H79','H82','H97','Fj','IN63','IN65','IN72']

bac_data = []
for kind in ['Pair', 'Trio']:
    sub_kind = df_bac[df_bac['sample_kind'] == kind]
    for sample in sorted(sub_kind['sample'].unique()):
        sub = sub_kind[sub_kind['sample'] == sample]
        sp = [s for s in sp_all if sub[s].sum() > 0]
        if len(sp) < 2:
            continue
        final = sub[sub['transfer'] == 38]
        if len(final) < 3:
            continue

        fracs, winners = [], []
        for _, row in final.iterrows():
            c = np.array([row[s] for s in sp], dtype=float)
            t = c.sum()
            if t > 0:
                fracs.append(c / t)
                winners.append(sp[np.argmax(c / t)])
        if len(fracs) < 3:
            continue

        fa = np.array(fracs)
        K = len(fa)
        dists = [np.linalg.norm(fa[i] - fa[j])
                 for i, j in combinations(range(K), 2)]
        dist_obs = np.mean(dists)

        P_emp = np.array([winners.count(s) / K for s in set(winners)])
        het = 1 - np.sum(P_emp ** 2)

        bac_data.append(dict(sample=sample, kind=kind,
                             het=het, dist=dist_obs, K=K))

bac_df = pd.DataFrame(bac_data)
pairs = bac_df[bac_df['kind'] == 'Pair']
trios = bac_df[bac_df['kind'] == 'Trio']
print(f"  Bacteria pairs: {len(pairs)}, trios: {len(trios)}")

# ================================================================
# LOAD AND PROCESS YEAST DATA (Khristich et al. 2025)
# ================================================================
print("Loading yeast data...")
df_yeast = pd.read_csv('data/khristich2025_barcode_counts.csv')
df_yeast = df_yeast.dropna(subset=['replicate', 'environment', 'timepoint'])
df_yeast['replicate'] = df_yeast['replicate'].astype(str)

yeast_envs = ['M3-37', 'M3-ETH', 'SC', 'SC-PRO', 'SC-SDS']
yeast_data = []

for env in yeast_envs:
    sub = df_yeast[df_yeast['environment'] == env]
    reps = [r for r in sub['replicate'].unique()
            if '-C' not in r and r != 'R0']
    tps = sub['timepoint'].unique()
    tp_nums = {t: int(t[1:]) for t in tps
               if t.startswith('T') and t[1:].isdigit()}
    if not tp_nums:
        continue
    tp_late = max(tp_nums, key=tp_nums.get)

    freq_vectors = []
    all_bc = set()
    for rep in reps:
        s = sub[(sub['replicate'] == rep) & (sub['timepoint'] == tp_late)]
        if len(s) == 0:
            continue
        total = s['count'].sum()
        if total < 1000:
            continue
        bc_fracs = dict(zip(s['Barcode_ID'], s['count'] / total))
        all_bc.update(bc_fracs.keys())
        freq_vectors.append((rep, bc_fracs))

    if len(freq_vectors) < 5:
        continue
    bc_list = sorted(all_bc)
    n_reps = len(freq_vectors)
    fmat = np.zeros((n_reps, len(bc_list)))
    for i, (rep, bc_fracs) in enumerate(freq_vectors):
        for j, bc in enumerate(bc_list):
            fmat[i, j] = bc_fracs.get(bc, 0)

    winners = [bc_list[np.argmax(fmat[i])] for i in range(n_reps)]
    wc = Counter(winners)
    P_emp = np.array([wc[w] / n_reps for w in wc])
    het = 1 - np.sum(P_emp ** 2)
    dists = [np.linalg.norm(fmat[i] - fmat[j])
             for i, j in combinations(range(n_reps), 2)]
    dist_obs = np.mean(dists)

    yeast_data.append(dict(env=env, het=het, dist=dist_obs, K=n_reps))

yeast_df = pd.DataFrame(yeast_data)
print(f"  Yeast environments: {len(yeast_df)}")

# ================================================================
# COMPUTE TEMPORAL DYNAMICS (Meroz data)
# ================================================================
print("Computing temporal dynamics...")
transfers = sorted(df_bac['transfer'].unique())
gens = {t: t * 10.5 for t in transfers}

unanim_dists = {t: [] for t in transfers}
contest_dists = {t: [] for t in transfers}

for _, row_info in bac_df.iterrows():
    sample = row_info['sample']
    sub = df_bac[(df_bac['sample'] == sample) &
                 (df_bac['sample_kind'].isin(['Pair', 'Trio']))]
    sp = [s for s in sp_all if sub[s].sum() > 0]
    if len(sp) < 2:
        continue
    is_unanimous = row_info['het'] < 0.1

    for t in transfers:
        t_data = sub[sub['transfer'] == t]
        if len(t_data) < 3:
            continue
        fracs = []
        for _, row in t_data.iterrows():
            c = np.array([row[s] for s in sp], dtype=float)
            total = c.sum()
            if total > 0:
                fracs.append(c / total)
        if len(fracs) < 3:
            continue
        fa = np.array(fracs)
        d = np.mean([np.linalg.norm(fa[i] - fa[j])
                     for i, j in combinations(range(len(fa)), 2)])
        if is_unanimous:
            unanim_dists[t].append(d)
        else:
            contest_dists[t].append(d)

# ================================================================
# FIGURE 2
# ================================================================
print("Generating figure...")
fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

# --- Panel (a): Universal d vs H ---
ax = axes[0]
H_th = np.linspace(0, 0.75, 100)
d_th = np.sqrt(2 * H_th)
ax.plot(H_th, d_th, 'k-', lw=2, alpha=0.5, zorder=0,
        label=r'$d = \sqrt{2\mathcal{H}}$')

ax.scatter(pairs['het'], pairs['dist'], s=20, c='#2166ac', marker='o',
           alpha=0.6, edgecolors='none', zorder=2,
           label=f'Bacteria pairs (n={len(pairs)})')
ax.scatter(trios['het'], trios['dist'], s=25, c='#4393c3', marker='^',
           alpha=0.6, edgecolors='none', zorder=2,
           label=f'Bacteria trios (n={len(trios)})')
ax.scatter(yeast_df['het'], yeast_df['dist'], s=80, c='#d6604d',
           marker='s', alpha=0.9, edgecolors='k', linewidth=0.5,
           zorder=3, label=f'Yeast environments (n={len(yeast_df)})')

for _, row in yeast_df.iterrows():
    ax.annotate(row['env'], (row['het'], row['dist']), fontsize=5.5,
                xytext=(4, 4), textcoords='offset points', color='#b2182b')

r_p, p_p = spearmanr(pairs['het'], pairs['dist'])
r_t, p_t = spearmanr(trios['het'], trios['dist'])
r_y, p_y = spearmanr(yeast_df['het'], yeast_df['dist'])

ax.text(0.03, 0.97,
        f'Bacteria pairs: $r_s$={r_p:.2f}\n'
        f'Bacteria trios: $r_s$={r_t:.2f}\n'
        f'Yeast: $r_s$={r_y:.2f}',
        transform=ax.transAxes, fontsize=7, va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

ax.set_xlabel(r'Winner heterogeneity $\mathcal{H} = 1 - \sum P_i^2$',
              fontsize=10)
ax.set_ylabel(r'Mean replicate distance $d$', fontsize=10)
ax.set_title('(a) Universal validation', fontsize=10, fontweight='bold')
ax.legend(fontsize=6.5, loc='lower right', framealpha=0.9)
ax.set_xlim(-0.03, 0.78)
ax.set_ylim(-0.03, 1.1)
ax.tick_params(labelsize=8)

# --- Panel (b): Temporal dynamics ---
ax = axes[1]
t_u, d_u, e_u = [], [], []
t_c, d_c, e_c = [], [], []
for t in sorted(transfers):
    if t == 0:
        continue
    if unanim_dists[t]:
        t_u.append(gens[t])
        d_u.append(np.mean(unanim_dists[t]))
        e_u.append(np.std(unanim_dists[t]) / np.sqrt(len(unanim_dists[t])))
    if contest_dists[t]:
        t_c.append(gens[t])
        d_c.append(np.mean(contest_dists[t]))
        e_c.append(np.std(contest_dists[t]) / np.sqrt(len(contest_dists[t])))

ax.errorbar(t_u, d_u, yerr=e_u, fmt='o-', color='#2166ac', ms=4,
            lw=1.5, capsize=2,
            label=r'Unanimous ($\mathcal{H}$=0)')
ax.errorbar(t_c, d_c, yerr=e_c, fmt='s-', color='#d6604d', ms=4,
            lw=1.5, capsize=2,
            label=r'Contested ($\mathcal{H}$>0)')

ax.set_xlabel('Generations', fontsize=10)
ax.set_ylabel(r'Mean replicate distance $d$', fontsize=10)
ax.set_title('(b) Temporal dynamics', fontsize=10, fontweight='bold')
ax.legend(fontsize=7, loc='upper left')
ax.set_xlim(0, 420)
ax.set_ylim(0, 0.7)
ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig('fig2_validation.png', dpi=300, bbox_inches='tight')
plt.close()

# ================================================================
# PRINT STATISTICS FOR PAPER
# ================================================================
print(f"\nSaved: fig2_validation.png")
print(f"\n=== PAPER STATISTICS ===")
print(f"Bacteria pairs: n={len(pairs)}, Spearman r={r_p:.3f}, p={p_p:.2e}")
print(f"Bacteria trios: n={len(trios)}, Spearman r={r_t:.3f}, p={p_t:.2e}")
print(f"Yeast envs:     n={len(yeast_df)}, Spearman r={r_y:.3f}, p={p_y:.4f}")
print(f"Total communities: {len(pairs) + len(trios) + len(yeast_df)}")
