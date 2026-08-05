"""Shared figure style: PRL single-column, no downscaling.

The manuscript inserts figures at \columnwidth = 3.375 in.  Building them at
7-11 in and letting LaTeX shrink them by 0.31-0.48 rendered every label at
1.8-4.8 pt, well below legibility.  Everything here is built at final size,
so the scale factor is 1 and the numbers below are what the reader sees.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COL = 3.375          # PRL single-column width, inches
BASE, SMALL, TINY = 8.0, 7.0, 6.5

def use():
    plt.rcParams.update({
        'font.size': BASE,
        'axes.labelsize': BASE,
        'axes.titlesize': BASE,
        'xtick.labelsize': SMALL,
        'ytick.labelsize': SMALL,
        'legend.fontsize': TINY,
        'lines.linewidth': 1.2,
        'lines.markersize': 3.2,
        'axes.linewidth': 0.7,
        'xtick.major.width': 0.7,
        'ytick.major.width': 0.7,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'legend.frameon': False,
        'legend.handlelength': 1.4,
        'legend.handletextpad': 0.5,
        'legend.labelspacing': 0.25,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        # NOT 'tight': tight cropping moves artists after the layout is
        # fixed, so a collision check run before saving does not describe the
        # file that gets written.  Fixed margins keep the two identical.
        'savefig.bbox': 'standard',
    })
