"""Find an empty corner for a panel label, instead of guessing one.

Rasterises each axes and reports which of nine candidate anchors is free of
dark ink under a label-sized box.  Light gridlines are ignored: the label sits
on a white patch, so only curves, markers and text matter.
"""
import re, sys, numpy as np, matplotlib
matplotlib.use('Agg')
src = re.sub(r"plt\.savefig\([^)]*\)", "", open(sys.argv[1]).read())
ns = {}; exec(src, ns)
fig = [v for v in ns.values() if hasattr(v, 'canvas')][0]
fig.canvas.draw()
img = np.asarray(fig.canvas.buffer_rgba())[..., :3].mean(axis=2)
H = img.shape[0]
for k, ax in enumerate(fig.get_axes()):
    bb = ax.get_window_extent()
    free = []
    for fy in (0.97, 0.60, 0.22):
        for fx in (0.03, 0.42, 0.80):
            px = bb.x0 + fx*bb.width
            py = bb.y0 + fy*bb.height
            x0, x1 = int(px - 2), int(px + 0.14*bb.width)
            y0, y1 = int(H - py), int(H - py + 0.16*bb.height)
            patch = img[max(y0,0):y1, max(x0,0):x1]
            if patch.size and (patch < 150).mean() < 0.005:
                free.append((fx, fy))
    print(f"  ax{k}: empty anchors {free if free else 'NONE'}")
