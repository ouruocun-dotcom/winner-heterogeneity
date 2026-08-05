"""Collision audit for a figure script.

Reports text over text, text over markers, and text over LINES.  The earlier
version sampled only marker vertices, so a legend sitting on top of a curve
passed as clean; lines are now densely resampled before the test.

Usage: python3 _collide.py make_xxx_figure.py
"""
import re
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')

src = re.sub(r"plt\.savefig\([^)]*\)", "", open(sys.argv[1]).read())
ns = {}
exec(src, ns)
fig = [v for v in ns.values() if hasattr(v, 'canvas')][0]
fig.canvas.draw()
rend = fig.canvas.get_renderer()
bad = 0

for k, ax in enumerate(fig.get_axes()):
    # a label drawn on an opaque patch stays readable over an image or a
    # curve, so it is exempt from the data test (but not from text-vs-text)
    texts, shielded = [], set()
    for t in ax.texts + [ax.title, ax.xaxis.label, ax.yaxis.label]:
        if not t.get_text().strip():
            continue
        lab = t.get_text()[:24].replace('\n', '/')
        texts.append((lab, t.get_window_extent(rend)))
        bb = t.get_bbox_patch()
        if bb is not None and bb.get_alpha() and bb.get_alpha() > 0.5:
            shielded.add(lab)
    lg = ax.get_legend()
    if lg:
        texts.append(('LEGEND', lg.get_window_extent(rend)))
    # matplotlib emits tick labels outside the axis limits; they are clipped
    # and never drawn, so counting them produces false collisions
    ab = ax.get_window_extent()

    def drawn(t, vertical):
        bb = t.get_window_extent(rend)
        c = 0.5*(bb.y0 + bb.y1) if vertical else 0.5*(bb.x0 + bb.x1)
        lo, hi = (ab.y0, ab.y1) if vertical else (ab.x0, ab.x1)
        return lo - 1 <= c <= hi + 1

    ticks = [(t.get_text(), t.get_window_extent(rend))
             for t in ax.get_xticklabels() if t.get_text().strip() and drawn(t, False)]
    ticks += [(t.get_text(), t.get_window_extent(rend))
              for t in ax.get_yticklabels() if t.get_text().strip() and drawn(t, True)]

    ink = []
    for ln in ax.get_lines():
        xy = np.column_stack(ln.get_data()).astype(float)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) < 2:
            ink += list(ax.transData.transform(xy)) if len(xy) else []
            continue
        # resample densely so a legend lying on a curve is detected
        d = ax.transData.transform(xy)
        seg = np.linspace(0, 1, 25)[:, None]
        dense = np.concatenate([d[i] + seg*(d[i+1] - d[i])
                                for i in range(len(d) - 1)])
        ink += list(dense)
    for c in ax.collections:
        try:
            ink += list(ax.transData.transform(c.get_offsets()))
        except Exception:
            pass
    ink = np.array(ink) if ink else np.zeros((0, 2))

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if texts[i][1].overlaps(texts[j][1]):
                print(f"  ax{k} TEXT/TEXT  '{texts[i][0]}' x '{texts[j][0]}'")
                bad += 1
    for lab, bb in texts:
        for tl, tb in ticks:
            if bb.overlaps(tb):
                print(f"  ax{k} TEXT/TICK  '{lab}' x '{tl}'")
                bad += 1
    for lab, bb in texts:
        if lab in shielded:
            continue
        hit = ((ink[:, 0] >= bb.x0) & (ink[:, 0] <= bb.x1) &
               (ink[:, 1] >= bb.y0) & (ink[:, 1] <= bb.y1)).sum() if len(ink) else 0
        if hit:
            print(f"  ax{k} TEXT/DATA  '{lab}' covers {hit} sampled points")
            bad += 1

# cross-axes pass: a colorbar label can collide with the next panel's y label,
# and an outside legend with the neighbouring title.  Per-axes loops miss both.
allt = []
for k, ax in enumerate(fig.get_axes()):
    for t in ax.texts + [ax.title, ax.xaxis.label, ax.yaxis.label]:
        if t.get_text().strip():
            allt.append((f'ax{k}:' + t.get_text()[:20].replace('\n', '/'),
                         t.get_window_extent(rend)))
    ab = ax.get_window_extent()
    for t, vert in ([(x, False) for x in ax.get_xticklabels()]
                    + [(y, True) for y in ax.get_yticklabels()]):
        if not t.get_text().strip():
            continue
        bb = t.get_window_extent(rend)
        c = 0.5*(bb.y0 + bb.y1) if vert else 0.5*(bb.x0 + bb.x1)
        lo, hi = (ab.y0, ab.y1) if vert else (ab.x0, ab.x1)
        if lo - 1 <= c <= hi + 1:
            allt.append((f'ax{k}:tick {t.get_text()[:12]}', bb))
    lg = ax.get_legend()
    if lg:
        allt.append((f'ax{k}:LEGEND', lg.get_window_extent(rend)))
for i in range(len(allt)):
    for j in range(i + 1, len(allt)):
        if allt[i][0].split(':')[0] == allt[j][0].split(':')[0]:
            continue
        if allt[i][1].overlaps(allt[j][1]):
            print(f"  CROSS-AXES  '{allt[i][0]}' x '{allt[j][0]}'")
            bad += 1

print('CLEAN' if bad == 0 else f'{bad} COLLISIONS')
sys.exit(1 if bad else 0)
