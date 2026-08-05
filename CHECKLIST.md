# Figure checklist

Run before shipping any figure. Each step exists because skipping it produced
a defect that reached the manuscript.

## 1. Build at final size

Figures go into the manuscript at `\columnwidth` = 3.375 in. Build them at
that width via `_figstyle.py`, so the scale factor is 1 and the point sizes in
the code are the point sizes the reader sees.

*Why:* figures were once built 7–11 in wide and shrunk by 0.31–0.48, putting
axis labels at 1.8–4.8 pt. A PDF viewer can zoom, but a referee who prints
cannot, and APS asks for legibility at final size.

## 2. Fixed margins, never `bbox_inches='tight'`

`_figstyle.py` sets `savefig.bbox` to `standard`; position axes with
`subplots_adjust`.

*Why:* tight cropping moves artists after the layout is fixed, so a check run
before saving does not describe the file written. This is why panel labels
that passed the collision test still landed outside the axes.

## 3. Find label anchors, do not guess them

    python3 _place.py make_xxx_figure.py

Rasterises each panel and reports which of nine anchors is free of dark ink.
Place `(a)`/`(b)` at one of those, with `bbox=dict(fc='white', alpha=0.75)`.

## 4. Collision check

    python3 _collide.py make_xxx_figure.py

Flags text over text, text over tick labels, and text over data. Lines are
densely resampled, so a legend lying on a curve is caught — an earlier version
sampled only marker vertices and reported such figures clean. Text on an
opaque patch is exempt from the data test, which is correct for a label over a
heat map.

Must print `CLEAN`.

## 5. Clipping check, on the saved file

    python3 _clip.py fig1.png fig2.png fig3.png figS1.png figS2.png

Reports ink touching the canvas edge, which means content was cut off. With
fixed margins there is no automatic re-cropping, so a legend placed above the
axes can run off the top and simply vanish; this is what removed half of a
legend title from Fig. 2.

Measure the file, not the in-memory figure: a script that builds two figures
will hand the inspector the wrong one, which produced several false reports
before this step was written this way.

## 6. Look at it

The checks catch overlap, not ugliness: a legend can be collision-free and
still crowd the data, and a colorbar can be collision-free and still sit too
close. Open the PNG.

## 7. Recount the length

Figure aspect ratio enters the PRL word count as `150/aspect + 20`. Changing a
figure's shape changes the budget; `wordcount.py` assumes a fixed aspect and
must be corrected by hand.
