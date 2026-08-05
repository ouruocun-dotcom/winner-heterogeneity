"""Report ink touching the edge of a saved figure, i.e. content cut off.

Measuring the file is unambiguous, whereas inspecting the in-memory figure
misfires when a script builds more than one and reports on the wrong object.

Usage: python3 _clip.py fig1.png fig2.png ...
"""
import sys
import numpy as np
from PIL import Image

bad = 0
for f in sys.argv[1:]:
    a = np.array(Image.open(f).convert('L'))
    hit = {'top': int((a[0, :] < 200).sum()),
           'bottom': int((a[-1, :] < 200).sum()),
           'left': int((a[:, 0] < 200).sum()),
           'right': int((a[:, -1] < 200).sum())}
    cut = [k for k, v in hit.items() if v]
    print(f"  {f:14s} {a.shape[1]}x{a.shape[0]}  "
          + ("clean" if not cut else f"CLIPPED at {cut}  {hit}"))
    bad += len(cut)
print("ALL CLEAN" if not bad else f"{bad} clipped edges")
sys.exit(1 if bad else 0)
