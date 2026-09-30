# Thresholding and evaluation

This page explains how to turn an optimized (partly grey) density into a clean 0/1 part, and how to report numbers that belong to that part. It is for anyone publishing results, or passing a design to CAD or manufacturing.

## Why the grey design's numbers aren't the part's numbers
Density methods (MMA, OC) end with some intermediate densities, where the SIMP law makes grey material weak. The objective reported for that design describes the grey structure, not the part you'd build: a 0/1 version has a different compliance, volume and stress.

## Thresholding
```python
import numpy as np
import torch
from pyto.autodiff.material_model import MaterialModel

x = np.asarray(fe.mesh.elemPseudoDensity)
x01 = torch.as_tensor((x > 0.5).astype(float))           # fixed threshold 0.5
with torch.no_grad():
    u01 = fe.solve(x01, MaterialModel.SIMP)                # analyse the 0/1 design
compliance01 = float(np.dot(fe.bc.force, u01.numpy()))     # load-driven structural compliance
```
From the post-processing example (cantilever, 30 % volume):
```
grey design compliance 25.54 J at volume 0.300; 0/1 design 17.36 J at volume 0.319
```
The 0/1 part is stiffer here, but it uses a little more material than the limit. A fixed 0.5 threshold doesn't preserve volume.

**Volume-preserving threshold:** the drivers' `binarize_topology=True` sorts the densities and picks the threshold that keeps the mean density. The last history entry is then the 0/1 design's value. Pareto and LevelSet return 0/1 designs anyway.

## When thresholding goes wrong
Thresholding can **cut thin grey members** that carried load or heat. The 0/1 design is then disconnected, and its value is meaningless: for example, 1.1e10 instead of 1540 for a thin thermal conduction path. The symptoms:
- the last history value is orders of magnitude worse than the one before;
- `fe.mesh.find_connected_components()` returns more than one component for the solid part.

What to do:
- **Evaluate without thresholding** (`binarize_topology=False`), and threshold yourself after checking the picture.
- **Heaviside projection** (`to_params.HeavisideProjection = True`, MMA) pushes the design toward 0/1 during the optimization, so little changes at the end. It needs more iterations, because β is doubled every `HeavisideBetaInterval` iterations (default 50) up to `HeavisideBetaMax` (32).
- **A finer mesh or a larger filter radius**, so members are several elements thick.
- **`Eliminate_Hanging_Elements`** (`TOParams`, MMA) keeps only the largest connected solid component after binarization, removing floating islands.

## Reporting checklist
- [ ] State whether values are for the grey design or the 0/1 design.
- [ ] Give the volume fraction of the design you report on.
- [ ] Report stresses on the solid region only (`final_design_summary`; see [Fields and plots](fields-and-plots.md)).
- [ ] Check connectivity: one solid component connecting loads and supports.
- [ ] For publication, re-analyse the final geometry (smooth STL or 0/1 voxels) with a finer mesh ([Export](export.md)).

---
Next: [Export](export.md) · [Back to contents](../README.md)
