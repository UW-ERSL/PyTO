# Analysis

This page covers the **Analysis** window: meshing, solver choice, the three analyses, and viewing results of the initial or optimized design. It is for GUI users who want to check a setup before optimizing, or inspect a result afterwards.

## Mesh
| Control | Meaning |
|---|---|
| **Mesh Quality** | presets for the element count: Very Coarse 10 000, Coarse 25 000, Normal 50 000, Fine 75 000, Very Fine 100 000 |
| **#Elements** | the target number of voxel elements; edit it directly for any other size |
| **Generate Mesh** | voxelizes the STL at that size and shows the mesh |

- A structural model has about 3 × (elements) unknowns.
- **Coarse (25 000)** is a good start: optimizations run in minutes. Refine once the setup is right.
- Thin walls need several elements across their thickness; otherwise they come out with holes, or disappear.

## Solver
**Solver Type** chooses the linear solver for analysis **and** for optimization:

| Option | Notes |
|---|---|
| **PARDISO** | default and fastest; not on macOS |
| **SPSOLVE** | works everywhere; use on macOS |
| **DPCG**, **PCG**, **PYAMG** | iterative solvers, used in analysis. During optimization they are not available and PyTO falls back to SPSOLVE (with a warning) |

## Analyses
| Button | Solves | Needs |
|---|---|---|
| **Thermal Analysis** | temperature | thermal loads (a fixed temperature somewhere) |
| **Structural Analysis** | displacement and stress | supports and structural loads (and/or body force) |
| **Thermo-Structural Analysis** | temperature, then displacement and stress including thermal expansion | both; **Zero-strain T** = the stress-free temperature, in the thermal loads' unit |

All three analyse the **full design** (every element solid). The message area reports maximum values.

## Show results
Choose **Initial design** or **Optimized design**, then **Deformation**, **Stress** or **Temperature**.
- **Initial design:** the results of the last analysis above.
- **Optimized design:** the result of the last optimization in TopOpt Execute, analysed for its own density. Stress and deformation are then reported on the solid part.
- Buttons for fields that don't exist yet are disabled; the tooltip says which analysis or optimization to run first.

This is the quickest way to see how an optimized part deforms and where it is stressed. For figures and numbers from code, see [Fields and plots](../postprocessing/fields-and-plots.md).

---
Next: [Optimization setup](optimization-setup.md) · [Back to contents](../README.md)
