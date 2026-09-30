# Projects

This page explains saving and loading GUI work as project files. It is for GUI users who want to come back to a setup, or share one.

## Save and load
Open **Projects** in the sidebar:
- **Save Project** writes a `.pyto` file. The default name is the STL's name, next to it.
- **Load Project** opens a `.pyto` file and restores the setup; each window's sidebar icon is updated.

Examples: `Models/Cantilever/CantileverMidLoad.pyto` and others in `Models/`.

## What a project holds
A `.pyto` file is JSON:

| Key | Content |
|---|---|
| `version` | project format version |
| `stl_file_path` | the STL, **relative** to the project file where possible; keep them together when you move them |
| `settings` | unit system, temperature and angle units |
| `material_data` | material properties (SI) and the chosen name |
| `structuralBC`, `thermalBC` | loads and supports: the picked STL triangles, picked node positions, box regions and their values |
| `body_force` | the acceleration, if any |
| `topopt_constraints` | TopOpt Options (symmetry, extrusion, cyclic symmetry, keep fixed faces) |
| `optimization` | the formulation from TopOpt Execute, as an `OptimizationSpec` (objective, constraints, method, gradient, filter, Heaviside) |
| `analysis_settings` | element count and solver |

The mesh and results are **not** stored. After loading, click **Generate Mesh**, then analyse or optimize again. The optimized design is written separately, as a VTU (and an STL after post-processing) next to the STL.

**Older projects** (before the TopOpt Execute tabs) load fine. Their performance limits become constraint rows, switched off, so you can review them before use.

## Safety
A formulation that uses **Python function...** stores only the file path and function name. Loading the project doesn't run it; running the optimization asks for confirmation first, because that executes the file.

---
Next: [From the GUI to code](gui-to-code.md) · [Back to contents](../README.md)
