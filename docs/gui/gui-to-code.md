# From the GUI to code

This page shows how to take a problem set up in the GUI into a Python script: to run variations, sweeps or larger meshes, or to extend it. It is for users moving from clicking to coding.

## What carries over directly: the formulation
The GUI's objective, constraints, method and settings are an `OptimizationSpec`, saved in the project file under `"optimization"`:
```python
import json
from pyto.topopt.spec import OptimizationSpec

with open("Models/MyPart/MyPart.pyto") as f:
    project = json.load(f)
spec = OptimizationSpec.from_dict(project["optimization"])
print(spec.to_json())
```
The same spec is compiled in code exactly as the GUI does it; see [Formulations with specs](../code/formulation-with-spec.md). It refers to selections by name (`Load1`, `Support1`, ...), so give `compile_spec` a `selections` dict with those names.

## What you rebuild: geometry, loads and supports
The project stores loads as picks on the STL: triangle indices, node positions, box regions. In code, the robust way is to find the same places by geometry:
1. **Mesh the same STL:**
   ```python
   mesh = HexMesher()
   mesh.createMeshFromSTLFile(project_stl_path, nElemsDesired=25000)
   mesh.createEdofMatStructural()
   ```
2. **Find each support and load face:** by coordinates, bounding-box planes, or STL face normals; see [Selecting nodes and faces](../code/selecting-nodes-and-faces.md#by-stl-face). A face picked in **Facet** mode in the GUI is one CAD face: select it in code by its normal and position.
3. **Build the `BC`:** as in [A problem from scratch](../code/problem-from-scratch.md#3-supports-and-loads). The GUI shares a force equally among the face's nodes, so do the same (`force / len(nodes)`) to get identical loading.
4. **Name the selections** as in the GUI, and compile:
   ```python
   selections = {"Load1": {"nodes": load_nodes}, "Support1": {"nodes": support_nodes}}
   to_params = compile_spec(spec, selections)
   ```

## Matching the GUI's settings
| GUI | Code |
|---|---|
| Material window | `mat_lib.create_material_with_defaults(name, youngs_modulus=..., ...)` with the SI values from `project["material_data"]` |
| Analysis → Solver Type | `solver=Solvers.PARDISO` (or `SPSOLVE`) |
| Analysis → #Elements | `nElemsDesired=` |
| Body force | `elem_body_force` ([A problem from scratch](../code/problem-from-scratch.md#3-supports-and-loads)) |
| TopOpt Options | `spec.manufacturing` (already in the saved spec) |
| Method tab | `spec.method`, `spec.regularization` (already in the saved spec) |
| Optimize | `topopt_mma(fe, to_params, maxMMAIterations=spec.method.max_iterations, move_limit=spec.method.move_limit, binarize_topology=False)`; the GUI doesn't binarize |

## Going the other way
To study a problem built in code with the GUI's tools, save its STL (from CAD) and set it up in the GUI. Or keep working in code, and use [Part 3](../postprocessing/history-and-convergence.md) for plots and export.

---
Next: [Part 3: Results and post-processing](../postprocessing/history-and-convergence.md) · [Back to contents](../README.md)
