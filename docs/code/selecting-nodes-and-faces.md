# Selecting nodes and faces

This page shows how to find the nodes and elements for supports, loads, named selections and keep-solid regions. It is for anyone building a problem in code.

All examples assume a mesh already exists (see [A problem from scratch](problem-from-scratch.md)), plus:
```python
import numpy as np
xyz = mesh.node_xyz                 # (num_nodes, 3), metres
```

## Degrees of freedom
| Physics | dofs per node | dofs of node `i` |
|---|---|---|
| Structural, thermo-structural (displacement) | 3 | `3*i` (x), `3*i+1` (y), `3*i+2` (z) |
| Thermal, and the thermal half of thermo-structural (temperature) | 1 | `i` |

To fix the x and z displacement of nodes `n`: `np.concatenate([3 * n, 3 * n + 2])`.

## By bounding-box face
The fastest way for box-like parts:
```python
left = mesh.getNodesOnBoundingBoxPlane(0, True)      # axis 0 = x, 1 = y, 2 = z; True = min side, False = max side
top = mesh.getNodesOnBoundingBoxPlane(1, False)
```

## By coordinates
Any condition on the coordinates works. Always compare with a tolerance, never with `==`:
```python
tol = 1e-6 * np.ptp(xyz, axis=0).max()
right = np.where(xyz[:, 0] > xyz[:, 0].max() - tol)[0]                          # x-max face
mid_line = np.where((xyz[:, 0] > xyz[:, 0].max() - tol) &
                    (np.abs(xyz[:, 1] - 0.25) < 0.03))[0]                          # a line on that face
disk = np.where(np.hypot(xyz[:, 0] - 0.5, xyz[:, 2] - 0.5) < 0.1)[0]             # inside a cylinder along y
```
For voxel meshes, a node lies exactly on the grid, so choose windows wider than half an element (`mesh.elem_size`).

## By a point and radius
```python
near_hole = mesh.get_nodes_within_radius(np.array([0.0, 0.25, 0.05]), 0.06)
ring = mesh.get_nodes_within_annular_region(pt=np.array([0.5, 0.5, 0.0]), axis=np.array([0, 0, 1.0]),
                                            r_inner=0.08, r_outer=0.12)
closest = mesh.get_nodes_from_locations(np.array([[0.0, 0.0, 0.0], [1.0, 0.25, 0.05]]))   # one node per point
```

## By STL face
For parts made from an STL, you can pick the original CAD faces and take the mesh nodes on them. This is what the GUI does when you click a face.
```python
stl = mesh.stlGeom                                   # the STL used to build the mesh
normals = np.asarray(stl.tri_normals)                # (triangles, 3)
centres = stl.mesh.vectors.mean(axis=1)              # (triangles, 3)
right_face = np.where((normals[:, 0] > 0.99) & (centres[:, 0] > centres[:, 0].max() - 1e-6))[0]
nodes = mesh.get_nodes_on_triangles(right_face)      # mesh nodes within ~0.7 element of those triangles
```
On `Models/Cantilever/Cantilever.STL` with 3000 elements, this gives 8 triangles and 75 nodes on the x = 2 m face.

## Elements
| Goal | Code |
|---|---|
| Elements near a point | `mesh.get_elems_within_radius(np.array([0.5, 0.2, 0.05]), 0.1)` |
| The element at a point | `elem, _ = mesh.get_element_containing_point(np.array([0.5, 0.2, 0.05]))` (returns `(-1, None)` if outside) |
| Elements by their centre | `np.where(mesh.elem_centers[:, 1] > 0.4)[0]` |
| Elements touching given nodes | `np.where(np.isin(mesh.elemArray, nodes).any(axis=1))[0]` |
| Elements on the outer surface | `mesh.get_boundary_elements()` |

## Using selections
- **Boundary conditions:** turn nodes into dofs (above) for `BC(fixed_dofs=..., force=...)`.
- **Expressions:** pass a dict of named selections. Node-based quantities use `"nodes"`, region quantities use `"elements"`:
  ```python
  selections = {"Tip": {"nodes": mid_line},
                "Flange": {"elements": np.where(mesh.elem_centers[:, 1] > 0.4)[0]}}
  # "Displacement(Tip, y, mean)", "VolumeFraction(Flange)"
  ```
- **Keep solid:** elements that should stay material: `to_params.ElemsToKeep = list(elements)` (from code), or `ManufacturingSpec(keep_selection="Flange")` in a spec. The density drivers give these elements the most favourable sensitivity each iteration, so they fill in. It is a strong nudge, not a hard bound.

## Loads spread over a face
A total force $F$ spread evenly over the selected nodes:
```python
force = np.zeros(3 * mesh.num_nodes)
force[3 * right + 1] = -1000.0 / len(right)          # 1 kN in -y
```
Even spreading is simple, and fine for optimization. Corner and edge nodes of a face carry a smaller area, so for accurate stresses near the load, weight each node by its tributary area instead.

---
Next: [Formulations with specs](formulation-with-spec.md) · [Back to contents](../README.md)
