# Quick start: GUI

This page walks through a complete optimization in the PyTO GUI: load a part, support and load it, optimize, and export the result. It is for anyone who prefers clicking to coding; allow about 10 minutes.

The example is the cantilever in `Models/Cantilever/Cantilever.STL`: clamped on one end, pushed down at the other.

## Start the GUI
From the repository folder:
```bash
python run_gui.py
```
If the Python you use lacks PyTO's packages, the launcher restarts itself in the `PyToLib` conda environment. See [Installation](installation.md).

The window has a **sidebar** of steps on the left and a **3D view** on the right. Work down the sidebar from top to bottom. A step's icon turns into a check mark when it's done, and steps that need earlier ones stay disabled until then.

## Steps
1. **Units.** Choose the unit system used for display and input (SI by default). Optimization limits are always entered in SI units; see [Expressions](../code/expressions.md).
2. **Geometry.** Click **Load STL Geometry** and open `Models/Cantilever/Cantilever.STL`. The part appears in the 3D view.
3. **Material.** Pick **Steel** and click **Apply**.
4. **Structural Loads**, the support:
   1. Set **Selection** to **Facet**, then click the left end face in the 3D view.
   2. Set **Load Type** to **Fixed XYZ** and click **Apply**.
5. **Structural Loads**, the load:
   1. Increase **Load Set**, click the right end face, and set **Load Type** to **Force**.
   2. Enter **Y** = `-1000` (N) under **Force Components** and click **Apply**. Arrows show the load.
6. **Analysis.**
   1. Click **Generate Mesh**, then **Structural Analysis**.
   2. Under **Show results**, look at the deformation and the stress of the full design.
7. **TopOpt Options.** Leave the defaults (no symmetry or extrusion) and click **Apply**.
8. **TopOpt Execute:**
   1. **Objective** tab: **Minimize**, type **Compliance**.
   2. **Constraints** tab: one row `VolumeFraction()  <=  0.3`.
   3. **Method** tab: **MMA**, gradient **Automatic differentiation**, 100 iterations.
   4. Click **Validate formulation**. The message lists the values at the start design and any problems.
   5. Click **Optimize**. The live plot shows the objective and constraints, and the **Iterations** tab shows each iteration's numbers. Tick **Show topology every iteration** to watch the shape form (slower on big meshes).
9. **Look at the result.** In the **Analysis** window, **Show results** now also offers the optimized design: deformation, stress and temperature.
10. **TopOpt Postprocess.** Click **Apply Postprocess** to turn the voxel result into a smooth STL next to the original file. **Generate TetMesh** makes a tetrahedral mesh of it.
11. **Projects.** Save everything (geometry path, loads, formulation) to reopen later.

> TODO (screenshot): main window with the sidebar, and TopOpt Execute during a run.

## What next
- Other objectives (deflection at a point, stress, temperature) and constraints: [Optimization setup](../gui/optimization-setup.md) and [Expressions](../code/expressions.md).
- Picking nodes instead of faces, and box selection: [Loads and supports](../gui/loads-and-supports.md).
- Doing the same from Python: [From the GUI to code](../gui/gui-to-code.md).

---
Next: [Quick start: notebook](quickstart-notebook.md) · [GUI overview](../gui/overview.md) · [Back to contents](../README.md)
