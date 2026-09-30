# GUI overview

This page introduces the PyTO desktop application: its layout, the order of work, and what each sidebar step does. It is for GUI users; the other GUI pages go into each step.

Start it with `python run_gui.py` from the repository folder; see [Installation](../getting-started/installation.md).

## Layout
- **Sidebar (left):** the steps, top to bottom. Each has an icon:

  | Icon | Meaning |
  |---|---|
  | arrow | next to do |
  | check mark | done |
  | cross | waiting for an earlier step |

  Steps stay disabled until what they need exists: Material needs a geometry, loads need a material, and so on.
- **3D view (right):** the part, the loads and supports, the mesh, and results. Rotate with the left mouse button, pan with the middle one, zoom with the wheel.
- **Message area:** progress and results of every action.
- **Status bar:** the GUI version.

> TODO (screenshot): the main window with the sidebar and a loaded part.

## The steps
| Sidebar | Window | What you do there | Page |
|---|---|---|---|
| **Units** | Units | unit system (MKS, mmKS, IPS), temperature (Kelvin, Celsius, Fahrenheit), angle | [Geometry and materials](geometry-and-materials.md) |
| **Geometry** | Geometry | load an STL (or get one from SolidWorks) | [Geometry and materials](geometry-and-materials.md) |
| **Material** | Material | choose or define the material | [Geometry and materials](geometry-and-materials.md) |
| **Structural Loads** | Structural Loads | forces, torques and supports on faces or nodes | [Loads and supports](loads-and-supports.md) |
| **Thermal Loads** | Thermal Loads | temperatures, heat flux, total heat | [Loads and supports](loads-and-supports.md#thermal-loads) |
| **Body force** | Body Force | gravity or another acceleration | [Loads and supports](loads-and-supports.md#body-force) |
| **Display Options** | Display Options | what the 3D view shows; save an image; animate | – |
| **Analysis** | Analysis | mesh, solver, analyses, results of the initial or optimized design | [Analysis](analysis.md) |
| **TopOpt Options** | TopOpt Options | symmetry, extrusion, cyclic symmetry, keep fixed faces | [Manufacturing options](manufacturing-options.md) |
| **TopOpt Execute** | TopOpt Execute | objective, constraints, method, gradient; validate; optimize | [Optimization setup](optimization-setup.md) |
| **TopOpt Postprocess** | TopOpt Postprocess | smooth STL of the result; tetrahedral mesh | [Export](../postprocessing/export.md) |
| **Projects** | Projects | save and load everything | [Projects](projects.md) |

## Physics
The GUI works out the physics from the loads you apply:
- only structural loads and supports → **structural**;
- only thermal loads → **thermal**;
- both → **thermo-structural**.

The objective and constraint lists in TopOpt Execute adapt to the physics.

## Behind the GUI
The GUI builds the same objects you'd build in code: a mesh, `BC`s, an FE solver, and an `OptimizationSpec` for the formulation. It then calls the same drivers. [From the GUI to code](gui-to-code.md) shows how to reproduce a GUI setup in a script.

---
Next: [Geometry and materials](geometry-and-materials.md) · [Back to contents](../README.md)
