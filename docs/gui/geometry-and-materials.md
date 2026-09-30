# Geometry and materials

This page covers the first three sidebar steps: units, geometry and material. It is for GUI users starting a new model.

## Units
**Units** sets how values are **displayed and entered** in the loads, material and results windows.

| Setting | Options |
|---|---|
| **Unit System** | `MKS` (m, N, Pa), `mmKS` (mm, N, MPa), `IPS` (inch, lbf, psi) |
| **Temperature Unit** | Kelvin, Celsius, Fahrenheit |
| **Angle Unit** | Degree, Radian |

Two exceptions:
- **STL files are read as metres.** Scale the STL in your CAD tool if it was exported in mm.
- **Bounds in TopOpt Execute** (constraints, formulas) are **always SI**, whatever the unit system; the window says so.

## Geometry
**Load STL Geometry** opens an STL file and shows it. Examples are in `Models/`, one folder per part.
- **Binary and ASCII** STL both work.
- **Multi-body STLs** (several closed shells) are meshed as one design domain with a component id per body.
- The STL should be **watertight** (closed, no holes). Open or self-intersecting surfaces give missing elements when meshing, and the message area says so. Repair the STL in your CAD tool or a mesh tool.
- **Update Geometry** reloads the file after you've changed it. **Get STL from SolidWorks** (Windows, with SolidWorks and `pypiwin32`) exports the active part directly.

The mesh isn't made here; it is generated in **Analysis → Generate Mesh**, at a size you choose ([Analysis](analysis.md)).

## Material
Pick a material from the list: **Steel**, **Aluminum**, **Titanium**, or **Custom**. For **Custom**, the fields become editable:

| Field | Used for |
|---|---|
| Young's Modulus, Poisson's Ratio | structural stiffness |
| Yield Strength | stress failure factor, stress reports |
| Density | mass, self-weight |
| Thermal Conductivity | thermal analysis |
| Thermal Expansion | thermo-structural analysis |
| Spec. Heat Capacity, Price | stored; not used by the current analyses |

Click **Apply**. The **Do not optimize** checkbox is shown but not used yet. To keep regions solid, use **Keep Fixed Faces** in [Manufacturing options](manufacturing-options.md), or a keep-solid selection in code.

In code, materials are `pyto.core.mat_lib` objects; see [A problem from scratch](../code/problem-from-scratch.md#2-material).

---
Next: [Loads and supports](loads-and-supports.md) · [Back to contents](../README.md)
