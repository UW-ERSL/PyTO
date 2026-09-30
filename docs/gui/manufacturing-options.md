# Manufacturing options

This page explains the **TopOpt Options** window: symmetry, cyclic symmetry, extrusion and keeping faces solid. It is for GUI users who need a design that is symmetric or can be made by a given process.

Tick an option to enable it, set its value, and click **Apply**. The options are applied as filters on the design ([Add a filter](../extending/add-a-filter.md) explains how), so every method respects them.

| Option | Values | Effect |
|---|---|---|
| **X-Symmetry**, **Y-Symmetry**, **Z-Symmetry** | – | the design is mirror-symmetric about the mid-plane normal to that axis (the plane through the middle of the part's bounding box) |
| **CyclicSym(Z)** | (2) 180° ... (8) 45° | the design repeats n times around the z axis through the middle of the bounding box, as for wheels, rotors and flanges |
| **Extrude** | XDir, YDir, ZDir | the density is constant along that direction, so the part has a constant cross-section and can be extruded, cut or cast straight |
| **Keep Fixed Faces** | – | elements at the supports are kept solid (also available as the selection `FixedFaces`) |
| **AM Build**, **Draw Direction** | directions | shown, but **not applied yet**: no method implements overhang or draw constraints so far |

Notes:
- **Symmetry is about the mesh's bounding box.** A part that isn't itself symmetric gets approximate mirror pairs.
- **Loads and supports are not made symmetric for you.** For a symmetric result they should be symmetric too; otherwise the design is symmetric while the loading isn't.
- **2.5D problems** (plates loaded in their own plane): use **Extrude** along the thickness direction to get a through-thickness design.
- From code, the same options are `ManufacturingSpec(extrude=..., symmetry=[...], cyclic_z=..., keep_selection=...)` or the `TOParams` fields ([Formulations with specs](../code/formulation-with-spec.md)).

The **filter radius**, which sets the minimum member size, is on the Method tab of TopOpt Execute ([Optimization setup](optimization-setup.md#method-tab)).

---
Next: [Projects](projects.md) · [Back to contents](../README.md)
