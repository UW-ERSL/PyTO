# PyTO: Python Topology Optimization

PyTO is a Python library and desktop GUI for **finite element analysis (FEA)** and **topology optimization (TO)** of 3D parts on voxel (hexahedral) meshes. You bring an STL of the design space, apply supports and loads, and PyTO finds where material should go to minimize (or maximize) an objective under constraints. Gradients come from **automatic differentiation** (PyTorch autograd through the FE solve), so new objectives and constraints need no manual sensitivities.

---

## Capabilities

### Analysis
| Physics | What it solves |
|---|---|
| Structural (linear elastic) | displacement, strain, von Mises stress; point, face, torque and body (gravity/acceleration) loads; fixed faces or nodes in x/y/z |
| Thermal (steady conduction) | temperature from fixed temperatures and heat sources |
| Thermo-structural (coupled) | thermal solve, then thermal-expansion load, then structural solve; stresses from the elastic strain |

Meshing is automatic from an STL (single or multi-body). Linear solvers: PARDISO (default), SciPy direct, and iterative solvers (PCG, deflated PCG, AMG) for large models.

### Topology optimization
| Method | Gradient | Handles |
|---|---|---|
| **MMA** (default) | autograd through the FE solve, or manual | every objective and constraint below, all physics |
| **OC** (optimality criteria) | autograd, or manual | minimize compliance with one volume limit |
| **Pareto** (topological sensitivity) | manual (topological sensitivity) | minimize compliance with one volume limit, incl. body force |
| **LevelSet** | manual (shape derivative) | minimize compliance with one volume limit |

**Objectives and constraints** (minimize or maximize; `<=` or `>=` limits), available per physics: compliance, volume fraction, mass, displacement at a selection (x/y/z/magnitude; mean or smooth maximum), temperature, reaction force, strain energy, von Mises stress (p-norm), stress failure factor (stress/yield), mechanical compliance (load work without the thermal-expansion part), thermal compliance.

**Your own objective or constraint**, without writing a gradient:
- a formula over the quantities above, for example `Displacement(Load1, y, mean) / 1e-3 + 0.1 * VolumeFraction()`. It is checked against a whitelist, never executed as code.
- or a Python function `fn(sol, x, fe_solver)` written with torch operations.

**Gradient choice (MMA, OC):** automatic differentiation by default. *Manual* gradients (no autograd graph) cover only compliance, p-norm stress (structural) and volume fraction; everything else, and design-dependent body forces, needs automatic differentiation. Pareto and LevelSet always use their own manual sensitivities.

**Manufacturing and regularization:** density filter, Heaviside projection, symmetry planes, cyclic symmetry, extrusion, keep-solid regions.

**Before a long run:** "Validate" (one FE solve showing every value), "Check gradient" (autograd against finite differences), and a per-iteration cost estimate.

### Validation
- **Benchmarks:** 46 benchmark problems (structural 2.5D and 3D, thermal, stress/mass, body force, thermo-structural). In the last full sweep, 82 of 89 problem/method pairs matched (within 1 %) or beat the saved reference results.
- **Thermo-structural problems** reproduce published results: Rodrigues & Fernandes (1995) BiClamp trends with temperature, and the Ooms et al. (2023) MBB beam (mechanical case within 0.8 %). Side-by-side images are in `Results/Publications/PyTO_comparison/`.
- **Tests:** about 220 automated tests, including gradient checks against finite differences.

---

## Installation

PyTO is tested with **Python 3.10** in a conda environment named **`PyToLib`** (the name the GUI launcher looks for).

```bash
git clone https://github.com/UW-ERSL/PyTO
cd PyTO
conda create -n PyToLib python=3.10
conda activate PyToLib
pip install -r requirements.txt          # add -dev for the test suite: pip install -r requirements-dev.txt
```

Notes:
- **PyTorch:** the CPU build is enough (`pip install --index-url https://download.pytorch.org/whl/cpu torch`, before the line above, to skip the large GPU build).
- **macOS:** `pypardiso` is not available; PyTO falls back to the SciPy solver (choose SPSOLVE in the GUI).
- **Optional packages** (see the end of `requirements.txt`): `cupy` for the GPU deflation solver, `petsc4py` for PETSc solvers, `pypiwin32` for the SolidWorks interface on Windows.

---

## How to run

### 1. The GUI (interactive)
```bash
python run_gui.py            # from the repository folder; any Python works
python run_gui.py --check    # only report which environment would be used
```
If the Python you start it with lacks the GUI's packages, `run_gui.py` restarts itself in the `PyToLib` environment through `conda run`, so activating first is optional. Set `PYTO_CONDA_ENV` to use another environment name.

**Typical workflow** (left sidebar, top to bottom):
1. **Geometry**: load an STL (examples in `Models/`).
2. **Material**: pick or define one.
3. **Structural Loads** / **Thermal Loads** / **Body force**: select faces (*Facet* or *Triangle* mode) or nodes (*Node* mode). In Node mode you can click nodes or use **Box select nodes** and drag a rectangle; a box keeps working after remeshing.
4. **Analysis**:
   - Generate Mesh, then Thermal, Structural or Thermo-Structural Analysis.
   - **Show results** displays deformation, stress or temperature of the initial design, or of the last optimized design.
5. **TopOpt Options**: manufacturing options (extrusion, symmetry, cyclic, keep fixed faces).
6. **TopOpt Execute**:
   - Tabs **Objective / Constraints / Method**; methods that cannot solve your formulation are disabled, with the reason shown. The Method tab also chooses the gradient (automatic differentiation or manual).
   - Validate the formulation, optionally check the gradient, then **Optimize**.
   - A live plot shows the objective and constraints; tick **Show topology every iteration** to watch the design change.
   - The **Optimized design** buttons show density, deformation, stress or temperature of the result.
7. **TopOpt Postprocess**: recover a smooth STL of the optimized part, or a tetrahedral mesh.
8. **Projects**: save and load everything, including the formulation.

### 2. The notebook (scripted, one problem at a time)
Open `main.ipynb` with the PyToLib kernel. Choose a benchmark problem and a method in the first cell and run all cells.
- The run is identical to one row of the full benchmark sweep.
- It shows the numbers, topology, deformation, stress or temperature, and convergence.
- It includes a worked example of a user-defined objective.

### 3. Benchmarks and reports (terminal, repository root)
Headless runs need `PYTHONPATH=src MPLBACKEND=Agg PYVISTA_OFF_SCREEN=true`:
```bash
# full sweep: every problem x method, each in its own process; output in Results/Results_<date>/ (many hours)
nohup python -m pyto.examples_benchmarks.topopt_run_benchmarks > benchmark_run.log 2>&1 &

# paper comparison images for the thermo-structural problems
python -m pyto.examples_benchmarks.make_publication_comparison Results/Results_<date>/ThermoStructural/DENSITYMMA

# cost of forward / forward+backward / backward (averaged over repeats)
python tests/test_gradient_cost.py --ndof 25000 --repeat 10 --solver PARDISO
```
For a subset, e.g. one category with one method:
```python
from pyto.examples_benchmarks.topopt_run_benchmarks import *
run_all_benchmarks_isolated(optimizationMethods=[TO_METHODS.DENSITYMMA], problems=benchmarks_structural_noncompliance)
```

### 4. Tests
```bash
pip install -r requirements-dev.txt
PYTHONPATH=src MPLBACKEND=Agg PYVISTA_OFF_SCREEN=true QT_QPA_PLATFORM=offscreen pytest tests -q
```

### 5. Using PyTO as a library
```python
from pyto.examples_benchmarks.topopt_run_benchmarks import build_benchmark
from pyto.examples_benchmarks.topopt_structural_benchmarks import StructuralTOExamples
from pyto.topopt.spec import OptimizationSpec, ObjectiveSpec, ConstraintSpec, compile_spec
from pyto.topopt.drivers.mma import topopt_mma
import numpy as np

fe, to_params, _ = build_benchmark(StructuralTOExamples.CantileverTipLoad)   # FE model + default settings
tip = np.unique(np.nonzero(fe.bc.force)[0] // 3)                           # loaded nodes
spec = OptimizationSpec(objective=ObjectiveSpec("Displacement(Tip, y, mean)", "maximize"),
                        constraints=[ConstraintSpec("VolumeFraction()", "<=", 0.5)])
to_params = compile_spec(spec, selections={"Tip": {"nodes": tip}}, base=to_params)
u, history, success, message, n_fea = topopt_mma(fe, to_params=to_params)
```

---

## Repository layout
| Path | Contents |
|---|---|
| `run_gui.py` | GUI launcher |
| `main.ipynb` | one-problem notebook |
| `src/pyto/core` | meshing, boundary conditions, materials |
| `src/pyto/physics` | structural, thermal and thermo-structural FE solvers |
| `src/pyto/autodiff` | differentiable sparse solve, material laws, responses (objectives/constraints) |
| `src/pyto/topopt` | optimizers (MMA, OC, Pareto, LevelSet), filters, formulation (`spec.py`, `expressions.py`, `capabilities.py`) |
| `src/pyto/examples_benchmarks` | benchmark problems and the sweep runner |
| `src/pyto/gui` | the GUI |
| `src/pyto/io` | STL reading, STL recovery, project files |
| `Models/` | example STL files |
| `tests/` | automated tests |
| `Results/` | sweep outputs, comparison reports, publications (not tracked by git) |

---

## Contributing
Open an issue or a pull request. Please run the test suite before submitting.

## License
Licensing information will be provided in a future update.
