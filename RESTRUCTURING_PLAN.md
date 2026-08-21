# PyTO restructuring + autodiff fix plan (branch: `AutoPyTO`)

Working log of an in-progress effort: land `src/` on a proper package structure
(`core/`, `physics/`, `solve/`, `autodiff/`, `topopt/`, `io/`, `examples_benchmarks/`, `gui/`)
while fixing the autodiff bugs found along the way, one gated phase at a time. Kept in the
repo so the state and reasoning survive across sessions, not just in local planning tools.

**Status:** All phases (0 through 7) are complete. Phases 0, 2, 3-5, and 6 are **committed**
locally on `AutoPyTO` (commits `df57d87`, `c8be271`, `9efef66`, `09cb5ee` — see `git log`);
Phase 7 (this session) is complete and about to be committed. **Not pushed.** `main` and
`automatic_differentiation` are untouched throughout — everything lives on `AutoPyTO`, created
off `main` and merged with `automatic_differentiation`.

**If you're picking this up cold** (new session, compacted context, etc.): read this whole
file top to bottom, then `git log --oneline` to confirm which commits are actually present.
Every planned phase is done; `src/` is now a real `pyto` package with no remaining flat
compatibility shims, and `pytest tests/` is the one command that verifies all of it.

## Context

`AutoPyTO` currently holds ~50 flat files in `src/` with no package structure, no test
suite, and two only-partially-reconciled sensitivity systems (hand-derived NumPy adjoint vs.
PyTorch autograd). Prior analysis on this branch found concrete breakage: OC/Pareto drivers
pass NumPy arrays into a `solve()` that now requires torch tensors; thermal FEA had no torch
material-model twin so it couldn't run through the autograd MMA driver; several `TO_QOI`
objective/constraint branches (`MASS`, `GVECTOR`, `MAX_VONMISES_STRESS`,
`COMPLIANCE`-as-constraint) crashed when handed grad-tracked tensors; sensitivity state
leaked through module globals (`SIMP_PENALTY`) and a function attribute
(`compute_constraint_and_gradient.stress_scaling`); and two incompatible `Solvers` enums
existed (`linear_solvers.Solvers` vs `torch_spsolve.Solvers`).

All ~50 files use flat, non-package imports (`import hex_mesher`,
`from topopt_common import *`) — every file assumes `src/` is directly on `sys.path`. This
means any directory move breaks imports repo-wide unless compatibility shims are used during
the transition. No environment had the full dependency set to even import everything:
`PyToLib` had everything except `torch`; `VAEBeamCS` had `torch` but was missing
`pyamg`/`numba`/`tetgen`/`numpy-stl`/`pypardiso`. Neither had `pytest`.

**Decisions locked in:** (1) fix known bugs opportunistically as each file is touched during
its move, rather than doing a purely mechanical relocation first; (2) standardize the test
environment on `PyToLib` + `torch` (CPU) + `pytest`.

## Target structure

```
src/pyto/
  core/            mesh_base.py, hex_mesher.py, tet_mesher.py, bc.py, mat_lib.py
  physics/
    structural/    element.py, problem.py
    thermal/       element.py, problem.py
    thermoelastic/ (from topopt_thermostructural_sensitivity.py, check_thermoelasticity_sensitivity.py)
  solve/           solvers.py (ONE Solvers enum), numpy_backend.py (was linear_solvers.py), deflation.py
  autodiff/
    sparse_solve.py        (was torch_spsolve.py)
    assembly.py             (new: shared torch assembly for both physics)
    material_model.py       (torch SIMP/RAMP, structural AND thermal)
    qoi/                    (compliance, volume_fraction, mass, stress, gvector — one file each, registry-based)
    reference_adjoint/      (legacy NumPy adjoint formulas — test-oracle only, never imported by a driver)
  fea/             fea_solver.py, postprocess.py
  topopt/
    common.py, filters.py
    drivers/       _shared.py (factored boilerplate), mma.py, oc.py, pareto.py, levelset.py
  io/              stl_reader.py, project_manager.py, solidworks_interface.py
  examples_benchmarks/   all *_examples.py, *_benchmarks.py, topopt_run_benchmarks.py (unchanged internals)
  gui/             PyTOGUI.py, PyTO_demos.py

tests/
  conftest.py             adds src/ (and src/pyto) to sys.path; small fixture mesh/problem
  test_imports_smoke.py   imports every module, fails loudly listing broken ones
  test_sparse_solve_gradcheck.py
  test_qoi_gradients.py   gradcheck each registered QOI against reference_adjoint, both physics
  test_drivers_smoke.py   each driver runs 2-3 iterations, both physics, objective decreases
```

During the transition, every moved file leaves a one-line shim at its old flat path (e.g.
`src/mat_lib.py: from pyto.core.mat_lib import *`) so the ~50 untouched flat-import call
sites keep working until Phase 7 removes the shims.

## Phase gates (summary)

| Phase | What | Gate |
|---|---|---|
| 0 | Test environment + safety net, no restructuring | `pytest tests/` green in `PyToLib` |
| 1 | File → target-bucket inventory, no code moves | mapping reviewed, tests unchanged |
| 2 | Move `core/` (`mat_lib.py`, `bound_cond.py`) | tests green after each move |
| 3 | Move `solve/`+`autodiff` core, unify `Solvers` enum | gradcheck green + numeric equivalence verified |
| 4 | Fix + relocate QOI/material-model layer (bug-fixing) | `test_qoi_gradients.py` green, both physics, vs. oracle |
| 5 | Unify + relocate optimizer drivers | `test_drivers_smoke.py` green, both physics |
| 6 | Relocate I/O, examples/benchmarks, GUI (last) | tests green (met); manual GUI checklist (pending human) |
| 7 | Remove shims, finalize | NOT STARTED — blocked on moving `physics/`+`core/` files first, see below |

---

## Phase 0 — Environment + safety net

Committed as `df57d87`.

- Added CPU `torch` + `pytest` to the `PyToLib` conda environment; recorded in
  `requirements-dev.txt`.
- Built `tests/` from scratch: `conftest.py` (adds `src/` to `sys.path`, small real fixture
  problems via the existing benchmark generators), `test_imports_smoke.py` (imports every
  `src/*.py` module), `test_sparse_solve_gradcheck.py` (gradcheck on the `SparseLinearSolve`
  adjoint primitive), `test_qoi_gradients.py` (FD checks against a real mesh, seeded from the
  pattern already present in `check_thermoelasticity_sensitivity.py`), `test_drivers_smoke.py`
  (MMA/OC smoke tests, with `xfail` documenting known-broken paths).
- **Merge-fallout bugs found and fixed** (blocking — nothing could even run without these):
  - `topopt_material_model.py`: `get_structural_material_model_scaling_torch` referenced
    `EVOID_RELATIVE`/`SIMP_PENALTY`/`RAMP_PENALTY`, which `main` had renamed to
    `_EVOID_RELATIVE`/`_SIMP_STRUCTURAL_PENALTY`/`_RAMP_PENALTY` — `NameError`.
  - `topopt_obj_cons_sensitivities.py`: same stale-name problem, compounded by wildcard
    imports never bringing in underscore-prefixed names — fixed with an explicit import.
  - `topopt_mma.py`: the density filter (`H`, `Hs`) is a `scipy.sparse` matrix matmul'd
    directly against a grad-tracked torch tensor — added a torch-sparse copy of the filter
    for that step. Also fixed 3 references to the removed
    `to_params.APPLY_FILTER_TO_SENSITIVITY` (folded into `APPLY_FILTER_TO_DENSITY` by `main`).
- Also dropped 6 stray numba `.nbc`/`.nbi` cache binaries under `src/__pycache__` that were
  tracked before `.gitignore` excluded that path.

**Result:** 46 passed, 3 xfailed (thermal MMA, structural OC, `hex_transient_thermal` import).

---

## Phase 1 — Inventory & target mapping

No commit needed (analysis only). All ~49 `.py` files plus 4 non-code assets mapped to a
target bucket, with 5 real ambiguities resolved by reading actual file content:

- **`hex_plotter.py` → `gui/`** — pure pyvista/matplotlib rendering, not numerics.
- **`mmaWrapper.py` → `topopt/drivers/`** — the GCMMA solver itself, only ever called by `mma.py`.
- **`topopt_stl_recovery.py` → `io/`** — flagged: uses `torch`/`nn.Module` incidentally for a
  CNN density-smoother, unrelated to the sensitivity-autodiff work.
- **`topopt_obj_cons_sensitivities.py` doesn't move as one file** — splits into
  `autodiff/qoi/` (torch-native) and `autodiff/reference_adjoint/` (legacy NumPy), done in
  Phase 4.
- New physics variants (large-deformation, modal, transient thermal) grouped into existing
  `physics/structural/`/`physics/thermal/` buckets rather than new top-level folders.

Full file → bucket table lives in the plan history; see git log / `~/.claude/plans/cheerful-dazzling-seahorse.md`
for the complete listing if needed.

**Result:** mapping reviewed; Phase 0 tests reconfirmed green (nothing changed).

---

## Phase 2 — Move `core/`

Committed as `c8be271`.

- Created the `src/pyto/` package skeleton (`pyto/`, `pyto/core/`).
- Moved `mat_lib.py` → `pyto/core/mat_lib.py`, then `bound_cond.py` → `pyto/core/bc.py`, one
  at a time, re-testing after each.
- Left a 2-line compatibility shim at each old flat path. Verified shim transparency
  directly: `mat_lib.Material is pyto.core.mat_lib.Material` (same object, not a copy).

**Result:** 46 passed, 3 xfailed after each move, unchanged.

---

## Phase 3 — Move `solve/` + `autodiff` core, unify `Solvers`

Committed as part of `9efef66`.

1. New `src/pyto/solve/solvers.py`: the single canonical `Solvers` enum (7 members —
   `linear_solvers.Solvers` was already the superset of `torch_spsolve.Solvers`).
2. `linear_solvers.py` → `pyto/solve/numpy_backend.py`; its own enum definition replaced with
   an import from the shared one.
3. `torch_spsolve.py` → `pyto/autodiff/sparse_solve.py`; same swap, removed the now-unused
   `import enum`.
4. `deflation.py` → `pyto/solve/deflation.py` (pure move).
5. Old flat paths become shims, same pattern as Phase 2.

Verified two things beyond "tests pass":
- **Enum identity**: `linear_solvers.Solvers is torch_spsolve.Solvers is pyto.solve.solvers.Solvers`
  — `True`. Before this, these were two distinct classes with matching member *names* but
  different identities — a `Solvers.PARDISO` from one was never `==`/`is` the "same" value
  from the other.
- **Numeric equivalence**: compliance and gradient-norm on the fixture problem, bit-identical
  before/after the move to 12 decimal places.

**Found, not fixed (flagged for later):** `hex_structural_fea.py`'s own demo block sets
`solver = linear_solvers.Solvers.DPCG`, but `torch_spsolve.solve()`'s dispatch only
special-cases `"PETSC"`/`"PARDISO"` by name — `"DPCG"` silently falls through to plain
`spsolve`, so the deflation solver has never actually run through the torch path.
Pre-existing, unrelated to the unification.

**Result:** 46 passed, 3 xfailed throughout; enum identity and numeric equivalence both
explicitly verified.

---

## Phase 4 — Fix and relocate the QOI / material-model layer

Committed as part of `9efef66`. The biggest phase so far — real content changes, not just moves.

**Thermal made torch-native:**
- Added `get_thermal_material_model_scaling_torch` to `pyto/autodiff/material_model.py`,
  mirroring the structural twin exactly.
- Rewrote `HexThermalFEA.__init__`/`.solve()` to assemble K(x) with torch ops and call
  `pyto.autodiff.sparse_solve`, mirroring `HexStructuralFEA`'s pattern. Also fixed
  `postprocess()` and `get_thermoelastic_force()` (plain-NumPy math on `self.sol`, which is
  now torch) by converting at the boundary.
- `test_mma_thermal_runs` flipped from `xfail` to a real passing assertion — thermal MMA now
  genuinely optimizes end-to-end.
- **Deliberately not done:** moving `hex_thermal_fea.py`/`hex_structural_fea.py` into
  `physics/thermal/`/`physics/structural/` — doing only one would leave them asymmetric;
  both move together later.

**QOI / reference_adjoint split** (of `topopt_obj_cons_sensitivities.py`, the file mixing
both paradigms):
- `pyto/autodiff/qoi/`: `compliance.py`, `volume_fraction.py`, `stress.py` (moved unchanged),
  plus two **new** torch-native QOIs: `mass.py`, `gvector.py`. `MAX_VONMISES_STRESS` needed
  no new function — reuses `compute_pnorm_stress_autograd`'s existing second return value.
  `qoi/__init__.py` hosts the dispatchers, rewritten so every branch returns a plain torch
  scalar (no more `(value, numpy_gradient)` tuples).
- `pyto/autodiff/reference_adjoint/legacy_sensitivities.py`: every hand-derived NumPy
  formula, moved verbatim — test-oracle only, no driver imports it.
- Old flat `topopt_obj_cons_sensitivities.py` is now a shim re-exporting both.

**`stress_scaling` off the function-attribute global:** added `stress_scaling: float = 1.0`
to `TOParams`; `compute_constraint_and_gradient` reads/writes `to_params.stress_scaling`
instead of a mutable function attribute. Regression-tested: two separate `TOParams()`
instances no longer share this state. **`SIMP_PENALTY`'s mutators left alone** — confirmed
dead code (never called anywhere); refactoring inert code wasn't justified.

**Merge-fallout bugs found and fixed** (same class as Phase 0's — only breaks once exercised):
- `hex_structural_fea.py` `postprocess()`: same bare `EVOID_RELATIVE`/`PNORM_EXPONENT` issue,
  never caught before because nothing in the test suite called `postprocess()` until now.
- `pyto/autodiff/qoi/__init__.py`'s constraint branch referenced `TO_QOI.STRESS_SAFETY_FACTOR`,
  which **doesn't exist** — the real member is `STRESS_FAILURE_FACTOR`, actively used by
  `topopt_structural_benchmarks.py`'s `LBracketTopLoad_Vol_StressFailureFactor`. Fixed the
  reference (and a matching cosmetic string-comparison bug in `topopt_mma.py:297`).
  **Did not touch** the constraint formula's `1.0/constraintLimit` term, which looks
  semantically suspicious given `STRESS_FAILURE_FACTOR`'s "opposite of safety factor"
  convention — flagged in-code, not fixed (couldn't verify the intended physics).

**Test coverage added:** FD checks for MASS/GVECTOR/MAX_VONMISES_STRESS, a thermal compliance
FD check (both-physics coverage — needed a larger step size than structural's; verified this
is pure floating-point step-size sensitivity, not a formula bug), two direct value
comparisons against `reference_adjoint` as oracle, and a regression test for the
`stress_scaling` fix. **Interesting finding, not a bug:** the first `MAX_VONMISES_STRESS`
gradient test failed at uniform `x0=0.5` — traced to an exact numerical tie between two
mesh-symmetric elements where `max()` has a genuine kink (argmax flips at a 1e-7
perturbation); fixed by testing at a non-uniform `x0` instead of loosening the tolerance —
confirmed ~1e-9 relative error once the tie is broken.

**Result:** 54 passed, 2 xfailed (only `test_oc_structural_runs` remains, correctly deferred
to Phase 5).

---

## Phase 5 — Unify and relocate optimizer drivers

Committed as part of `9efef66`. Larger than expected: two of the four drivers turned out to have
a second, deeper problem beyond what Phase 0's `xfail`s had already caught.

**Foundational moves (done):**
- `topopt_common.py` → `pyto/topopt/common.py`, `topopt_filters.py` → `pyto/topopt/filters.py`,
  `mmaWrapper.py` → `pyto/topopt/drivers/mmaWrapper.py` — mechanical moves, shims, tested.
- **Corrected a Phase 4 misclassification**: `compute_objective_topological_sensitivity_compliance`,
  `computeStructuralTopologicalSensitivity`, `compute_pnorm_stress_and_TS`,
  `computeThermalTopologicalSensitivity` had been grouped into `autodiff/reference_adjoint/`
  (test-oracle-only) alongside the true legacy-adjoint functions. That was wrong: topological
  sensitivity is a genuinely different mathematical object from a gradient (an asymptotic
  small-hole-insertion expansion, not `d(objective)/d(density)`), and `topopt_pareto.py` calls
  these as its actual production algorithm, not for testing. Moved to a new
  `pyto/topopt/topological_sensitivity.py`, with its own docstring explaining why it isn't
  superseded by anything torch-native. `reference_adjoint`'s docstring claim ("nothing in
  pyto.topopt imports from this module") is accurate again as a result.
- New `pyto/topopt/drivers/_shared.py`: factors the ~40-line setup block (density filters,
  elements-with-forces, element stiffness matrix, nodal body force) that was copy-pasted
  across all three drivers into `setup_driver_state()`, plus `torch_sparse_filter()` for the
  torch-sparse copy of the filter needed inside an autograd chain.

**MMA (done):** `topopt_mma.py` → `pyto/topopt/drivers/mma.py`, refactored to call
`setup_driver_state()`/`torch_sparse_filter()` instead of its own inline copy of the same
logic. Already-working torch-autograd chain unchanged. Tested green.

**OC (done — the real fix):** `topopt_ocm.py` → `pyto/topopt/drivers/oc.py`. Found this driver
had **two** separate bugs, not the one already `xfail`-documented:
1. The known one: `x` was a plain NumPy array passed straight into `fe_solver.solve()`, which
   requires torch. Fixed by rewriting the loop to build the same
   `x_raw (requires_grad) → solve → objective → torch.autograd.grad` chain `mma.py` already
   uses, converting back to NumPy only at the boundary (for the OC bisection update itself,
   which is legitimately NumPy control flow, not sensitivity computation).
2. **A second, previously-masked bug**: `topopt_ocm.py` called
   `compute_objective_and_gradient(feaMode, to_params, sol, x, fe_solver, KE)` — a
   `feaMode`-first signature that doesn't match the current dispatcher
   (`compute_objective_and_gradient(to_params, sol, x, fe_solver, KE, material_model)`) at
   all. This is `main`'s own independent evolution of the call convention (`topopt_ocm.py`
   auto-merged cleanly, unlike `topopt_mma.py` which kept `automatic_differentiation`'s
   version entirely via the earlier merge-conflict resolution) — never caught before because
   bug #1 crashed first, every time, before execution ever reached this line. Fixed by
   dropping `feaMode` from `topopt_optimality_criteria`'s signature entirely (nothing else in
   the function used it; the current dispatcher doesn't need it, since it infers structural
   vs. thermal from `KE`'s shape generically through torch ops).
   Also found and fixed 4 call sites where `fe_solver.mesh.setPseudoDensity(x)` was passed a
   bare NumPy `x` — `setPseudoDensity()` now unconditionally does `rho.detach().cpu().numpy()`
   internally (part of the same broader torch migration touching `hex_mesher.py`), so those
   raised `AttributeError: 'numpy.ndarray' object has no attribute 'detach'` immediately
   after fixing bug #1. Wrapped each in `torch.tensor(...)`, matching what `mma.py` already
   does at its own `setPseudoDensity` calls.
   Verified with a real run (not just the smoke test): objective decreased monotonically
   across iterations (13.53 → 8.48 → 6.07 → 3.88), confirming the OC update itself is
   correct, not just crash-free. `test_oc_structural_runs` promoted from `xfail` to a real
   passing assertion.

**Levelset (done):** `topopt_levelset.py` → `pyto/topopt/drivers/levelset.py`. Fully
self-contained — its own `compute_compliance_and_sensitivity(feaMode, rho, fe_solver, void)`
(pure NumPy shape-derivative sensitivity, no SIMP penalization; level-set density is binary
0/1), no dependency on `topopt_obj_cons_sensitivities.py`/`pyto/autodiff/qoi`/
`pyto/topopt/topological_sensitivity` at all. Same bug class as OC's #1, at more call sites:
`fe_solver.solve(rho, material_model)` (3 sites) and `mesh.setPseudoDensity(...)` (3 sites)
both received plain NumPy; `compute_compliance_and_sensitivity` also reads `fe_solver.sol`
directly (not passed as a parameter) rather than a value returned from `solve()`, so it needed
its own NumPy conversion at the top, mirroring `hex_thermal_fea.py`'s Phase 4 fix. No
`torch.autograd.grad` chain needed anywhere here — Levelset's sensitivity is a hand-derived
NumPy shape derivative, not a torch gradient, so `rho` never needs `requires_grad=True`, just
to be a plain torch tensor at the solve/mesh boundary. Verified with a real run: volume
fraction trends steadily toward the 0.5 target (0.889 → 0.660 over 40 iterations on the small
fixture) — a real, converging optimization, not just "doesn't crash."

**Pareto (done):** `topopt_pareto.py` → `pyto/topopt/drivers/pareto.py`. Same solve/
`setPseudoDensity` bug class as Levelset, at 2 + 7 call sites. One extra wrinkle Levelset
didn't have: `sol` itself has to be converted back to NumPy immediately after `solve()`
(`.detach().cpu().numpy()`), not just the inputs going in — confirmed by first attempting the
fix without it and hitting `TypeError: unsupported operand type(s) for *: 'numpy.ndarray' and
'Tensor'` inside Pareto's local `compute_objective_topological_sensitivity_compliance`, which
mixes `sol` into plain NumPy expressions (`np.dot(...) * sol[dofMat]...`) — NumPy silently
coerces one torch operand via `__array__` but not both sides of a `*`. Verified with a real
run to full completion (not truncated): volume fraction traces cleanly from 1.0 down to the
exact target 0.5 across 31 FEA solves, `success=True`, in well under a second — confirming
Pareto's own topological-sensitivity update logic is correct, not just crash-free.

**Known duplication flagged, not fixed** (discovered while reading `topopt_pareto.py` in
full to plan this): it does **not** call the shared `pyto/topopt/topological_sensitivity.py`
module moved out of `reference_adjoint/` earlier in this same phase — it carries its own
second, independent copy of the same four formulas
(`compute_objective_topological_sensitivity_compliance`,
`computeStructuralTopologicalSensitivity`, `compute_pnorm_stress_and_TS`,
`computeThermalTopologicalSensitivity`), with an incompatible physics-dispatch convention:
`pareto.py`'s local copy takes an explicit `feaMode` and reads
`mesh.edofMatStructural`/`mesh.edofMatThermal` directly (patch-independent); the shared
module infers physics from `KE.shape[0]` and reads `mesh.edofMat` (needs the manual
`mesh.edofMat = mesh.edofMatStructural` patch from `tests/conftest.py`, Phase 0). Consolidating
them means first resolving that `edofMat`-vs-`edofMatStructural`/`edofMatThermal` split across
the codebase (also touching `hex_structural_fea.py`, `pyto.autodiff.qoi.compliance`, ...) — a
separate, cross-cutting cleanup, deliberately not attempted here (same category of decision as
Phase 3's DPCG finding). Documented in both files' docstrings so it's discoverable without
re-deriving it.

**Why gradient-checking wasn't the right test for Pareto/Levelset** (a real distinction from
Phase 4, not just a smaller effort bar): Phase 4's QOI tests all did finite-difference
gradient checks because every QOI there is a true `d(objective)/d(density))`. Topological
sensitivity (Pareto) and shape sensitivity (Levelset) are not that — an asymptotic
small-hole-insertion expansion and a boundary-velocity field, respectively, neither of which
FD-checks meaningfully against a scalar density perturbation. The validation used instead:
(1) **value agreement, not gradient agreement** — `test_pareto_compliance_value_matches_compute_compliance_torch`
checks that Pareto's own compliance formula and `compute_compliance_torch` agree at the same
`(x, sol)`, since both reduce to the same `u^T K(x) u` sum via different code paths, catching
indexing/formula slips from the torch/NumPy boundary fix without claiming to validate a
gradient that isn't one; (2) **convergence sanity** — `test_levelset_structural_runs`/
`test_pareto_structural_runs` check that volume fraction moves toward the target and the
objective stays finite, since neither driver had a "pre-fix" numeric baseline to compare
against (both crashed on iteration 1 before this phase).

**Result:** 58 passed, 1 xfailed (only the pre-existing, out-of-scope
`hex_transient_thermal` import issue from Phase 0 remains). Gate met — `test_drivers_smoke.py`
green for all 4 drivers (MMA, OC, Pareto, Levelset) on structural; MMA also verified on
thermal.

### Independent verification: bit-for-bit topology comparison against `main`'s pre-migration originals

Beyond the test suite, directly checked that fixing Pareto/Levelset's torch/NumPy boundary
didn't change their actual behavior — not just "doesn't crash," but "produces the identical
optimized topology as before the migration." `main`'s own `HexStructuralFEA`/`HexThermalFEA.solve()`
predates the torch-autodiff migration entirely (confirmed: `main`'s `solve()` takes
`x: np.ndarray`, uses `linear_solvers`/plain NumPy throughout) — so `main`'s own
`topopt_pareto.py`/`topopt_levelset.py`, completely unmodified, are a genuine working
pre-migration baseline, not a hypothetical one.

Checked out `main` into an isolated `git worktree` (so `AutoPyTO`'s uncommitted Phase 0-5
work was never touched), ran both drivers there on the same fixture problems used throughout
this plan (`ShortCantileverTipLoad` structural, `HeatPlate` thermal), and diffed the results
against this branch's fixed-and-moved versions:

| Driver | Physics | Final topology match | Objective max rel. diff | Volfrac max abs. diff |
|---|---|---|---|---|
| Pareto | structural | **288/288 elements identical** | 6.7e-14 | 1.8e-14 |
| Levelset | structural | **288/288 elements identical** | 5.5e-14 | 0.0 |
| Pareto | thermal | **200/200 elements identical** | 2.4e-11 | 1.8e-11 |
| Levelset | thermal | **200/200 elements identical** | 4.6e-11 | 0.0 |

Every final topology matches exactly, element-for-element. The objective/volume-fraction
traces agree to floating-point noise (1e-11 to 1e-14 — the thermal runs are a couple of
orders larger than structural's, consistent with a different but equivalent floating-point
summation order between the torch sparse solve and SciPy's, not a real discrepancy). This
confirms directly, not just by code inspection, that Pareto and Levelset's own
topological-sensitivity and shape-sensitivity formulas were genuinely left untouched — the
fix only changed the torch/NumPy interface at the `solve()`/`setPseudoDensity()` boundary, as
intended, with zero effect on the actual optimization algorithm or its result.

---

## Phase 6 — Relocate I/O, examples/benchmarks, and the GUI (last)

Committed as `09cb5ee` (file renames landed a commit earlier, in `9efef66`, because `git mv`
stages both sides of a rename immediately — the shims/fixes/`__main__` bug landed in `09cb5ee`).

**Moved:**
- `io/`: `stl_reader.py`, `project_manager.py`, `solidworks_interface.py`,
  `testSolidWorksInterface.py`, `topopt_stl_recovery.py`.
- `examples_benchmarks/`: all 12 `*_examples.py`/`*_benchmarks.py`/`topopt_run_benchmarks.py`/
  `topopt_solver_comparison.py` files. None define `__all__` or use underscore-prefixed
  public names, so their shims are plain `from pyto.examples_benchmarks.X import *` — no
  explicit re-export lists needed, unlike earlier phases' shims.
- `gui/` (last, as planned): `PyTOGUI.py`, `PyTO_demos.py`, `hex_plotter.py`, and the 3 icon
  PNGs. Icons moved **alongside** `PyTOGUI.py` itself (not a `resources/` subfolder as Phase
  1's mapping guessed before reading the code) — its `get_icon()` method does
  `os.path.dirname(__file__)` directly, no subfolder.

**Found and fixed while moving:**
- **`__file__`-relative `'../Models/...'` paths broke** in 4 places, since these files moved
  2 directories deeper than `script_dir = os.path.dirname(os.path.abspath(__file__))`
  assumed: all 7 `examples_benchmarks/` files with that pattern, `project_manager.py`'s two
  `Models/` lookups, and the `__main__`-block demo paths in `stl_reader.py`/
  `topopt_stl_recovery.py`. Fixed by adding the right number of extra `os.path.dirname(...)`
  calls (verified precisely with a throwaway Python snippet before applying, not by
  eyeballing path arithmetic) so each resolves to the same absolute `Models/` path as before
  the move.
- **A systemic bug across every driver/script shim from Phases 5-6**: a shim doing
  `from pyto.X.Y import *` does NOT make `python <flatfile>.py` run `Y`'s own
  `if __name__ == "__main__":` block — under a plain import, `__name__` inside `Y` is
  `"pyto.X.Y"`, never `"__main__"`. Found while writing `PyTOGUI.py`'s shim more carefully
  than the earlier ones. Fixed in every affected shim (`mma.py`, `oc.py`, `pareto.py`,
  `levelset.py`, `stl_reader.py`, `project_manager.py`, `solidworks_interface.py`,
  `topopt_run_benchmarks.py`, `topopt_stl_recovery.py`, `PyTOGUI.py`) by adding
  `runpy.run_module("pyto.X.Y", run_name="__main__")` inside the shim's own
  `if __name__ == "__main__":` block. This had been silently broken since Phase 5 — worth
  remembering as a general lesson for Phase 7's remaining moves: **every new shim for a file
  with a real `__main__` block needs this pattern, not just `from X import *`.**
- **`PyTOGUI.py` was wrongly marked untestable back in Phase 0**: `_SKIP`'s reason said "opens
  a Qt/VTK window at import time," but `QApplication` creation is properly guarded behind
  `if __name__ == "__main__":` (line ~5729). Verified directly (subprocess import with a
  timeout, no window opened) rather than trusting the old assumption, then removed it from
  `test_imports_smoke.py`'s `_SKIP` set — it now gets real import coverage.

**Result:** 59 passed, 1 xfailed (unchanged pre-existing issue; +1 test from `PyTOGUI` now
being genuinely covered).

**Known gap, not fixed here:** `main.ipynb` needs no import changes (its flat imports resolve
through the shims unchanged), but it does `from topopt_mma import run_topopt_mma` — and no
`run_topopt_mma` function exists anywhere in `mma.py` (only `topopt_mma(...)` does; the other
three drivers each have a `run_*` convenience wrapper, `mma.py` never got one). This is
pre-existing and unrelated to the restructuring — not introduced or worsened here, just
flagged since it was noticed while checking the notebook's imports for this phase.

**Gate:** automated tests green (met). The manual GUI smoke-test checklist (launch the app,
run one structural and one thermal optimization end-to-end through the UI) has **not** been
run — it needs a human; there's no way to drive a PyQt5/VTK desktop app from this environment.
Suggested checklist for whoever does this:
1. `cd src && python PyTOGUI.py` (or the equivalent launch path) — does the window open at all?
2. Load or define a structural problem, run a short MMA or OC optimization through the UI —
   does it complete and show a result?
3. Same for a thermal problem.
4. Anything involving `hex_plotter`/`HexFEAPlotter` (mesh, deformation, stress plots) — do
   plots render?

---

## Phase 7 — final move, import-qualification pass, and shim removal

**Complete.** All three parts of the gate are done: the 12 remaining flat files are moved,
every internal import across `pyto/` is `pyto`-qualified, and all ~50 compatibility shims
are deleted. Full suite: **81 passed, 1 xfailed** (up from 59+1 pre-Phase-7 — the increase is
`test_imports_smoke.py` now parametrizing over 69 individually-discovered `pyto` submodules
instead of ~50 flat files, a finer-grained partition of the same coverage, not new tests).

### Part 1 — moved the last 12 files

Read `hex_mesher.py` (1,513 lines) and `hex_element_stiffness.py` in full first, as planned,
since they're the two most-depended-on files left. Moved one at a time in dependency order
(mesher → element stiffness → structural physics → thermal physics → thermoelastic, so each
file's own internal imports could reference already-moved siblings), testing after each:

| File | Target | Notes |
|---|---|---|
| `hex_mesher.py` | `core/hex_mesher.py` | imports `stl_reader`, `hex_plotter` fixed; `__main__` STL path depth fixed (1x→3x `dirname`) |
| `tet_mesher.py` | `core/tet_mesher.py` | no internal `pyto` imports needed; same 1x→3x path fix |
| `hex_element_stiffness.py` | `physics/hex_element_stiffness.py` | only needed `mat_lib` fixed |
| `hex_structural_fea.py` | `physics/structural/` | |
| `tet_structural_fea.py` | `physics/structural/` | |
| `hex_largeDeformationFEA.py` | `physics/structural/` | |
| `hex_modal_fea.py` | `physics/structural/` | |
| `hex_thermal_fea.py` | `physics/thermal/` | |
| `tet_thermal_fea.py` | `physics/thermal/` | |
| `hex_transient_thermal.py` | `physics/thermal/` | still xfails on the same pre-existing `mat_lib.ThermalMaterial` bug, unaffected by the move |
| `tet_transient_thermal.py` | `physics/thermal/` | depends on `tet_thermal_fea` (moved first) |
| `check_thermoelasticity_sensitivity.py` | `physics/thermoelastic/` | `__main__` STL path depth fixed (1x→4x `dirname`, verified against the real `Models/Beam/beam.STL` path before trusting it) |
| `topopt_thermostructural_sensitivity.py` | `physics/thermoelastic/` | flagged in its own docstring: a second, independent `ThermoElasticSensitivity` implementation vs. `check_thermoelasticity_sensitivity.py`'s — not consolidated, same category as Phase 5's Pareto/`topological_sensitivity.py` duplication note |

`script_dir = os.path.dirname(os.path.abspath(__file__))` in `hex_structural_fea.py`,
`hex_thermal_fea.py`, and `hex_modal_fea.py` is dead code (computed, never used) — left as-is,
no path-depth fix needed. Separately noticed but **not fixed** (pre-existing, out of scope,
same category as the Phase 3 DPCG finding): `hex_structural_fea.py`'s own `__main__` block
references `linear_solvers` and `time` without importing either — would `NameError` if ever
run directly; never exercised by the test suite.

Added `.gitignore` entries for `*.nbc`/`*.nbi` (numba's `@njit(cache=True)` disk cache files,
generated freely by `hex_element_stiffness.py`'s functions) and a general `__pycache__/`
pattern — the existing `.gitignore` only covered `/__pycache__` and `/src/__pycache__` at the
repo/src root, not the now much-deeper `pyto/physics/__pycache__` etc.

**Gate:** tests green after each individual move (stayed at 59 passed, 1 xfailed throughout,
matching Phase 6's exit state) — met.

### Part 2 — qualified every internal import across `pyto/`

This was the actual bulk of Phase 7: every file moved in Phases 2-7 still imported its
sibling modules by old flat name internally (`import mat_lib`, `from bound_cond import ...`)
— that's precisely what the shims existed to paper over, and it's what made "just delete the
shims" impossible until this pass ran. Built a small mapping (flat name → `pyto.`-qualified
path, one entry per moved module) and applied it as `import <target> as <original_alias>` /
`from <target> import ...` substitutions across every `.py` file under `src/pyto/`, preserving
each file's original local names (so nothing inside a function body needed touching, only the
import line itself).

One case needed manual attention beyond the mechanical substitution: `topopt_obj_cons_sensitivities.py`
was Phase 4's split file (torch-native QOIs → `autodiff/qoi/`, legacy NumPy adjoint formulas →
`autodiff/reference_adjoint/`), so `from topopt_obj_cons_sensitivities import *` in the driver
files doesn't map to a single target. `mma.py`/`levelset.py`/`pareto.py` only ever used the
QOI-dispatcher half, but `oc.py` also calls `compute_volume_constraint_and_gradient` (a plain
analytic NumPy formula — trivial closed-form gradient of a mean, deliberately not migrated to
torch autograd), which only exists in `reference_adjoint/legacy_sensitivities.py`. Caught this
via the test suite itself (`test_oc_structural_runs` failed with `NameError:
compute_volume_constraint_and_gradient` after the bulk substitution — exactly the kind of
regression this pass was at risk of introducing silently), fixed by adding the one explicit
import `oc.py` actually needs from `reference_adjoint` alongside its `autodiff.qoi` import.

**Gate:** full grep across `src/pyto/` and `tests/` for any remaining bare `import <name>` /
`from <name> import` referencing an old flat module name returns nothing (verified twice —
once right after the bulk substitution, once again after Part 3's test-suite conversion) — met.

### Part 3 — converted `tests/` and `main.ipynb`, deleted the shims

The Phase 7 gate as originally scoped ("grep outside `pyto/` returns nothing") also covers
`tests/` and `main.ipynb`, both of which relied on the shims for their own flat imports
(`import hex_structural_fea`, `from topopt_material_model import MaterialModel`, etc.) — left
that way deliberately through Phases 0-6 since rewriting tests wasn't those phases' concern.
Converted all 4 test files + `conftest.py` to `pyto`-qualified imports the same way.

`test_imports_smoke.py` needed a real rewrite, not just an import-line substitution: it used
to `glob.glob("src/*.py")` and import each flat module by filename — with the shims gone,
that glob would find nothing. Replaced the discovery mechanism with
`pkgutil.walk_packages(pyto.__path__, prefix="pyto.")`, which recursively finds every real
submodule inside the package (69 of them). `_SKIP` and `_XFAIL_IMPORTS` keys changed from bare
filenames (`"PyTOGUI"`) to dotted module paths (`"pyto.gui.PyTOGUI"`) to match.

`main.ipynb`'s import cell (`from topopt_structural_benchmarks import *`, etc.) was rewritten
to `pyto`-qualified imports the same way, including the already-known-broken
`from topopt_mma import run_topopt_mma` line (real function is `topopt_mma`, not
`run_topopt_mma` — flagged in Phase 6, still not fixed, deliberately: it's a pre-existing bug
unrelated to the restructuring, and this phase's job was to relocate imports faithfully, not
silently fix bugs it happens to walk past). Verified by running the notebook's import cell's
equivalent code directly: every import succeeds except that one, which fails with the exact
same `ImportError` as before the move — confirms the move neither introduced nor silently
papered over the pre-existing bug.

Verified `PyTOGUI.py` still imports safely with no window opened (same subprocess-style check
as Phase 6, re-run after the shim deletion since its own imports changed in Part 2).

Deleted all ~50 remaining flat shim files under `src/*.py` (every one confirmed via grep to
contain the `"""Compatibility shim: ...` docstring marker before deletion — nothing deleted
without first confirming it wasn't a real file). `src/__init__.py` (empty, predates this plan,
not itself a shim) was left in place.

**Gate:** full suite green with shims removed — **81 passed, 1 xfailed**. Met.

### What's still flagged, not fixed (unchanged from earlier phases, listed here for continuity)

- `mesh.edofMat` vs. `mesh.edofMatStructural`/`edofMatThermal` convention split (Phase 5).
- Pareto's local topological-sensitivity duplication vs. `pyto.topopt.topological_sensitivity`
  (Phase 5), now joined by `topopt_thermostructural_sensitivity.py`'s duplicate
  `ThermoElasticSensitivity` vs. `check_thermoelasticity_sensitivity.py`'s (this phase).
  Genuinely inert `SIMP_PENALTY` mutators, the DPCG dispatch gap (Phase 3), `hex_structural_fea.py`
  `__main__`'s missing `time`/`linear_solvers` imports (this phase), and `main.ipynb`'s
  `run_topopt_mma` `ImportError` (Phase 6/this phase) — none touched, all pre-existing and out
  of this plan's scope (autodiff correctness + file layout, not general bug-hunting).
- Manual GUI smoke-test checklist (launch `PyTOGUI.py`, run one structural and one thermal
  optimization end-to-end through the UI) — still requires a human; not run this session.

This restructuring plan's phases (0 through 7) are now all complete on `AutoPyTO`, committed
locally, not pushed.
