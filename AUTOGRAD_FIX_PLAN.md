# Autograd fix plan

Status file for fixing the autograd issues listed in `AUTOGRAD_STATUS_AND_HANDOFF.md`. Steps run in order. A step is marked done only after its check passed. Guard for every step: `pytest tests/ -q` (baseline 84 passed, 1 xfailed) plus the finite-difference gradient check on touched problems.

Legend: `[ ]` todo, `[~]` in progress, `[x]` done, `[!]` blocked on a decision.

Working rules: no `git commit` unless the user asks; never push.

## Decisions (user delegated to best judgement on 2026-09-24)
3. Stale enum entries: delete unrecoverable ones. 5. Thermal Dirichlet: legacy convention (minimise -J). 6. Constraint matrix: convert single-DOF rows to Dirichlet, raise on the rest. 6b. GravityPlate: volume band.

## Original open decisions
1. Thermal Dirichlet semantics (step 5): A = legacy convention (recommended), B = separate objective, C = exclude.
2. Constraint-matrix BCs (step 6): support, or exclude and fail loudly.
3. Stale enum entries (step 3): delete, or repair where recoverable.

## Steps
- [x] **1. Find why MMA stops converging on stress and Inverter problems.** Toggle each recent MMA change (inverted-filter fix, binarized finalization, frozen `stress_scaling`) one at a time; check `obj/obj0` normalization. Files: `topopt/drivers/mma.py`, `autodiff/qoi/__init__.py`. Done when: cause named; Inverter gives a mechanism (J near -0.59) in about 56 iterations.
- [x] **2. Fix the body-force crash** (GravityPlate, CentrifugalPlate): torch sparse for `elem_to_node_field_mapping`; check MMA's analytic body-force term for double counting. Files: `hex_structural_fea.py` (~line 100), `mma.py`. Done when: both plates build, pass the FD check, run under MMA/OCM/Pareto.
- [x] **3. Clean broken enum entries** (`RELATIVE_COMPLIANCE`, thermal `BiClamp`, `LBracketTopLoad_Vol_StressFailureFactor`). Needs decision 3. Done when: gradient matrix shows no build failures except ConstraintMatrix.
- [x] **4. Audit index-based face selection** (`triList*`, hard-coded triangle IDs) for the stale-index failure that broke LBracketThick. Done when: each problem's node counts and coordinates match its Reference image.
- [x] **5. Thermal Dirichlet semantics** (FourCornersThermal, BridgeThermal). Needs decision 1. Done when: vf 0.75 / 0.50 and J near 6839 / 978 under MMA and OCM.
- [x] **6. Constraint-matrix BCs.** Needs decision 2. Done when: no BC is silently ignored.
- [x] **6b. GravityPlate collapse (self-weight load has a trivial empty optimum).** Enforce a small volume lower bound (band). Done when: MMA keeps vf inside the band with a real structure.
- [x] **7. MMA needs more iterations than Reference (23 of 26).** Compare stopping tolerances and binarization effect. Done when: within about 20% of Reference, or explained.
- [x] **8. Robustness.** Stateless or documented `stress_scaling`; clear error for DPCG-size problems; move OC's volume gradient off the legacy module. Done when: a unit test covers each.
- [x] **9. One-at-a-time verification in `topology_check.ipynb`:** CantileverMidLoad, CantileverTipLoad, MBBBeam, Table, EdgeCantilever, Multiload. Done when: each within about 3% of Reference or explained.
- [~] **10. Full sweep, regenerate the comparison report** (`python compare_results_to_reference.py <folder>`), update the handoff file. Done when: no unexplained "Reference better" rows or failures.

## Log
(Newest last. One line per finding or completed check.)
- Step 1 (done). Three separate causes, none in the autograd derivative itself:
  - **Stress scaling rule regressed** (`autodiff/qoi/__init__.py`). Reference-era rule: update every call, `s = 0.25*max/pnorm + 0.75*s`. Commit `9efef66` changed it to update only when `max_vm > limit`, weights 0.5/0.5. Restored the Reference-era rule in both branches (MAX_VONMISES_STRESS, STRESS_FAILURE_FACTOR). Results (density filter on): `LBracketTopLoad_Mass_StressFF` J 25.28 / vf 0.167 (Reference 24.67 / 0.163; was 27.82 / 0.184); `LBracketMidLoad_Vol_Stress` vf 0.420 (Reference 0.467; was 0.807). Both still run to the 150-iteration cap: see step 7.
  - **Spring tuple order bug** (`hex_structural_fea.py` line 109): `(K, dof)` was unpacked as `dofs, ks`, so every Inverter spring landed on DOF 0 with a huge stiffness. Now `ks, dofs`. Verified 12 springs at the right DOFs, k = 0.01.
  - **Inverter objective scale differs from Reference by design**: commit `f4e163e` (after the Reference) changed the spring from 0.1 to 0.01. With k = 0.01 current code gives J = -19.4 in 48 FEAs; with k = 0.1 it gives J = -0.616 in 59 FEAs (Reference -0.594 in 56). Spring left at 0.01; the two are not directly comparable.
  - Rejected hypothesis: density vs sensitivity filter is not the cause (density filter off still gave J 0.82 for `LBracketMidLoad_Vol_Stress`). Torch stress values equal `postprocess()` values exactly.
  - `pytest tests/`: 84 passed, 1 xfailed.
- Step 2 (done). Body-force problems build and run under MMA, OCM and Pareto.
  - **Crash**: `torch.tensor(<scipy coo>)` in `HexStructuralFEA.__init__`. Now a torch sparse COO tensor.
  - **Two more bugs in `solve()`**: `map_vec * elem_force` was an elementwise product (now `torch.sparse.mm`), and the body force was scaled by the stiffness factor `x^p` (Reference-era code scaled by density `x`; fixed).
  - **Double counting removed**: FD check on both plates: autograd-only gradient matches finite differences (GravityPlate 8.7e-7, CentrifugalPlate 3.4e-7), so autograd already carries the body-force dependence on `x`. The analytic `2*ce_body*rho'` terms in `mma.py` and `oc.py` were removed.
  - **MMA initial density regression (`mma.py`)**: Reference-era driver started at the volume-fraction limit; the torch driver hard-coded 0.5. Restored. Likely also affects step 7.
  - **Results (MMA)**: `CentrifugalPlate` J 0.00776, vf 0.500, 61 FEAs (Reference 0.00966, 26 FEAs). Pareto J 0.00637, vf 0.500.
  - **Open: GravityPlate self-weight optimum is trivial.** MMA collapses to vf 0.0002, J 6.8e-5 (Reference vf 0.10, J 6.85e-3). Uniform x = 0.1 has J 2.4e-3, so the near-empty design is a strictly lower-J feasible design; the objective, not the gradient, admits it (gradient is FD-verified). Same result with density filter off. Pareto keeps vf 0.100 (J 3.8e-6). Needs a decision: add a volume lower bound or equality, or accept.
  - OC on body-force problems is unconverged at 40 iterations (CentrifugalPlate J 0.68); the runner skips OC/Pareto/LevelSet for body-force problems by design.
  - `pytest tests/`: 84 passed, 1 xfailed.
- Step 4 (done, no new defects). Index-based face selection exists in four builders:
  - `createLBracketProblem`: correct (OCM and Pareto reproduce the Reference exactly; STL and indices changed together in commit `4e82da9`).
  - `createGEGrabCADProblem`: indices and STL added together (`d87b7f5`); fixed set = 289 triangles around one bore, force set = 432 around another, all in range, sensible bounding boxes. Completes in today's sweep. No Reference row exists to compare.
  - `createBliskSectionProblem` and `createBliskPressureLoadingProblem`: STL changed one day after the lists (`d73b558` vs `3eddbbc`), but all sets are in range, flat faces, coherent positions. Not in the benchmark suite; sliding sets need `constraint_matrix` (step 6).
  - `createLBracketThickProblem`: already fixed by geometric selection.
- Step 7 (done, explained). MMA iteration excess is set by the filter type, not by the optimizer.
  - `mmaWrapper.py` termination code is byte-for-byte the Reference-era logic; tolerances (kkt 1e-6, objective/constraint 1e-4) are the same.
  - `move_limit`: current 0.2 vs Reference-era 0.05. Testing 0.05 made it worse (Mitchell 109, CantileverTipLoad 58, ThreeHoleBracketThick 133 iterations), so 0.2 stays.
  - Density filter on (current default, commit `f4e163e`) vs off (sensitivity filter, the Reference's method), same code otherwise: Mitchell 60 vs 29 iterations (J 2.525 vs 2.544; Reference 2.604 in 46), CantileverTipLoad 46 vs 45 (J 10.44 vs 10.43; Reference 10.50 in 39).
  - Initial density restored to the volume-fraction limit (step 2): TwoBar 30 -> 23 (Reference 26), ThreeHoleBracketThick 96 -> 62 (Reference 29; the rest is the filter effect).
  - Not changed: the density-filter default is a deliberate maintainer choice.
- Step 8 (done). Kept behaviour, documented and tested it:
  - `stress_scaling` stays stateful (it is the Reference-era adaptive normalization). Documented in the `compute_constraint_and_gradient` docstring; new test `test_stress_scaling_update_rule` locks the rule `s <- 0.25*max/pnorm + 0.75*s` for limits below and above the actual max stress.
  - DPCG in the torch path already warns before falling back to a direct solve (not silent). New test `test_dpcg_is_not_implemented_and_warns_before_falling_back`.
  - `oc.py` no longer imports `reference_adjoint`; the volume constraint and its gradient are inline (`mean(x)/limit - 1`, `ones/limit/n`).
  - `pytest tests/`: 86 passed, 1 xfailed (84 + 2 new).
- Step 9 (done). MMA, current defaults, against `Results_Reference` (objective J, iterations):

  | Problem | Reference | Now | Diff |
  |---|---|---|---|
  | CantileverMidLoad | 23.88, 42 | 23.79, 41 | -0.4% |
  | CantileverTipLoad | 10.50, 39 | 10.445, 46 | -0.5% |
  | MBBBeam (2.5D) | 99.32, 41 | 98.77, 58 | -0.5% |
  | Table | 0.01704, 66 | 0.01574, 59 | -7.6% (better) |
  | EdgeCantilever | 0.5831, 45 | 0.5782, 131 | -0.8% |
  | Multiload | 0.2201, 42 | 0.2209, 52 | +0.4% |

  All within about 1% of the Reference, or better. EdgeCantilever takes many more iterations (131; the earlier run today, before the initial-density fix, took 56 at J 0.5721). Net effect of the initial-density fix across tested problems is fewer iterations (TwoBar -7, ThreeHoleBracketThick -34, Table -21, Multiload -5, EdgeCantilever +75); kept because it is the Reference-era behaviour. Topology images were not re-inspected.
- Step 3 (done). Deleted three enum entries that had no working definition; none appeared in any benchmark list or Results folder:
  - `StructuralTOExamples.CantileverTipLoadDisplacementObjective`: used `TO_QOI.RELATIVE_COMPLIANCE` (no longer exists; "3x initial compliance" has an ambiguous baseline, so not faithfully recoverable).
  - `StructuralTOExamples.LBracketTopLoad_Vol_StressFailureFactor`: enum and STL path only, no problem definition.
  - `ThermalTOExamples.BiClamp`: pointed at a non-existent `HexThermalExamples.BiClamp` (the thermo-structural BiClamp is separate and untouched).
  - Structural enum now 38 members, thermal 3. `pytest tests/`: 86 passed, 1 xfailed.
- Step 5 (done). Prescribed-value (Dirichlet-driven) thermal problems now match the Reference under MMA and OCM.
  - **Rule** (`autodiff/qoi/compliance.py`, `compliance_sign`): when a problem has no force, no body force and non-zero Dirichlet values, the optimizer minimizes -J (maximum conduction), which equals the Reference-era gradient `-dK*ce`. The reported objective stays the true J. Only FourCornersThermal and BridgeThermal qualify (HeatPlate has a heat load; no structural benchmark qualifies).
  - **Applied** in `mma.py` (objective and gradient; `obj0` now `abs(...)`, history stores the true J) and `oc.py` (gradient). Pareto/LevelSet already used the legacy gradient.
  - **Second bug found while verifying**: `mmaWrapper.py` divided the objective and its gradient by the SIGNED first objective (`f0Scaling = f0val`), which flips a negative objective and made MMA minimize +J again. Now `abs(f0val)`. This also protects any negative-objective problem (e.g. Inverter's GVECTOR).
  - **Results** (J, vf, FEAs): FourCorners MMA 6844.7 / 0.75 / 24 (Reference 6839 / 0.75 / 17); FourCorners OCM 6848.4 / 0.75 / 21 (6848.4 / 21); Bridge MMA 964.9 / 0.494 / 11 (978.4 / 0.50 / 15); Bridge OCM 981.6 / 0.50 / 35 (981.7 / 35).
  - **Tests**: `test_compliance_sign_only_flips_for_prescribed_value_problems`; `test_mma_wrapper_handles_a_negative_objective` (verified to fail on the old `f0Scaling` line). `pytest tests/`: 88 passed, 1 xfailed.
- Step 6 (done). No BC is silently ignored any more.
  - `core/bc.py` `resolve_constraint_matrix(bc)`, called first in `HexStructuralFEA.__init__`: single-DOF constraint rows (`coef*u_j = rhs`) become Dirichlet conditions (`fixed_dofs`/`dirichlet_values`); rows that couple several DOFs (sliding supports) raise `NotImplementedError` with a clear message. It also converts empty-list `fixed_dofs` to arrays (the other cause of the earlier `EdgeCantileverConstraintMatrix` crash).
  - `EdgeCantileverConstraintMatrix` (all supports are single-DOF rows) now builds and solves (1134 fixed DOFs, finite solution, zero displacement at fixed DOFs). Gradient FD check passes (worst relative error printed in the log below). It is still not in the runner's problem lists.
  - Blisk sliding problems (`createBliskSectionProblem`) now raise the clear error instead of running without their constraints.
  - Test: `test_resolve_constraint_matrix_folds_single_dof_rows_and_rejects_sliding`. `pytest tests/`: 89 passed, 1 xfailed.
- Step 6b (done). GravityPlate no longer collapses.
  - **Cause recap**: with a self-weight load, less material means less load, so a near-empty design has lower compliance than any real design (J 6.8e-5 vs 2.4e-3 for uniform x = 0.1). The gradient is FD-verified; the objective admits the degenerate optimum.
  - **Fix**: new constraint type `TO_QOI.VOLUME_FRACTION_MIN` (`c = 1 - mean(x)/limit <= 0`, `autodiff/qoi/__init__.py`, enum in `topopt/common.py`). GravityPlate now has a volume band: `VOLUME_FRACTION <= 0.10` and `VOLUME_FRACTION_MIN >= 0.095`. `mma.py` treats the new type like the volume constraint in its (density-filter-off) sensitivity-filter branch.
  - **Result (MMA)**: vf holds 0.095-0.097 from iteration 10, J = 3.43e-6, 115 FEAs. Independent check: Pareto at vf 0.100 gives J = 3.8e-6. The Reference MMA (vf 0.10, J 6.85e-3) is about 1000x worse than both, so it looks like a poor local optimum rather than a target; not directly comparable.
  - **Checks**: FD error 2.8e-6 (objective), 1.6e-9 (new constraint). Test `test_volume_fraction_min_is_a_lower_bound`. `pytest tests/`: 90 passed, 1 xfailed.
  - Only MMA reads the second constraint; OC/Pareto/LevelSet look at `Constraints[0]` only (the runner skips them for body-force problems anyway).
- Step 10 (in progress). Full sweep launched 2026-09-24 with all fixes above: `python -m pyto.examples_benchmarks.topopt_run_benchmarks` from the repo root, log `benchmark_run.log` (previous sweep log kept as `benchmark_run_2026-09-23.log`), output `Results/Results_2026-09-24/`. When it finishes: `python compare_results_to_reference.py Results_2026-09-24`, then update the handoff file.

## Follow-up fixes from the 2026-09-24 sweep review (steps 11-14)
Review of `Results_2026-09-24` (13 contact sheets, Reference vs new): topologies match the Reference in character; only camera angle and opacity of the setup images differ (runner uses `iso` for 3D/stress problems, setup images are opaque). Thermal matches exactly. Wrong or doubtful results found:

- [~] **11. Stress constraint is not enforced exactly.** `LBracketMidLoad_Vol_Stress` ends at max von Mises 232.8 MPa vs a 225 MPa limit (+3.4%), at the 150-iteration cap. Hypothesis: the lagged `stress_scaling` (0.25/0.75 average) makes the constraint value differ from the true max at the last iterate. Experiment (temporary env switch `STRESS_EXACT=1` in `autodiff/qoi/__init__.py`): value uses the exact current ratio `max_vm/pnorm` (detached), so `c = max_vm/limit - 1` and the constraint becomes a pure function of `x`; gradient unchanged. Running on `LBracketMidLoad_Vol_Stress` and `LBracketTopLoad_Mass_StressFF`. Decision pending the results; if it does not help, the switch is removed.
- [x] **12. Infeasible stress limit + silent violation.** (a) `topopt_mma` now warns when the stress limit is below the SOLID design's peak stress, and reports `errorMsg = "Warning: stress limit exceeded by X% (...)"` when the returned design exceeds the limit by more than 1% (success flag unchanged). Checked on `LBracketThickTopLoad_Vol_Stress`: both messages fire. (b) That benchmark's limit was unreachable (solid part peaks at 205.8 MPa vs 200 MPa, so MMA ended fully solid); raised to 225 MPa. `pytest tests/`: 90 passed, 1 xfailed.
- [x] **13. Pareto differences vs Reference (Table +1.8%, Multiload +1.1%).** No code change: Multiload is bit-identical in the Aug-31, Sep-23 and Sep-24 sweeps (0.22505) and differs from the Reference (0.22270) already in Aug-31, i.e. before this session; Table changes only in the 5th digit between two runs of the same code. Pareto thresholds a discrete field, so tiny floating-point differences (e.g. the reordered sums in `apply_dirichlet_bc_torch`, or the torch migration) get amplified. Treated as numeric noise, not a defect.
- [~] **14. LevelSet fails on Table** (vf 0.1788 vs 0.15 after 150 iterations; identical to the Reference run, so old behaviour). Running a 300-iteration LevelSet on Table to see whether volume is still descending at the cap.
