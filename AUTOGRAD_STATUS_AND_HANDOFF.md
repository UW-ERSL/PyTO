# Autograd status and handoff notes (PyTO, branch `AutoPyTO`)

Written 2026-09-23 as the starting point for the next phase. Everything under "Verified" was run and measured in that session; anything marked **unverified** was not. Read the "Working rules" section first.

## Latest status (2026-09-28, full sweep `Results/Results_2026-09-25`)
Sections further down were written on 2026-09-23 and are partly superseded; this section wins where they disagree.
- **Comparison report:** `Results/COMPARISON_Results_2026-09-25_vs_Reference.md` (all categories, 104 problem/method pairs; regenerate with `python compare_results_to_reference.py Results_2026-09-25`). Of the 89 pairs that succeeded in both sweeps, 82 match (within 1 %) or beat the Reference. OCM, Pareto and LevelSet match on all structural and thermal problems (Pareto `Multiload` +1.1 %).
- **Still worse than the Reference or not converged (MMA):** `LBracketTopLoad_Vol_Stress` +6.0 %, `TwoBar` +2.2 %, `CantileverMidLoadVolumeCompliance` +1.0 %. Four stress/mass problems hit the 150-iteration cap (`LBracketMidLoad_Vol_Stress`, `LBracketTopLoad_Mass_StressFF`, `LBracketTopLoad_Stress_Vol`, `LBracketTopLoad_Vol_Stress`); three of them still end with a better objective than the Reference. MMA needs more iterations than the Reference in 23 of 28 cases.
- **Thermo-structural is supported now** (Plan B, `THERMOELASTIC_PLAN_B_LOG.md`): `ThermoStructuralFEA` couples the thermal and structural solves and autograd differentiates through both, so MMA and OC run them (Pareto and LevelSet still skip them). The problems were redefined to match the papers and a thermal-force dof-ordering bug was fixed, so the report marks them "not comparable" with the Reference. Validation against the papers is in `Results/Publications/PyTO_comparison/` (rebuild with `python -m pyto.examples_benchmarks.make_publication_comparison Results/Results_<date>/ThermoStructural/DENSITYMMA`).
- **OC on thermo-structural problems is not usable:** it 2-cycles because the thermally loaded compliance is not monotone in the design. In the sweep it hit the iteration cap on `BiClamp` and `MBBBeam`, and it ended at vf 0.79 / 0.78 (limit 0.4) on `BiClampDT4` / `BiClampDT10`. It only behaves on the purely mechanical variants (`BiClampDT0`, `MBBBeamNoHeat`). Use MMA for these.
- **2026-09-28 follow-ups:**
  - *Stress problems are not missing from the sweep.* The four MMA stress/mass runs are in it with `success=True`, and they behave exactly as in `topology_check.ipynb` (that notebook run of `LBracketTopLoad_Stress_Vol` also used all 150 iterations: 6.60e8 vs 6.62e8 in the sweep). The report lists them because they use all 150 iterations, and three end with the binarized design over the stress limit (1.5 %, 8.3 %, 19 %); the report now shows that warning.
  - *Inverter spring back to 0.1* (the Reference value; `f4e163e` had changed it to 0.01). MMA now gives J = -0.616 vs Reference -0.594 (3.8 % better, same vf 0.3).
  - *Body-force plates:* Pareto now runs both plates (GravityPlate arch, J 4.1e-6 at vf 0.10; CentrifugalPlate J 0.00637, better than MMA's 0.00776). OC now runs CentrifugalPlate (J 0.00714, hits the iteration cap). OC collapses GravityPlate (tried SIMP and RAMP), and LevelSet fails on both plates even with the body-force term added to its shape derivative, so those stay skipped (comments in the runner).
  - *Cleanup:* removed the dead `StructuralTOExamples.BiClamp` (no builder; the real one is `ThermoStructuralTOExamples.BiClamp`), fixed the `StresssFailureFactor` typo, the wrong STL path of `LBracketThickTopLoad_Stress_Vol` in `getSTLPath_TOProblem`, and `Mitchell_1` in `gui/PyTO_demos.py`. The 11 problems that build but are not in the sweep are listed in `benchmarks_not_in_sweep` in the runner (all checked to build, including `EdgeCantileverConstraintMatrix`, whose single-dof constraint rows are now folded into Dirichlet conditions).
- **Open:** Plan A (GravityPlate self-weight) skipped by the user; B7 (transient heat conduction) skipped by the user; MBB steady-state compliance is 29-37 % above Ooms (topology matches; see the Plan B log B6). The FourCorners/BridgeThermal question below is resolved (`compliance_sign` in `autodiff/qoi/compliance.py`); all four methods now match the Reference on both.

## Working rules from the user
- **Do not `git commit` unless the user explicitly asks in that message. Never `git push`.** (Commit `f3c3fbf` was made unprompted in the previous session; it is local and unpushed.)
- Reuse the codebase's own plotters (`fe.plot_mesh`, `plot_pseudo_density`, `plot_deformation`, `plot_vonMisesStress`); do not write new topology plots.
- The user wants problems fixed **one at a time**, checked against `Results/Results_Reference`, using `topology_check.ipynb`.
- Do not add new autograd / non-autograd options; the method chosen decides the gradient type (see below).

## Environment
```bash
cd /home/akshayk/Documents/GitHub/PyTO
PYTHONPATH=src MPLBACKEND=Agg PYVISTA_OFF_SCREEN=true /home/akshayk/anaconda3/envs/PyToLib/bin/python <script>
```
- Use the `PyToLib` conda interpreter. The system `python3` has no torch/nbformat.
- Test suite baseline: `python -m pytest tests/ -q` gives **84 passed, 1 xfailed**.
- Machine: 78 GB RAM, 20 cores. A process that runs out of memory can take the whole VSCode session down (the OOM killer works per cgroup), so run heavy jobs with a memory watchdog or in subprocesses.
- `nbconvert` is not installed; notebooks were validated by extracting their code cells and running them as scripts. They have **not** been run inside Jupyter itself.

## How each method gets its gradient (verified in code)
| Method | Driver | Gradient source |
|---|---|---|
| MMA (`DENSITYMMA`) | `topopt/drivers/mma.py`, `topopt_mma` | `torch.autograd.grad` (objective and per-constraint) |
| OCM (`DENSITYOCM`) | `topopt/drivers/oc.py` | `torch.autograd.grad` (objective); volume gradient from `reference_adjoint` |
| Pareto | `topopt/drivers/pareto.py` | Hand-derived topological sensitivity, `topopt/topological_sensitivity.py`. **No autograd.** |
| LevelSet | `topopt/drivers/levelset.py` | Hand-derived shape derivative, `compute_compliance_and_sensitivity`. **No autograd.** |

- The one hand-derived piece inside the autograd path is the backward of `SparseLinearSolve` in `autodiff/sparse_solve.py`: one adjoint solve `A^T y = grad_x`, then `-y_i x_j` on the stored nonzeros. Everything else is autograd.
- `autodiff/reference_adjoint/legacy_sensitivities.py` (old NumPy `-dK*ce` sensitivities) is meant as a test oracle, but **`oc.py` still imports and calls `compute_volume_constraint_and_gradient` from it** (lines 247 and 263); no other driver uses it. `topological_sensitivity.py` is *not* an oracle; Pareto needs it.
- `physics/thermoelastic/topopt_thermostructural_sensitivity.py` is a hand-derived adjoint used only by the GUI and a check script, not by any driver.
- An earlier statement in that session that "all four methods use autograd" was wrong and was corrected. A grep for `autograd.grad` matches comments in pareto.py and levelset.py; count real calls, not matches.

## Where autograd WORKS (verified)
Method: for each problem, build the FE model in an isolated subprocess, evaluate objective and every constraint through the drivers' own `compute_objective_and_gradient` / `compute_constraint_and_gradient`, take autograd gradients, and compare with central finite differences (eps 1e-4) on each quantity's largest-gradient elements plus one random element. Uniform random `x` in [0.3, 0.7], SIMP, no density filter (the filter chain is checked separately by optimization behaviour, not by this table).

- **37 of 43 problems pass**, worst relative error about 2e-5 after the fixes below.
- Supported and verified quantities: COMPLIANCE, VOLUME_FRACTION, MASS, PNORM_STRESS, MAX_VONMISES_STRESS, STRESS_FAILURE_FACTOR, GVECTOR. Structural and thermal (HeatPlate, FourCornersThermal, BridgeThermal).
- Peak memory for a gradient evaluation is at most 3 GB, including GEGrabCAD at 116,172 DOF.

| Problem | Physics | DOF | Objective | Constraints | Worst FD rel. error | Peak RSS (GB) |
|---|---|---|---|---|---|---|
| Bridge | structural | 30906 | COMPLIANCE | VOLUME_FRACTION | 1.8e-05 | 1.2 |
| BridgeThermal | thermal | 10423 | COMPLIANCE | VOLUME_FRACTION | 1.5e-07 | 0.7 |
| CantileverMidLoad | structural | 27744 | COMPLIANCE | VOLUME_FRACTION | 1.9e-06 | 1.3 |
| CantileverMidLoadVolumeCompliance | structural | 27744 | VOLUME_FRACTION | COMPLIANCE | 3.6e-06 | 1.4 |
| CantileverTipLoad | structural | 27744 | COMPLIANCE | VOLUME_FRACTION | 4.2e-06 | 1.3 |
| DistributedLoad | structural | 69336 | COMPLIANCE | VOLUME_FRACTION | 3.9e-05 | 1.8 |
| EdgeCantilever | structural | 84960 | COMPLIANCE | VOLUME_FRACTION | 1.5e-05 | 2.4 |
| EdgeCantileverLargeDOF | structural | 58968 | COMPLIANCE | VOLUME_FRACTION | 9.0e-06 | 2.0 |
| FourCornersThermal | thermal | 9245 | COMPLIANCE | VOLUME_FRACTION | 4.1e-08 | 0.7 |
| GEGrabCAD | structural | 116172 | COMPLIANCE | VOLUME_FRACTION | 2.2e-06 | 2.8 |
| HeatPlate | thermal | 9245 | COMPLIANCE | VOLUME_FRACTION | 2.5e-04 | 0.7 |
| Inverter | structural | 74529 | GVECTOR | VOLUME_FRACTION, COMPLIANCE | 9.3e-06 | 2.0 |
| KnuckleAssembly | structural | 74646 | COMPLIANCE | VOLUME_FRACTION | 3.7e-04 | 2.3 |
| LBracketMidLoad | structural | 38961 | COMPLIANCE | VOLUME_FRACTION | 1.1e-04 | 1.6 |
| LBracketMidLoad_Vol_Stress | structural | 43173 | VOLUME_FRACTION | MAX_VONMISES_STRESS | 3.8e-06 | 1.6 |
| LBracketMidLoad_Vol_StresssFailureFactor_Compliance | structural | 43173 | VOLUME_FRACTION | STRESS_FAILURE_FACTOR, COMPLIANCE | 4.7e-06 | 1.7 |
| LBracketThickMidLoad | structural | 77625 | COMPLIANCE | VOLUME_FRACTION | 8.4e-06 | 2.2 |
| LBracketThickTopLoad | structural | 77625 | COMPLIANCE | VOLUME_FRACTION | 2.2e-04 | 2.3 |
| LBracketThickTopLoad_Stress_Vol | structural | 53196 | PNORM_STRESS | VOLUME_FRACTION | 1.0e-06 | 2.1 |
| LBracketThickTopLoad_Vol_Stress | structural | 53196 | VOLUME_FRACTION | MAX_VONMISES_STRESS | 5.9e-06 | 2.2 |
| LBracketTopLoad | structural | 38961 | COMPLIANCE | VOLUME_FRACTION | 1.2e-05 | 1.6 |
| LBracketTopLoad_Mass_StressFF | structural | 60480 | MASS | STRESS_FAILURE_FACTOR | 1.7e-07 | 2.3 |
| LBracketTopLoad_Stress_Vol | structural | 38961 | PNORM_STRESS | VOLUME_FRACTION | 4.2e-05 | 1.6 |
| LBracketTopLoad_Vol_Stress | structural | 60480 | VOLUME_FRACTION | MAX_VONMISES_STRESS | 1.7e-07 | 2.1 |
| MBBBeam | structural | 59220 | COMPLIANCE | VOLUME_FRACTION | 1.9e-05 | 2.3 |
| Mitchell | structural | 27735 | COMPLIANCE | VOLUME_FRACTION | 3.9e-07 | 1.5 |
| Mitchell_2 | structural | 27735 | COMPLIANCE | VOLUME_FRACTION | 1.4e-06 | 1.5 |
| Mitchell_3 | structural | 27735 | COMPLIANCE | VOLUME_FRACTION | 6.0e-07 | 1.4 |
| Multiload | structural | 84960 | COMPLIANCE | VOLUME_FRACTION | 4.7e-06 | 2.4 |
| ShortCantileverMidLoad | structural | 27735 | COMPLIANCE | VOLUME_FRACTION | 1.5e-06 | 1.4 |
| ShortCantileverTipLoad | structural | 27735 | COMPLIANCE | VOLUME_FRACTION | 2.0e-07 | 1.4 |
| Table | structural | 76176 | COMPLIANCE | VOLUME_FRACTION | 2.6e-08 | 2.2 |
| TensilePlate | structural | 86400 | COMPLIANCE | VOLUME_FRACTION | 3.7e-04 | 1.9 |
| ThreeHoleBracket | structural | 90540 | COMPLIANCE | VOLUME_FRACTION | 5.4e-06 | 2.4 |
| ThreeHoleBracketThick | structural | 82251 | COMPLIANCE | VOLUME_FRACTION | 2.6e-07 | 2.3 |
| TorquePlate | structural | 66210 | COMPLIANCE | VOLUME_FRACTION | 2.8e-06 | 1.8 |
| TwoBar | structural | 27744 | COMPLIANCE | VOLUME_FRACTION | 1.2e-07 | 1.4 |

### Two measurement pitfalls (already handled in the harness below)
1. **Stress constraints mutate state.** The `MAX_VONMISES_STRESS` and `STRESS_FAILURE_FACTOR` branches of `compute_constraint_and_gradient` (`autodiff/qoi/__init__.py`) update `to_params.stress_scaling` as a side effect whenever `max_von_mises > limit`. The constraint is therefore not a pure function of `x`. Finite differences showed error exactly 1.0 until the scaling was reset before each evaluation; then the error fell to under 6e-6. The autograd gradient is correct. Whether the state mutation is acceptable inside MMA's line search is **unverified**.
2. **Round-off.** Testing random elements with tiny gradients gave spurious errors of 1e-3 to 4e-2 on Bridge and a compliance constraint; choosing each quantity's largest-gradient elements removed them.

## Where autograd does NOT work, or is limited
### Problems that fail to build (6)
| Problem | Root cause | Suggested fix |
|---|---|---|
| `GravityPlate`, `CentrifugalPlate` | `torch.tensor(<scipy coo_matrix>)` in `HexStructuralFEA.__init__` (`physics/structural/hex_structural_fea.py`, around line 100, `self._elem_to_node_map_torch`). `mesh.elem_to_node_field_mapping` is a sparse matrix. | Convert to a torch sparse tensor (a dense conversion is roughly num_nodes x num_elems, too large). Then check how `solve()` uses it (around line 179) and verify the body-force gradient. |
| `CantileverTipLoadDisplacementObjective` | `topopt_structural_benchmarks.py` (around line 223) uses `TO_QOI.RELATIVE_COMPLIANCE`, which does not exist in `TO_QOI`. | Update the benchmark definition (probably `COMPLIANCE` with a relative limit) or remove the entry. |
| `BiClamp` (thermal enum) | `topopt_thermal_benchmarks.py` refers to `HexThermalExamples.BiClamp`, which does not exist. | Add the builder or drop the enum member. |
| `LBracketTopLoad_Vol_StressFailureFactor` | In the enum and the STL-path list, but no branch in `getStructuralTOProblem`'s problem chain (raises "Unknown problem"). | Add the branch or remove the entry. |
| `EdgeCantileverConstraintMatrix` | All supports are in `bc.constraint_matrix`, `fixed_dofs=[]`. The torch solve ignores `constraint_matrix` and `apply_dirichlet_bc_torch` fails on an empty list ("expected np.ndarray (got list)"). Not in the runner's problem lists. | Needs a real constraint-matrix path in the torch solve, or exclude it. |

### Structural limits
- **Thermo-structural** (superseded 2026-09-28, see Latest status): now run through `ThermoStructuralFEA` with MMA and OC; Pareto and LevelSet skip them. `main.ipynb` may still describe them as unsupported.
- **OCM, Pareto, LevelSet** raise `ValueError` unless the objective is COMPLIANCE and `Constraints[0]` is VOLUME_FRACTION. Only MMA handles stress/mass/GVECTOR/multiple constraints, so the non-compliance benchmarks run under MMA only.
- **QOIs never implemented** in `autodiff/qoi`: VOLUME, GFUNCTION, COST, TEMPERATURE_FAILURE_FACTOR, MAX_CRITICALITY, MEAN_CRITICALITY, FATIGUE_FAILURE_FACTOR, PBR.
- **DPCG (large-DOF iterative solver) is not implemented in the torch path** (`autodiff/sparse_solve.py` warns and falls back to a direct solve). No benchmark triggers it because the cutoff `DIRECT_SOLVER_DOF_CUTOFF = 100000` uses `>` and the largest `nDOFDesired` is exactly 100000, but a larger user problem would.
- **Multi-material**: only the first material's element stiffness is used (`compute_element_stiffness` in `drivers/_shared.py`).
- `check_rigid_body_constraint` (`core/bc.py`) skips BCs that use `constraint_matrix`.

### Open decision (needs the user) -- RESOLVED, see Latest status
**FourCornersThermal and BridgeThermal** have prescribed temperatures and no heat source. The Reference (legacy code) effectively *maximizes* conduction: volume stays at its limit and J is about 6839 (FourCorners, vf 0.75). The current autograd differentiates `J = u^T K(x) u` correctly, and J falls as material is removed, so MMA strips the design (vf about 0.15, J about 1e-5) and OCM ends at vf 1.0. The gradient is verified correct (they pass the FD check); the question is the intended objective for prescribed-value problems. Matching the Reference means adopting the legacy sign convention (roughly, treating the equivalent load as fixed). The user has not decided.

## Fixes already made in the working tree (all verified, see git state below)
1. **MMA sensitivity filter conditions were inverted** (`topopt_mma`, `optimizationFunction`). Autograd gradients w.r.t. the raw variable already include the chain rule through the density filter, so the extra filter must apply only when `APPLY_FILTER_TO_DENSITY` is **off**. This was why only DENSITYMMA drifted on structural problems while DENSITYOCM matched exactly.
2. **`compute_compliance_torch` used the structural SIMP scaling (p=3) for thermal solves** (`autodiff/qoi/compliance.py`). It now selects the scaling by solver type.
3. **MMA finalization** evaluated the pre-binarization design and overwrote the mesh with the grey field. It now binarizes the *filtered* density (volume-preserving), evaluates that design with the filter off, appends it to `history`, and leaves it on the mesh. Note: the last entry of `history["objective"]` is therefore the binarized design, not an iteration; the second-to-last is the last continuous iterate.
4. **Dense backward in `apply_dirichlet_bc_torch`** (`core/bc.py`): `torch.sparse.mm(K, u_pres)` produced a dense ndof x ndof gradient (about 48 GB at 77k DOF) and OOM-killed large problems. Replaced with an `index_add` over the stored nonzeros; checked identical (RHS diff 2e-16, gradient diff 0). Found by hooking every autograd node and printing RSS; a synthetic `sparse.mm` test did *not* reproduce it, so do not trust isolated micro-tests here.
5. **LBracketThick supports/loads** (`examples_benchmarks/hex_structural_examples.py`, `createLBracketThickProblem`): the STL was re-tessellated (32 to 596 triangles) but hard-coded triangle indices `[16,17]`, `[12,13]`, `[4,5]` stayed. They now select by geometry and feed the existing `get_nodes_on_triangles`. Verified node-for-node identical to the Reference (fixed 360, top-load 17, mid-load 30). This also explains the earlier `LBracketThickMidLoad` "under-constrained" failure. **Other problems that select faces by index may have the same latent problem; unaudited.**
6. **`plot_deformation`** (`physics/structural/hex_structural_fea.py`) crashed on torch tensors and overwrote the caller's `save_path` with `None`. Both fixed.
7. Earlier commits: rigid-body BC check (`a92c8e3`), external-spring crash on Inverter (`4d17767`), benchmark-runner repairs (`f112827`, `518491b`, `70f7c87`).

## Post-fix verification against Results_Reference (DENSITYMMA unless noted)
| Problem | Reference J | This code | Notes |
|---|---|---|---|
| HeatPlate | 0.5095 | 0.509 | OCM: 0.5085 vs 0.5086 |
| Mitchell | 2.604 | 2.525 | |
| TorquePlate | 0.7919 | 0.7273 | Matches Reference OCM (0.7271); MMA reference probably stopped early. Continuous plateau is about 0.85; 0.7273 is the binarized design. |
| ThreeHoleBracket | 0.5814 | 0.5721 | Continuous last iterate 0.5953 |
| LBracketThickTopLoad | 0.4145 | 0.4113 | Was 0.0097 before fix 5 |
Not yet re-run after the fixes: CantileverMidLoad, CantileverTipLoad, MBBBeam (2.5D, target J about 99.3), Table, EdgeCantilever, Multiload and the non-compliance/stress problems. Reference targets for these are in `Results_Reference`.

## About the result folders
- `Results/Results_Reference` was generated about 2026-02-21, **before** commit `f4e163e` (2026-03-19) made density filtering the default. PARETO and LEVELSET reproduce it exactly on current code, so it is a trustworthy target.
- `Results/Results_2026-08-31` is the last full sweep and **predates all the fixes above**. Its numbers for MMA, thermal and LBracketThick are known-wrong. Do not use it as a target.
- **Name collision:** `MBBBeam` exists both as a 2.5D structural problem (J about 99) and as a ThermoStructural problem (J 5.5e8, vf 0.40). Any lookup by name must exclude the `ThermoStructural` folders (`topology_check.ipynb` does).
- Sweep outcome counts (rows succeeded): Reference MMA 30/30, OCM 20/20, Pareto 20/20, LevelSet 19/20; Aug-31 MMA 23/28, OCM 15/22, Pareto 20/23, LevelSet 18/23.
- Known Aug-31 failures not yet re-checked after the fixes: thermal OCM (should be fixed by fix 2), `LBracketThickTopLoad`/`Table` under LevelSet ("Volume fraction not reached"), `GEGrabCAD` (memory; should be fixed by fix 4), `ThreeHoleBracket` MMA/OCM (memory; verified fixed for MMA).

## Notebooks
- `topology_check.ipynb`: runs one problem/method like the benchmark runner, then shows the setup plot, the result table (last continuous iterate vs final binarized design, Reference, 2026-08-31), topology, pseudo-density, deformation, von Mises, and a convergence curve, with the Reference images alongside. Edit `PROBLEM`/`METHOD` in the first code cell. Current setting: CantileverMidLoad, DENSITYMMA (a commented list of alternatives sits below it).
- `main.ipynb`: method/category/problem selector using the `run_topopt_*` wrappers; prints each method's gradient source; final section explains the full benchmark.

## Running the full benchmark
```bash
cd /home/akshayk/Documents/GitHub/PyTO
PYTHONPATH=src MPLBACKEND=Agg PYVISTA_OFF_SCREEN=true nohup /home/akshayk/anaconda3/envs/PyToLib/bin/python -m pyto.examples_benchmarks.topopt_run_benchmarks > benchmark_run.log 2>&1 &
```
- Entry point: `run_all_benchmarks_isolated(optimizationMethods=None, problems=None)` in `examples_benchmarks/topopt_run_benchmarks.py`. One spawned subprocess per (method, problem); the output date is fixed once; `combine_results()` and `create_summary_tables()` run at the end.
- Output: `./Results/Results_<YYYY-MM-DD>/`, so run from the repo root. Expect many hours.
- A `subprocess ... exited with code -9` line means an OS out-of-memory kill; that pair is missing from the CSV. Other failures are recorded with `success=False` and `errorMsg`.
- Subset example: `run_all_benchmarks_isolated(optimizationMethods=[TO_METHODS.DENSITYMMA], problems=[StructuralTOExamples.Mitchell])`.
- Do not launch it until the known build failures above are resolved or accepted; otherwise it only repeats them.

## Suggested next steps (in dependency order)
1. Resolve the FourCornersThermal / BridgeThermal objective question with the user.
2. Fix the body-force constructor crash (GravityPlate, CentrifugalPlate) and verify the body-force gradient (the MMA driver also adds an analytic body-force term to the autograd gradient in `optimizationFunction`; check it is not double counting, since `solve()` already builds the body force from `x`).
3. Clean up the broken enum entries (RELATIVE_COMPLIANCE, thermal BiClamp, missing StressFailureFactor branch, ConstraintMatrix).
4. Audit other problems that select faces or nodes by hard-coded STL triangle index (`get_nodes_on_triangles`, `triList*`) for the same stale-index failure as LBracketThick.
5. Continue one-at-a-time verification with `topology_check.ipynb`: CantileverMidLoad, CantileverTipLoad, MBBBeam, then Table, EdgeCantilever, Multiload, then the stress problems.
6. Decide whether `stress_scaling` should stay stateful inside the constraint evaluation.
7. Then run the full sweep and compare with `Results_Reference`.

## Git state at time of writing
Branch `AutoPyTO`. Local commits (unpushed): `f3c3fbf`, `4d17767`, `a92c8e3`, `70f7c87`, `518491b`, `f112827`. **Uncommitted** working-tree changes: `main.ipynb`, `topology_check.ipynb`, `src/pyto/core/bc.py` (fix 4), `src/pyto/examples_benchmarks/hex_structural_examples.py` (fix 5), `src/pyto/physics/structural/hex_structural_fea.py` (fix 6).

## Appendix: the finite-difference harness
This lived in the session scratchpad (not in the repo). Run as `S=<outdir> python grad_matrix.py [ProblemName ...]`; it writes one JSON per problem under `$S/gradres/`. It spawns a subprocess per problem with a 16 GB memory watchdog and a 900 s timeout.
```python
"""Per-problem autograd check: build -> forward solve -> objective/constraint gradients -> finite-difference comparison."""
import sys, os, json, time, multiprocessing as mp

def worker(name, out_path):
    import threading, psutil
    proc = psutil.Process()
    def wd():
        while True:
            if proc.memory_info().rss > 16e9: os._exit(9)
            time.sleep(0.5)
    threading.Thread(target=wd, daemon=True).start()
    sys.path.insert(0, "src")
    import matplotlib; matplotlib.use("Agg")
    import numpy as np, torch
    res = {"problem": name, "stage": "import"}
    def dump():
        json.dump(res, open(out_path, "w"))
    try:
        from pyto.examples_benchmarks.topopt_structural_benchmarks import StructuralTOExamples, getStructuralTOProblem
        from pyto.examples_benchmarks.topopt_thermal_benchmarks import ThermalTOExamples, getThermalTOProblem
        from pyto.physics.structural.hex_structural_fea import HexStructuralFEA
        from pyto.physics.thermal.hex_thermal_fea import HexThermalFEA
        import pyto.solve.numpy_backend as lin_solv
        from pyto.autodiff.qoi import compute_objective_and_gradient, compute_constraint_and_gradient
        from pyto.topopt.drivers._shared import compute_element_stiffness
        res["stage"] = "build"; dump()
        if name in StructuralTOExamples.__members__:
            p = StructuralTOExamples[name]; res["physics"] = "structural"
            mesh, mat, bc, ebf, tp = getStructuralTOProblem(p)
            fe = HexStructuralFEA(mesh=mesh, mat_prop=mat, bc=bc, solver=lin_solv.Solvers.PARDISO, rtol=1e-8, elem_body_force=ebf)
        else:
            p = ThermalTOExamples[name]; res["physics"] = "thermal"
            mesh, mat, bc, ebf, tp = getThermalTOProblem(p)
            fe = HexThermalFEA(mesh=mesh, mat_prop=mat, bc=bc, solver=lin_solv.Solvers.PARDISO, rtol=1e-8, elem_body_force=ebf)
        res.update(ndof=int(bc.num_dofs), n_elem=int(mesh.num_elems), objective=tp.Objective[0].name,
                   constraints=[c[0].name for c in tp.Constraints])
        res["stage"] = "forward"; dump()
        KE = torch.tensor(compute_element_stiffness(fe)); from pyto.autodiff.material_model import MaterialModel; mm = MaterialModel.SIMP  # same as the MMA driver
        rng = np.random.default_rng(0)
        x0 = 0.3 + 0.4 * rng.random(mesh.num_elems)
        s0 = tp.stress_scaling
        def evaluate(x_t):
            tp.stress_scaling = s0   # constraint eval mutates this as a side effect; keep the function pure
            sol = fe.solve(x_t, mm)
            obj = compute_objective_and_gradient(tp, sol, x_t, fe, KE, mm)
            c, _ = compute_constraint_and_gradient(tp, sol, x_t, fe, KE, mm)
            return obj, c.reshape(-1)
        x_t = torch.tensor(x0, requires_grad=True)
        t0 = time.time(); obj, c = evaluate(x_t); res["forward_s"] = round(time.time() - t0, 1)
        res["objective_value"] = float(obj); res["forward_finite"] = bool(torch.isfinite(obj).item())
        res["stage"] = "gradient"; dump()
        gs = {}
        (gs["objective"],) = torch.autograd.grad(obj, x_t, retain_graph=True)
        for i in range(c.numel()):
            (gs[f"constraint_{i+1}:{tp.Constraints[i][0].name}"],) = torch.autograd.grad(c[i], x_t, retain_graph=True)
        res["grad_finite"] = {k: bool(torch.isfinite(v).all().item()) for k, v in gs.items()}
        res["grad_nonzero"] = {k: bool((v.abs() > 0).any().item()) for k, v in gs.items()}
        res["stage"] = "finite_difference"; dump()
        rng2 = np.random.default_rng(1)
        per_key_idx = {}
        for k, v in gs.items():
            g = v.numpy(); order = np.argsort(-np.abs(g))
            per_key_idx[k] = sorted(set(order[:2].tolist() + [int(rng2.choice(np.where(np.abs(g) > 1e-6 * np.abs(g).max())[0]))]))
        all_idx = sorted(set(i for v in per_key_idx.values() for i in v))
        eps = 1e-4; fvals = {}
        for e in all_idx:
            fv = {}
            for sgn in (+1, -1):
                xp = x0.copy(); xp[e] += sgn * eps
                with torch.no_grad():
                    o, cc = evaluate(torch.tensor(xp))
                fv[sgn] = (float(o), cc.numpy().copy())
            fvals[e] = fv
        errs = {k: [] for k in gs}
        for k in gs:
            for e in per_key_idx[k]:
                fv = fvals[e]
                if k == "objective": fd = (fv[1][0] - fv[-1][0]) / (2 * eps)
                else:
                    j = int(k.split("_")[1].split(":")[0]) - 1
                    fd = (fv[1][1][j] - fv[-1][1][j]) / (2 * eps)
                ag = float(gs[k][e]); errs[k].append(abs(fd - ag) / max(abs(ag), abs(fd), 1e-12))
        res["fd_rel_err"] = {k: float(np.max(v)) for k, v in errs.items()}
        res["stage"] = "done"; res["peak_rss_gb"] = round(proc.memory_info().rss / 1e9, 1)
    except Exception as e:
        import traceback
        res["error"] = f"{type(e).__name__}: {str(e)[:200]}"; res["traceback_tail"] = traceback.format_exc().splitlines()[-4:]
    dump()

if __name__ == "__main__":
    sys.path.insert(0, "src")
    from pyto.examples_benchmarks.topopt_structural_benchmarks import StructuralTOExamples
    from pyto.examples_benchmarks.topopt_thermal_benchmarks import ThermalTOExamples
    names = [m.name for m in StructuralTOExamples] + [m.name for m in ThermalTOExamples]
    only = sys.argv[1:]
    if only: names = [n for n in names if n in only]
    outdir = os.environ["S"] + "/gradres"; os.makedirs(outdir, exist_ok=True)
    ctx = mp.get_context("spawn")
    for n in names:
        path = f"{outdir}/{n}.json"
        if os.path.exists(path): os.remove(path)
        t0 = time.time(); p = ctx.Process(target=worker, args=(n, path)); p.start(); p.join(900)
        if p.is_alive(): p.kill(); p.join(); status = "TIMEOUT"
        else: status = f"exit={p.exitcode}"
        r = json.load(open(path)) if os.path.exists(path) else {"problem": n}
        r["proc_status"] = status; r["wall_s"] = round(time.time() - t0, 1); json.dump(r, open(path, "w"))
        print(f"{n:55s} {status:8s} stage={r.get('stage','?'):18s} {r.get('error','')[:80]}", flush=True)
```
