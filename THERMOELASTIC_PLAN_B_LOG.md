# Thermo-mechanical coupling (BiClamp, MBB-thermo): plan B and change log

Part B of `SELFWEIGHT_AND_THERMOELASTIC_PLAN.md` (Part A, the self-weight modified SIMP, is skipped: PyTO keeps its current handling). Sources: Rodrigues & Fernandes 1995 (BiClamp = their Example A), Ooms et al. (MBB with transient heat conduction). Every step ends with a commit; the notes below say what changed and how it was checked.

Status: `[ ]` todo, `[~]` in progress, `[x]` done.

## Steps
- [x] **B1. Differentiable thermoelastic force** `F_th(x, T)` (torch), validated against the existing NumPy `get_thermoelastic_force` and by finite differences.
- [x] **B2. Structural solve accepts the thermal load** (and an element thermal strain for stresses).
- [x] **B3. Coupled solver object** (`ThermoStructuralFEA`) with the single-solver interface the drivers expect; gradient by autograd through both adjoints, checked by finite differences and against the hand-derived `ThermoElasticSensitivity`.
- [x] **B4. Drivers and runner** run the coupled problems (MMA and OC).
- [x] **B0. Problem definitions** matching the papers (BiClamp: R&F Example A; MBB: Ooms parameters). Done after B3 because it needs the coupled solver to validate.
- [x] **B5. BiClamp validation** against R&F and the Reference row.
- [x] **B6. MBB steady-state validation** against Ooms and the Reference row.
- [-] **B7. Transient thermal** (SKIPPED by user decision, 2026-09-25) (backward Euler, design-dependent C and K, objective summed over time steps).
- [x] **B8. Heaviside projection** (optional; done together with B6 because the MBB comparison needed it).
- [ ] **B9. Sweep and comparison report** include the two problems; update the handoff file.

## Change log
(Newest last.)

### B1 (done): differentiable thermoelastic force
- **Why:** `HexThermalFEA.get_thermoelastic_force` is plain NumPy loops, so the chain thermal solve -> force -> structural solve could not carry gradients.
- **Change** (`physics/thermal/hex_thermal_fea.py`):
  - `get_thermoelastic_force_torch(T, x, material_model)`: per element `f_e = s_beta(x_e) * E * alpha / (1 - 2 nu) * H0 (T_e - T_ref)`, vectorized and assembled with `index_add`. `getHMatrix` is proportional to `1/(2 nu - 1)`, so one reference matrix `H0 = getHMatrix(nu = 0)` serves every element and material (this also handles multi-material meshes without a Python loop). `s_beta` is the structural stiffness scaling, i.e. the thermal stress coefficient `beta = E*alpha` is penalized like `E` (Rodrigues & Fernandes; Ooms et al. use the same p = 3).
  - `get_element_thermal_strain(T)`: free thermal strain per element, `alpha (T_mean - T_ref) [1,1,1,0,0,0]`, for removing the thermal part when computing stresses (used in B2).
- **Checked** (`tests/test_thermoelastic.py`): the torch force equals the NumPy reference to `rtol 1e-10`; gradients w.r.t. both `T` and `x` match central finite differences.

### B2 (done): structural solve accepts the thermal load; fixed a dof-ordering bug in `getHMatrix`
- **Change** (`physics/structural/hex_structural_fea.py`): `solve(x, material_model, extra_force=None, elem_thermal_strain=None)`. `extra_force` (a torch vector, differentiable) is added to the mechanical load; the stored `thermo_elastic_force` is also honoured. `elem_thermal_strain` is kept and `postprocess()` subtracts it from the total strain, so stresses come from the elastic strain only (replaces a dangling `solElastic` branch).
- **Bug found** (`physics/thermal/hex_thermal_fea.py`): `getHMatrix` returned its 24 rows COMPONENT-major (all x of nodes 1..8, then y, then z), while `edofMat`, the element stiffness and the displacement vector are NODE-major (3*j + c). The thermal force was therefore scrambled: a freely expanding body did not expand freely. Fixed at the source by permuting the rows to node-major (`H.reshape(3,8,8).transpose(1,0,2).reshape(24,8)`). This also corrects `get_thermoelastic_force` (NumPy), `get_thermoelastic_force_torch` and `ThermoElasticSensitivity`, which all consume that matrix.
- **Consequence:** the old Reference/PyTO ThermoStructural numbers and the GUI thermo-elastic demo were produced with the wrong ordering and are not trustworthy; they need recomputing (B5/B6).
- **Checked** (`tests/test_thermoelastic.py`): free expansion with 6 minimal supports gives `u = alpha dT (X - X0)` exactly and ~zero von Mises; a bar clamped in x at both ends gives `sigma_xx = -E alpha dT`. Full suite: 94 passed, 1 xfailed.

### B3 (done): coupled solver object `ThermoStructuralFEA`
- **Approach:** the drivers take one `fe_solver` and call `solve(x, material_model)`. A new class `ThermoStructuralFEA(HexStructuralFEA)` (`physics/thermoelastic/thermostructural_fea.py`) owns a `HexThermalFEA` (`.thermal_fea`) and its `solve` runs `T = thermal.solve(x)` -> `F_th = get_thermoelastic_force_torch(T, x)` -> `HexStructuralFEA.solve(x, extra_force=F_th, elem_thermal_strain=...)`. It returns the displacement, so the existing qoi code (compliance energy form `u^T K u = F_total . u`, volume, stress) needs no change, and `isinstance(fe, HexStructuralFEA)` checks in the drivers stay true. `.temperature` keeps the last T.
- **No coupled sensitivity code:** everything is torch, so `autograd` goes through the thermal adjoint, the force and the structural adjoint. Nothing hand-derived enters the optimizers.
- **Pitfall handled:** `HexThermalFEA.__init__` overwrites the shared `mesh.edofMat` with the thermal connectivity. The thermal object is built first and the structural constructor last, which leaves `mesh.edofMat = edofMatStructural` (asserted in a test).
- **Checked** (`tests/test_thermoelastic.py`, hot face x=0 / cold face x=L, clamp + tip load, random densities):
  - coupled result equals the manual chain and `total_force = F_mech + F_th`;
  - autograd gradient of the compliance equals central finite differences (rtol 1e-5);
  - autograd gradient equals the independent hand-derived `ThermoElasticSensitivity` (3-term adjoint formula) to 5e-9 relative. That oracle uses the `getHMatrix` fixed in B2, so this also confirms the fix is consistent.
- **Note for B4:** the runner's THERMO_STRUCTURAL branch currently builds a plain `HexStructuralFEA` plus an unused thermal object; it must build `ThermoStructuralFEA` instead.

### B4 (done): drivers and runner run the coupled problems
- **Drivers:** no change needed. `ThermoStructuralFEA` is a `HexStructuralFEA`, so `topopt_mma` / `topopt_optimality_criteria` treat it as a structural problem (3 dof/node, structural filters, forces) and their autograd chain now goes through the thermal solve as well.
- **Runner** (`examples_benchmarks/topopt_run_benchmarks.py`): the THERMO_STRUCTURAL branch built a plain `HexStructuralFEA` and an unused thermal object; it now builds one `ThermoStructuralFEA`. The MMA `continue` for thermostructural problems (from the old two-solver design) is removed, and OC now runs them too. Pareto and LevelSet still skip them (hand-derived sensitivities for structural/thermal only).
- **Smoke runs** (default problem definitions, 33 750 dof, PARDISO, 15 iterations, drivers only, NOT yet a validation):
  - BiClamp: MMA J 461 -> 18.0, vf 0.250; OC J 461 -> 17.85, vf 0.250 (converged in 15 FEAs).
  - MBBBeam: MMA J 4.8e9 -> 9.3e8, vf 0.398; OC J 4.8e9 -> 5.8e8, vf 0.397 (OC hit the 15-iteration cap; the MBB parameters are still the old ones, B0 changes them).
  - Runner path (`runTOMethodOnBenchmarks`, OC, BiClamp) writes its CSV row and image as for other problems.
- **Finding (small hot/cold cantilever test problem, 30 iterations):** MMA oscillates for ~10 iterations (54 662, 14 594, 98 045, 14 054, 125 557, ...) and then converges to ~12 590. OC 2-cycles between 14 256 and 99 061 and never converges. Cause: the compliance of a thermally loaded design is not monotone in the density, so OC's assumption of negative sensitivities fails. MMA (used by R&F and Ooms) is the method to trust for coupled problems; OC results on BiClamp/MBB are informational only, and B5/B6 validate with MMA.
- **Test:** `test_mma_converges_and_oc_runs_on_the_coupled_solver` (MMA final objective below half the start and volume constraint respected; OC only has to run, because of the finding above). An earlier version of this test failed on the OC 2-cycle and was committed by mistake in c4be11d; fixed in the next commit.

### B0 (done): problem definitions matching the papers
- **BiClamp = Rodrigues & Fernandes (R&F) 1995, Example A** (`topopt_thermostructural_benchmarks.py`). The STL plate is already 72 x 48 cm. Changes: load 1000 kgf (9 806.65 N, was 1e5 N), volume 40 % (was 25 %), the clamped side columns are non-design (`KeepFixedElems`, R&F "black regions"). Four problems for the paper's four cases: `BiClampDT0`, `BiClamp` (dT = 1 K), `BiClampDT4`, `BiClampDT10`. The walls are held at `T_ref + dT` and conduction spreads it, so every connected design sits at the uniform `T_ref + dT` of the paper. Steel is kept: **R&F do not give E or alpha**, so only the trend with dT is comparable, not the compliance values.
- **MBBBeam = Ooms et al. 2023** (`hex_thermostructural_examples.createMBBBeamProblem`): concrete from their Table 1 (E = 30 GPa; note mat_lib's "Concrete" has E = 30 MPa and was not used), top edge 20 C, bottom edge 800 C stopping two nodes before the roller (as their code), stress-free temperature 20 C, load 5 kN on the half model per 10 mm of thickness (their code: `Fm = -10000/2`). The void is insulation: conductivity 0.03 of solid and conductivity SIMP exponent 3 (their `k0, k1`, `pk`). Filter radius 48 mm (their 0.04 x 240 elements x 5 mm). `MBBBeamNoHeat` is the same problem with the bottom at 20 C, i.e. their t_f = 0 case (Fig. 6a).
- **New per-problem settings:** `TOParams.ThermalReferenceTemperature`, `ConductivityPenalty`, `ConductivityVoidRatio` (None keeps the old defaults); `get_thermal_material_model_scaling_torch(x, model, penalty, void_ratio)`; `HexThermalFEA` / `ThermoStructuralFEA` take them as constructor arguments, the runner passes them.
- **Old definitions** (for the record): BiClamp had load 1e5 N and vf 0.25; MBBBeam had steel, Ta = Tf = 23 C (no heat at all) and load 1e8 N. The Reference rows for these problems are therefore not comparable to the papers or to the new numbers.

### B8 (done early): Heaviside projection in the MMA driver
- `TOParams.HeavisideProjection` (default False), `HeavisideBetaInterval` (50), `HeavisideBetaMax` (32): smooth projection `(tanh(b eta) + tanh(b (x~ - eta))) / (tanh(b eta) + tanh(b (1 - eta)))`, eta = 0.5, beta 1 -> 32 doubling every 50 iterations (Ooms Sect. 3.4). Inside the torch chain, so autograd includes it. With projection on, MMA's objective-change stop is disabled (the objective jumps at each beta step) and the final volume-preserving binarization is skipped (the design is already near 0/1).
- Used by MBBBeam and MBBBeamNoHeat only. Test: `test_mma_with_heaviside_projection_runs_and_is_differentiable`.

### B5 (done): BiClamp vs Rodrigues & Fernandes
MMA, 33 750 dof, `Results/Results_2026-09-25-thermo/ThermoStructural/DENSITYMMA`; side-by-side picture in `Results/Publications/PyTO_comparison/BiClamp_RodriguesFernandes.png`.

| dT | R&F volume used / 1319 | R&F c | PyTO vf (limit 0.4) | PyTO J [N m] | PyTO J / J(dT=0) | R&F c / c(dT=0) |
|---|---|---|---|---|---|---|
| 0 | 1.00 | 2.81 | 0.400 | 0.1089 | 1 | 1 |
| 1 K | 1.00 | 4.31 (4-node) / 5.24 (9-node) | 0.400 | 0.676 | 6.2 | 1.5-1.9 |
| 4 K | 0.81 | 12.08 | 0.279 | 6.58 | 60 | 4.3 |
| 10 K | 0.44 | 37.10 | 0.211 | 31.06 | 285 | 13.2 |

- **Qualitative match:** dT = 0 gives the V-shaped two-bar truss from the clamped corners to the load (R&F Fig. 4b). With dT > 0 the design detaches from the upper clamps and becomes a low arch hanging from the lower wall corners with a central block (R&F Fig. 4c/d, 6a), and above a certain dT the volume constraint stops being active (R&F: "the optimal structure will not use all the available material"). All three features are reproduced.
- **Quantitative:** the thermal effect is about 5x stronger in PyTO than in R&F (our dT = 0.2 K reproduces their dT = 1 K ratio 1.5: J 0.161 vs 0.109; our 0.8 K gives their 4 K ratio 4.4). That is the material: the ratio of thermal to mechanical load scales with `E alpha dT / F`, R&F do not publish E or alpha, and we use steel. Not a code discrepancy. Calibration runs (dT 0.2/0.8/1.2/1.6/2/3 K) in the scratchpad, not kept.
- **Old Reference row** (BiClamp J 3.87e8, vf 0.142) came from the old load/volume and the component-major `getHMatrix` bug (B2); not comparable.

### B6 (done): MBB steady state vs Ooms et al.
MMA with Heaviside, 350 iterations, 38 304 dof (~10.8 mm elements vs their 5 mm); side-by-side picture in `Results/Publications/PyTO_comparison/MBBBeam_Ooms.png`.
- **Units:** our model is 20 mm thick and carries 10 kN (5e5 N/m x 0.02 m); stiffness and both loads scale with thickness, so `c_Ooms(10 mm, 5 kN) = J_PyTO / 2`, and 1 N m = 1000 N mm.

| case | Ooms c [N mm] | PyTO J/2 [N mm] | diff | Ooms V | PyTO vf |
|---|---|---|---|---|---|
| t_f = 0 (no heating, Fig. 6a; `MBBBeamNoHeat`) | 19 166 | 19 010 | -0.8 % | 0.4000 | 0.400 |
| steady state (Fig. 6i) | 101 138 | 138 260 | +37 % | 0.3989 | 0.399 |

- **Mechanical case matches** (0.8 %, same two-triangle half-MBB truss as their Fig. 6a).
- **Steady state, topology matches:** solid pushed to the top, a row of 5 holes (cellular beam), an insulating void layer on the hot bottom edge that thickens towards the support, and a thin leg down to the roller (their Fig. 6i). The volume constraint is slightly inactive in both (0.399).
- **Steady state, compliance 37 % higher.** Causes, not yet separated: (1) mesh twice as coarse (10.8 mm vs 5 mm) with the same physical filter, so the thin insulation layer where the whole 780 K drop happens is under-resolved; (2) MMA ends in a period-2 oscillation at beta = 32 (J alternates between about 260 and 277 N m, i.e. 130 000 and 138 000 N mm) and the runner reports whichever iterate is last; a run with move limit 0.1 instead of 0.2 ended on 260.4 (130 200 N mm, +29 %); (3) Ooms stop on a 0.01 design change after 404 iterations, we run the fixed 350-iteration beta schedule. Next check if needed: 5 mm mesh (~290k dof) and the smaller move limit.
- **Old Reference row** (MBBBeam J 5.50e8) used steel, no temperature difference and a 1e8 N load; not comparable.
- **OC on these problems** was not validated (see B4: OC 2-cycles on thermally loaded compliance).
