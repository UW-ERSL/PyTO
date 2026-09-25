# Thermo-mechanical coupling (BiClamp, MBB-thermo): plan B and change log

Part B of `SELFWEIGHT_AND_THERMOELASTIC_PLAN.md` (Part A, the self-weight modified SIMP, is skipped: PyTO keeps its current handling). Sources: Rodrigues & Fernandes 1995 (BiClamp = their Example A), Ooms et al. (MBB with transient heat conduction). Every step ends with a commit; the notes below say what changed and how it was checked.

Status: `[ ]` todo, `[~]` in progress, `[x]` done.

## Steps
- [x] **B1. Differentiable thermoelastic force** `F_th(x, T)` (torch), validated against the existing NumPy `get_thermoelastic_force` and by finite differences.
- [x] **B2. Structural solve accepts the thermal load** (and an element thermal strain for stresses).
- [x] **B3. Coupled solver object** (`ThermoStructuralFEA`) with the single-solver interface the drivers expect; gradient by autograd through both adjoints, checked by finite differences and against the hand-derived `ThermoElasticSensitivity`.
- [x] **B4. Drivers and runner** run the coupled problems (MMA and OC).
- [ ] **B0. Problem definitions** matching the papers (BiClamp: R&F Example A; MBB: Ooms parameters). Done after B3 because it needs the coupled solver to validate.
- [ ] **B5. BiClamp validation** against R&F and the Reference row.
- [ ] **B6. MBB steady-state validation** against Ooms and the Reference row.
- [ ] **B7. Transient thermal** (backward Euler, design-dependent C and K, objective summed over time steps).
- [ ] **B8. Heaviside projection** (optional).
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
- **Test:** `test_mma_and_oc_run_on_the_coupled_solver` (small coupled problem: finite objective, decreases, volume constraint respected).
- **Open for later steps:** compliance of a thermally driven problem is not monotone in the design, so OC's assumption (negative sensitivities) can fail on it; MMA is the reference method in R&F and Ooms. Watch OC on MBB in B6.
