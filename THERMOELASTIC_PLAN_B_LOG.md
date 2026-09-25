# Thermo-mechanical coupling (BiClamp, MBB-thermo): plan B and change log

Part B of `SELFWEIGHT_AND_THERMOELASTIC_PLAN.md` (Part A, the self-weight modified SIMP, is skipped: PyTO keeps its current handling). Sources: Rodrigues & Fernandes 1995 (BiClamp = their Example A), Ooms et al. (MBB with transient heat conduction). Every step ends with a commit; the notes below say what changed and how it was checked.

Status: `[ ]` todo, `[~]` in progress, `[x]` done.

## Steps
- [x] **B1. Differentiable thermoelastic force** `F_th(x, T)` (torch), validated against the existing NumPy `get_thermoelastic_force` and by finite differences.
- [ ] **B2. Structural solve accepts the thermal load** (and an element thermal strain for stresses).
- [ ] **B3. Coupled solver object** (`ThermoStructuralFEA`) with the single-solver interface the drivers expect; gradient by autograd through both adjoints, checked by finite differences and against the hand-derived `ThermoElasticSensitivity`.
- [ ] **B4. Drivers and runner** run the coupled problems (MMA and OC).
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
