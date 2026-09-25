# Plan: GravityPlate (self-weight) and BiClamp / MBB-thermo (thermo-mechanical)

Written 2026-09-25. Sources are the three papers in `Results/Publications/`. Status markers: `[ ]` todo, `[~]` in progress, `[x]` done. Nothing below is implemented yet; this is the analysis and the order of work. Working rules from the user still apply: no commits unless asked, never push, one problem at a time, reuse the codebase's own plotters.

## What the papers say

**[BD] Bruyneel & Duysinx (2005), "Note on topology optimization of continuum structures including self-weight", Struct Multidisc Optim 29.**
- Self-weight is a density-dependent load, so the compliance is **non-monotonic** in the design variables (p. 247, eq. 7: the `dg/dmu` term does not vanish, so the derivative can be positive, negative, or change sign).
- Pure self-weight is **ill-posed**: the trivial design mu = 0 is feasible and has zero weight. A **minimum volume** V_min > 0 (they use 1%) rejects it (p. 246, eq. 5). "Well-posed problems require a non-design-dependent load or a non-structural mass."
- **Parasitic low-density effect**: with standard SIMP the ratio weight / stiffness = mu / mu^p tends to infinity as mu tends to 0, so low-density regions get unbounded displacements and the optimizer produces erratic appendages (p. 251, Fig. 11). Fix: a **modified SIMP** that is linear below mu_C = 0.25 (eq. 16, Fig. 12): E = mu^p E0 for mu > mu_C, and E = mu * mu_C^(p-1) * E0 for mu <= mu_C. Their bounded ratio makes the problem stable.
- **Approximation scheme**: monotonic approximations (CONLIN, plain MMA) oscillate or need many iterations; non-monotonic ones (GCMMA, gradient-based GBMMA) converge in a fraction of the iterations (Table 1: 438 vs 253 vs 139 iterations for the arch problem at TOL 1e-4). Plain MMA does still reach the answer thanks to move limits, just slowly.
- **The volume constraint need not be active** (Fig. 9, p. 250): the unconstrained optimum for their arch stays at 32.5% even when 80% is allowed. When the allowed volume is *below* that optimum, the constraint is active and the arch gets thinner.
- Expected result for a gravity plate on two corner supports: an **arch** (Fig. 13, Fig. 17 bottom).

**[RF] Rodrigues & Fernandes (1995), "A material based model for topology optimization of thermoelastic structures", IJNME 38.**
- Compliance including thermal load: `min  integral b.u + integral beta(mu) e(u) dT + integral t.u`, with the thermal stress coefficient beta = E * alpha (p. 1953, eq. 1). The load depends on the design through beta, so, like self-weight, it is design-dependent (their eq. 16 has the extra term `2 dbeta/dmu e(u) dT`).
- **Example A (p. 1960): 72 x 47.7 x 1 cm plate, clamped on the left and right sides, 1000 kgf downward load, volume 40%, 60x30 mesh.** This is PyTO's BiClamp. Results: dT = 0 gives a V-shaped truss (compliance 2.81); dT = 1 K gives 4.31; larger dT (4 K, 10 K) makes the optimum use **less than the allowed volume** (final volume 1069, then 575, vs 1319 allowed), because extra material only adds thermal stress.
- The temperature field is given and independent of the design; the paper does not couple to a heat-conduction solve.

**[OO] Ooms, Vantyghem, Thienpont, Van Coile, De Corte, "Thermoelastic topology optimization of structural components at elevated temperatures considering transient heat conduction" (167-line MATLAB).**
- This is the MBB thermo example (the code base cites it). 2D, 240 x 80 elements of 5 mm (1200 x 400 mm, 10 mm thick), point load F_m = 10 kN (half model), thermal Dirichlet **T_a = 20 C on the top edge and T_f = 800 C on the bottom edge** (bottom condition stops 2 nodes before the roller support), t_f = 3600 s, **4 time steps** suggested (Sect. 4.1.2), volume 40%, x0 = 0.4 (p. 14).
- Materials (Table 1): concrete E = 30 GPa, nu = 0.3, alpha = 12e-6, kappa = 1 W/mK, cp = 900, rho = 2400; void mimics insulation: E = 30e-6 MPa (solid 30e3 MPa), kappa = 0.03 W/mK, cp = 700, rho = 20 kg/m3.
- Interpolation (eq. 13-16): SIMP p_E = p_kappa = p_beta = 3 for E, conductivity and thermal stress coefficient beta = E*alpha; the volumetric heat capacity c_v and density are interpolated **linearly**.
- Objective (eq. 12): **sum over time steps t = 0..t_N of the step compliance c(t) = F(t)^T U(t)**, F(t) = F_m + F_th(t), F_th^e = integral B^T D alpha dT_e (eq. 10).
- Transient thermal: implicit backward Euler `(C/dt + K_th) T(t) = C T(t-1)/dt + Q` (eq. 5). Sensitivities by the adjoint method, discretize-then-differentiate (eq. 20-29). 3 x t_N linear solves per iteration.
- Density filter plus **Heaviside projection** (beta doubled every 50 iterations from 1 to 32, eq. 32-33), MMA, tolerance 0.01 on the max design change, 1000 iterations max.
- Reference numbers to compare against (Table 2, Table 3): t_f = 3600 s, 4 steps: c = 128,326 N mm, V = 0.4000, 783 iterations. Steady state (t_f = infinity, Fig. 6i): c = 101,138 N mm, V = 0.3989, 404 iterations. t_f = 0: c = 19,166 N mm, 374 iterations (a plain mechanical MBB beam).

## Part A: GravityPlate

### What GravityPlate is
1.0 x 0.5 x 0.01 m steel plate (`createGravityPlateProblem`), supported at the two bottom corners (4 nodes), gravity in -y as an element body force, half domain by X symmetry, volume <= 0.10. It is the [BD] arch problem (pure self-weight).

### What is missing (gap analysis)
| Paper requirement | PyTO today | Consequence |
|---|---|---|
| Modified SIMP, linear below mu_C = 0.25 (BD eq. 16) | Standard SIMP `Evoid + x^p (1-Evoid)` with p = 3, everywhere | Weight/stiffness ratio explodes at low density: the parasitic effect that lets MMA "cheat" toward an almost empty design. |
| Minimum volume V_min > 0 to reject mu = 0 (BD eq. 5) | Only an upper bound (0.10); I added a 0.095 lower bound as a stop-gap | Without V_min the empty design has the lowest compliance (J 6.8e-5 vs 2.4e-3 uniform). The 0.095 band works but is a workaround, not the paper's formulation. |
| Non-monotonic approximation (GCMMA / GBMMA) | `mmaWrapper.runMMA` uses plain `mmasub` (monotonic) | Slower and oscillation-prone on non-monotonic compliance (BD Table 1). Not required to converge, but relevant. |
| Material model chosen per problem | `mma.py` hard-codes `MaterialModel.SIMP`; only `oc.py` reads `to_params.materialModel` | A problem cannot select the modified law today. |
| Gradient of the design-dependent load | Correct since step 2 (autograd through `solve()`; FD verified 8.7e-7) | Nothing missing here. |

Diagnosis: the gradient is right. The optimizer collapses because the interpolation is wrong for a design-dependent load (parasitic low-density effect) and because V_min is missing, exactly the two difficulties BD name.

### Steps (no hacks: each is a formulation from the paper)
- [ ] **A1. Modified SIMP (BD eq. 16).** Add a material model (for example `MaterialModel.SIMP_LINEAR_LOW`) with mu_C = 0.25 to `autodiff/material_model.py` (torch scaling; NumPy twin and sensitivity for Pareto/LevelSet consistency). Continuity at mu_C and the derivative are unit-tested. Done when: tests pass and a finite-difference check on GravityPlate agrees.
- [ ] **A2. Let the problem choose its material model.** `mma.py` must read `to_params.materialModel` (as `oc.py` already does). GravityPlate (and any body-force problem) sets the modified law.
- [ ] **A3. Replace the 0.095 band by the paper's V_min.** Keep `VOLUME_FRACTION_MIN`, but set it to 1% (BD), not 0.095. Done when: MMA gives an arch with the volume settling where the physics puts it (0.10 if the constraint is active, as in BD Fig. 9).
- [ ] **A4. Check the result against the paper and the Reference.** Qualitative: arch (BD Fig. 13/17). Quantitative: Pareto on the same problem (J about 3.8e-6 at vf 0.10 in the sweep) and the Reference MMA (J 6.85e-3, vf 0.10, which looked like a poor optimum). If MMA still oscillates or stalls, go to A5; otherwise skip it.
- [ ] **A5. Only if A4 fails: non-monotonic approximation.** Use GCMMA (`mmapy.gcmmasub` with the inner conservativeness loop, which needs one extra objective evaluation per inner iteration) as an option in `mmaWrapper`, selected per problem. GBMMA is a further option from the paper if GCMMA is too conservative.
- [ ] **A6. Tests and notebook.** Unit tests for A1; a slow-marked regression that GravityPlate does not collapse; `topology_check.ipynb` set to GravityPlate for you to inspect.

## Part B: BiClamp and MBB thermo-mechanical

### What exists
- `HexThermalFEA.solve` is torch-native and differentiable (same sparse-solve adjoint as the structural solve).
- `HexThermalFEA.get_thermoelastic_force(x, material_model)` builds the nodal thermal force, but in **plain NumPy loops**, so it is not differentiable. It already uses the stiffness scaling for beta = E*alpha and a reference temperature (23 C).
- `HexThermalFEA.getHMatrix` (B^T D alpha operator per element).
- `physics/thermoelastic/topopt_thermostructural_sensitivity.py`: `ThermoElasticSensitivity`, a **hand-derived steady-state adjoint** with a finite-difference verifier (used only by the GUI). This becomes the test oracle.
- `HexTransientThermalFEA` (Newmark), NumPy only, no design dependence, not differentiable.
- Benchmark definitions in `topopt_thermostructural_benchmarks.py` / `hex_thermostructural_examples.py`; the Reference (Feb 2026, older coupled MMA) has results: BiClamp J = 3.87e8, vf = 0.142 (the 0.25 limit was NOT active), 50 FEAs; MBBBeam J = 5.50e8, vf = 0.400, 47 FEAs.

### What is missing (gap analysis)
| Need | PyTO today | Consequence |
|---|---|---|
| Structural solve includes the thermal force | `HexStructuralFEA.solve()` does **not** add `thermo_elastic_force` (the argument is stored and used only in stress post-processing) | Even if a thermal solve ran, the structure would see no thermal load. |
| Differentiable thermal force `F_th(x, T)` | NumPy loops, no gradient | Autograd chain thermal -> force -> structural is broken. |
| A single object the drivers can take | MMA/OC take one `fe_solver`; the old two-solver MMA was dropped in the torch migration; the runner skips these problems | Nothing can run them. |
| Coupled objective | `c = F^T u` with F = F_mech + F_th; the compliance QOI must use structural scaling for this object | Needs a wrapper that presents as structural. |
| Problem definitions matching the papers | BiClamp: steel, walls at 24 C vs 23 C reference (dT = 1 K), 3D STL, vf 0.25, load 1e5. MBBBeam: steel, `Ta = Tf = 23` in the benchmark (**no thermal gradient at all**), load 1e8, vf 0.4, steady state | MBB is currently a purely mechanical beam; the paper's 20/800 C concrete/insulation problem is not set up. |
| Transient conduction with a time-summed objective (OO eq. 12) | Not differentiable, no design-dependent `C`, `K` | Needed only for the transient MBB comparison. |
| Heaviside projection (OO eq. 33) | Density filter only | Needed for OO-like sharp results; optional first. |
| Interpolation (SIMP 3-3-3 for E, kappa, beta; linear c_v, rho; void = insulation) | Thermal void conductivity is 1e-8 relative; beta reuses the structural stiffness scaling | Close for the steady case; the insulating void (kappa ratio 0.03) is a property choice to add per problem. |

### Steps
- [ ] **B0. Fix and document the problem definitions.** (a) BiClamp: expose dT and build the R&F Example A comparison as a 2D extruded plate (72 x 47.7 cm, load 1000 kgf, vf 0.4, dT = 0, 1, 4, 10 K). Keep the current 3D STL version as a second case. (b) MBB: a definition matching [OO] (concrete, insulation void, 20/800 C, 240x80, F_m = 10 kN, vf 0.4). Done when: both build and their setup pictures show the supports, loads and thermal boundaries.
- [ ] **B1. Differentiable thermoelastic force.** A torch function `thermoelastic_force(T, x)`: per element `s_beta(x_e) * E * alpha * H (T_e - T_ref)`, vectorized and assembled with `index_add`. Validate against the existing NumPy `get_thermoelastic_force` (must agree to round-off) and by finite differences.
- [ ] **B2. Structural solve takes an extra, differentiable load.** `HexStructuralFEA.solve()` adds the thermal force to the RHS when given (keep the stress post-processing consistent). Done when: with a uniform dT on a clamped bar the displacement/stress equals the closed-form thermal-stress result.
- [ ] **B3. Coupled solver object** (for example `ThermoStructuralFEA`) exposing the single-solver interface: `solve(x)` runs the thermal solve, builds `F_th`, then the structural solve. It reports structural scaling to the compliance QOI. MMA and OC then work unchanged, and the gradient comes from autograd through both adjoints. Done when: the FD check on the objective and the volume constraint passes AND the gradient agrees with the hand-derived `ThermoElasticSensitivity` (the oracle) on a small case.
- [ ] **B4. Drivers and runner.** Remove the skip for ThermoStructural, build the coupled object in `topopt_run_benchmarks.py` and the notebooks, keep OC/Pareto/LevelSet skipped unless they only need `solve`. Done when: `topology_check.ipynb` runs BiClamp and MBB end to end.
- [ ] **B5. BiClamp validation against [RF] and the Reference.** dT = 0 gives a V-truss; growing dT makes the used volume fall below the limit (RF: 1319 -> 1069 -> 575). Compare qualitatively with RF Fig. 4-6 and quantitatively with the Reference row (J = 3.87e8, vf 0.142 at the current 3D setup).
- [ ] **B6. MBB steady-state validation against [OO].** t_f = infinity: c about 101,138 N mm, V = 0.3989, truss-like top with a thick insulating zone at the bottom (OO Fig. 6i). Also compare with the Reference row (J = 5.50e8, vf 0.40) for the existing definition.
- [ ] **B7. Transient thermal (OO).** Backward-Euler thermal solve in torch with design-dependent `C(x)` and `K(x)` for `t_N = 4` steps, objective summed over steps (eq. 12); gradients by autograd through the time loop (the sensitivities of OO eq. 29 are the oracle). Compare with OO Table 2/3 (t_f = 3600 s, 4 steps: c = 128,326 N mm, V = 0.4000, 783 iterations).
- [ ] **B8. Heaviside projection** (optional, after B6): add the three-field filter/projection (OO eq. 32-35) if the steady/transient results are too grey compared with the paper.
- [ ] **B9. Add both problems to the sweep and the comparison report** (`compare_results_to_reference.py`), and update `AUTOGRAD_STATUS_AND_HANDOFF.md`.

## Suggested order
A1 -> A2 -> A3 -> A4 (then A5 only if needed) -> A6; then B0 -> B1 -> B2 -> B3 -> B4 -> B5 (BiClamp) -> B6 (MBB steady) -> B7 (transient) -> B8/B9. Part A is small and self-contained, and its interpolation and design-dependent-load lessons carry straight into Part B, because the thermal load is also design-dependent (RF eq. 16, OO eq. 16).

## Risks and open questions
- **Non-monotonic compliance in Part B**: the thermal load depends on the design, so the same instability as self-weight can appear (RF report the volume constraint becoming inactive). Expect A1-style interpolation care in B.
- **Units and scale**: the current steel benchmarks use 1e5 to 1e8 N loads and 1 K, so absolute compliances will not match RF/OO (different units and materials); compare topologies, volume trends and ratios, and use the OO concrete setup for absolute numbers.
- **2D vs 3D**: RF and OO are 2D. PyTO's mesher is 3D hex; use the extruded (`ExtrudeZ`) single-layer setup and compare shapes and volume trends.
- **Reference values are from the older coupled code**, so a mismatch may again be a formulation change; treat them as a sanity check, not ground truth.
