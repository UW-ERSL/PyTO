# Gradients

This page explains how PyTO computes the gradient (sensitivity) of objectives and constraints, how to choose between automatic differentiation and manual gradients, and how to check a gradient. It is for anyone who wants to trust, speed up or extend the optimization.

## Automatic differentiation (default)
For MMA and OC, PyTO builds each objective and constraint as a PyTorch computation from the design `x`:

filter → projection → FE solve → quantity.

`torch.autograd` then differentiates backwards through all of it. The FE solve is a custom autograd function (`pyto.autodiff.sparse_solve`) whose backward pass solves the **adjoint system** with the same matrix. So:
- **any** quantity written with torch operations (built-in, [expression](expressions.md) or [Python function](user-defined-functions.md)) gets an exact gradient, with no derivation needed;
- the cost is about **one extra linear solve per quantity that depends on the solution**. Volume and mass need none, and with PARDISO the factorization is reused.

## Manual gradients
For the most common quantities, PyTO also has sensitivities derived by hand, in `pyto.topopt.manual_sensitivities`. They give the same values without building the autograd graph.

| Quantity | Manual gradient | Physics |
|---|---|---|
| `COMPLIANCE` (objective or constraint) | yes, needs no adjoint solve (self-adjoint) | structural, thermal, thermo-structural |
| `PNORM_STRESS` (objective or constraint) | yes, one adjoint solve | structural |
| `VOLUME_FRACTION` (`<=`), `VOLUME_FRACTION_MIN` (`>=`) | yes, trivial | all |
| everything else (mass, max stress, failure factor, expressions, Python functions) | no, automatic differentiation only | – |

Also not covered: design-dependent body forces (self-weight), per-problem conductivity laws (`ConductivityPenalty`, `ConductivityVoidRatio`), and compliance of problems driven by varying prescribed values together with loads. `manual_support(to_params, fe)` returns `(False, reason)` for these.

Turn it on with:
```python
to_params.Gradient = "manual"                 # default "autodiff"
```
or `MethodSpec(gradient="manual")` in a spec, or **Method → Gradient → Manual** in the GUI.

It applies to **MMA and OC**. For MMA, the manual gradient is taken with respect to the physical density and chained back through the Heaviside projection and the density filter by hand. **Pareto and LevelSet** always use their own derived sensitivities.

### Why use manual gradients
- **Speed:** no autograd graph. On a 3 000-dof cantilever, 10 MMA iterations took 1.70 s with manual gradients against 2.31 s with autograd, with identical iterates.
- **Reference:** an independent derivation to check autograd against, and a starting point for methods that need derived sensitivities.

### How they are verified
`tests/test_manual_sensitivities.py` compares every manual gradient with autograd. Agreement is at the 1e-14 level: 9e-15 for compliance, 1.7e-14 for p-norm stress, and below 1e-6 for thermo-structural compliance. The tests also run full MMA and OC optimizations both ways and require the same iterates. To add a manual gradient of your own: [Add a manual sensitivity](../extending/add-a-manual-sensitivity.md).

## Checking a gradient
```python
from pyto.topopt.capabilities import check_gradient

for r in check_gradient(to_params, fe, n_elements=5, rel_step=1e-4, tol=1e-3):
    print(r.name, f"{r.max_rel_error:.1e}", "OK" if r.ok else "MISMATCH")
```
- It compares the gradient in use (autograd, or manual if `to_params.Gradient == "manual"`) with central finite differences on `n_elements` random elements.
- The comparison is made at a random interior design (densities 0.3–0.9).
- The GUI's **Check gradient** button runs the same function.

A mismatch points to one of:
- a non-smooth quantity (`max`, `abs`, `where`);
- a function that uses NumPy on the solution;
- a finite-difference step that's too large or too small for a strongly nonlinear quantity. Try `rel_step=1e-5`.

## Cost estimate
```python
from pyto.topopt.capabilities import cost_estimate
print(cost_estimate(spec, selections))
# "Per iteration: 1 forward FE solve + 2 adjoint solves."
```
It counts the linear solves one MMA iteration needs. With manual gradients it counts only what the manual path needs (0 for compliance, 1 for p-norm stress).

## Measured cost
`tests/test_gradient_cost.py` times forward, forward+backward and backward separately for a given problem size:
```bash
PYTHONPATH=src python tests/test_gradient_cost.py --ndof 25000 --repeat 10 --solver PARDISO
```
Results for this repository are summarized in [Validation](../reference/validation.md).

---
Next: [Thermal and thermo-structural problems](thermal-and-thermostructural.md) · [Back to contents](../README.md)
