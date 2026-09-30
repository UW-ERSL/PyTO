# Testing your extension

This page lists the checks that show a new response, gradient, filter, physics or method is correct, and where PyTO's tests for each already live. It is for developers about to submit or rely on an extension.

**The principle:** every numerical addition is compared with something computed independently: finite differences, autograd, an analytic value, or a previous validated result. "It runs" is not a test.

## Checks by kind of change
| You added | Check | Template |
|---|---|---|
| a response | value on a case with a known answer; gradient vs finite differences; parses in an expression | `tests/test_responses.py` (`_check_gradient`), `tests/test_expressions.py` |
| a user function or expression (in your own work) | `check_gradient(to_params, fe)` before the long run | [Gradients](../code/gradients.md#checking-a-gradient) |
| a manual sensitivity | equals autograd to ~1e-12 at a random design; an MMA run gives identical iterates | `tests/test_manual_sensitivities.py` (`_compare`, `_history`) |
| a filter | rows sum to 1; `H @ x` has the intended property for random `x`; gradient check with it on | `docs/examples/custom_filter.py` |
| a physics | patch test with an analytic answer; gradient check; a few MMA iterations lower the objective | `tests/test_thermoelastic.py`, `tests/test_qoi_gradients.py` |
| an optimization method | a few iterations on the structural fixture run, lower the objective and respect the volume | `tests/test_drivers_smoke.py` |
| a material model | torch law = NumPy law; autograd derivative = NumPy derivative = finite differences | a new small parametrized test |
| GUI changes | the logic in `gui/optimization_setup.py` tested without Qt; windows driven headless | `tests/test_gui_optimization_setup.py`, `tests/test_gui_topopt_window.py` |

## Fixtures
`tests/conftest.py` provides small real problems, meshed once per test session:

| Fixture | What |
|---|---|
| `structural_problem` / `structural_fe_solver` | ShortCantileverTipLoad at a small size (a fresh solver per test) |
| `thermal_problem` / `thermal_fe_solver` | HeatPlate at a small size |
| `structural_KE`, `thermal_KE` | the element matrices |

Use them, rather than building meshes in each test, to keep the suite fast (about a minute in total).

## Finite-difference checks done right
- **Where:** evaluate at an **interior random design** (densities 0.3–0.9). At 0 or 1 some terms vanish and hide errors.
- **Step:** central differences with a relative step of about `1e-4` to `1e-6`. Too large is dominated by curvature; too small by round-off.
- **How much:** check a few random elements, not all; that is enough to catch formula errors.
- **Scale:** compare with a relative tolerance that takes the gradient's size into account (`test_responses._check_gradient` shows one way).
- **Stateful quantities:** reset any state that changes between calls (e.g. `to_params.stress_scaling`) before every evaluation. `check_gradient` does this.

## Regression against the benchmarks
For changes that could alter results (solvers, filters, drivers, material laws), run the affected benchmark problems and compare with a saved sweep:
```python
from pyto.examples_benchmarks.topopt_run_benchmarks import *
run_all_benchmarks_isolated(optimizationMethods=[TO_METHODS.DENSITYMMA],
                            problems=benchmarks_structural_2_5D_problems_1)
```
Then compare objective and volume fraction per problem with the reference ([Comparing results](../postprocessing/comparing-results.md)). Differences within about 1 % are usually noise from the optimizer path; larger ones need an explanation.

## Running the tests
```bash
pip install -r requirements-dev.txt
PYTHONPATH=src MPLBACKEND=Agg PYVISTA_OFF_SCREEN=true QT_QPA_PLATFORM=offscreen pytest tests -q
```
More in [Testing](../reference/testing.md).

---
Next: [Part 5: Theory](../theory/density-method.md) · [Back to contents](../README.md)
