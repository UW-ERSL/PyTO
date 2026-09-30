# Add a response

This recipe adds a new built-in quantity (a **response**) to PyTO. Once registered, it can be used in [expressions](../code/expressions.md) in code and in the GUI, like `Displacement(...)` or `Compliance()`. No gradient needs to be written: autograd differentiates it.

**When to use this rather than a [user-defined function](../code/user-defined-functions.md):** when the quantity is reusable, takes parameters (selections, components), should appear in the GUI's list, or should be combined with other quantities in formulas.

## The contract
A response is a function
```text
def my_response(sol, x, fe_solver, **params) -> torch scalar
```
- `sol`: the FE solution (displacements for structural and thermo-structural solvers, temperatures for thermal), part of the autograd graph.
- `x`: the physical density, one value per element.
- `**params`: its parameters. A **selection** parameter must be called `nodes` or `elements` and receives an index array, or `None` for the whole model. Other parameters receive option words (`"y"`, `"mean"`) or numbers (`p`).
- **torch operations only** on `sol` and `x`, returning one value.

## Example: flatness of a face
The standard deviation of one displacement component over a node selection. It is 0 when the face moves rigidly, so as a constraint it keeps a mounting face flat.
```python
import torch

def displacement_spread(sol, x, fe_solver, nodes=None, component="y"):
    """Standard deviation of one displacement component over the selected nodes (m): 0 = the face moves rigidly."""
    u = sol.reshape(-1, 3)[:, "xyz".index(component)]
    if nodes is not None:
        u = u[torch.as_tensor(nodes, dtype=torch.long)]
    return torch.sqrt(((u - u.mean()) ** 2).mean() + 1e-30)      # + 1e-30: smooth at 0
```

## Register it
### Permanently (in the source)
1. **Write the function** in `src/pyto/autodiff/qoi/responses.py`, next to the others.
2. **Add a `Response`** to the `RESPONSES` registry in the same file:
   ```python
   Response("DisplacementSpread", displacement_spread, _MECH,          # physics it applies to
            {"nodes": "node selection", "component": ("x", "y", "z")},  # parameters: allowed values or a description
            "m",                                                       # unit
            description="Standard deviation of a displacement component over a node selection (flatness)."),
   ```
   - **Physics sets:** `_ALL` (structural, thermal, thermo-structural), `_MECH` (structural, thermo-structural), `frozenset({THERMAL, COUPLED})`, or `frozenset({STRUCTURAL})`.
   - **`depends_on_solution=False`** for design-only quantities (like volume). It tells the cost estimate that no adjoint solve is needed.
3. **Declare positional parameters** in `POSITIONAL` in `src/pyto/topopt/expressions.py` if it takes any. The order is the order users write them in:
   ```python
   "DisplacementSpread": ("nodes", "component"),
   ```
   Without this entry, `DisplacementSpread(Top, y)` is rejected, and keyword arguments are rejected too.
4. **GUI list (optional):** add a label to `RESPONSE_LABELS` in `src/pyto/gui/optimization_setup.py`, e.g. `"Displacement spread": "DisplacementSpread"`. If the quantity takes parameters, extend `builtin_expression()` in the same file so choosing it fills in a sensible default formula, e.g. `DisplacementSpread({selection}, y)`.
5. **Test** (below).

### At run time (for experiments)
The same registration done from your script, with no source change:
```python
from pyto.autodiff.qoi import responses
from pyto.autodiff.qoi.responses import COUPLED, STRUCTURAL, Response
from pyto.topopt import expressions

responses.RESPONSES["DisplacementSpread"] = Response(
    "DisplacementSpread", displacement_spread, frozenset({STRUCTURAL, COUPLED}),
    params={"nodes": "node selection", "component": ("x", "y", "z")}, unit="m")
expressions.POSITIONAL["DisplacementSpread"] = ("nodes", "component")
```
A complete runnable example of run-time registration (a design-only "grayness" measure) is [`docs/examples/custom_response.py`](../examples/custom_response.py).

## Use it
```python
from pyto.topopt.spec import ConstraintSpec, ObjectiveSpec, OptimizationSpec, compile_spec

top = mesh.getNodesOnBoundingBoxPlane(1, False)
spec = OptimizationSpec(objective=ObjectiveSpec("Compliance()"),
                        constraints=[ConstraintSpec("VolumeFraction()", "<=", 0.3),
                                     ConstraintSpec("DisplacementSpread(Top, y)", "<=", 1e-5)])
to_params = compile_spec(spec, {"Top": {"nodes": top}})
```
On the benchmark cantilever (4 000 dofs), the full design's top face has a spread of 3.2e-5 m, and `check_gradient` gives relative errors of 9e-8 for this constraint.

## Test it
Add to `tests/test_responses.py`:
1. **Value** against an independent calculation on a known case: a rigid translation gives 0, and a linear field gives its known deviation.
2. **Gradient** against finite differences. The file's `_check_gradient(fe, "DisplacementSpread", nodes=..., component="y")` helper does this.
3. **Expression** parses and compiles: `compile_expression("DisplacementSpread(Top, y)", selections)`.

See [Testing your extension](testing-your-extension.md).

## Pitfalls
- **Non-smooth operations** (`max`, `abs`, `sqrt` at 0) give gradients that jump or are undefined. Smooth them, as with the `+ 1e-30` above.
- **Thermo-structural solvers** return the displacement as `sol`. The temperature is `fe_solver.temperature`, which is also on the autograd graph. `responses.temperature` shows how to handle both physics.
- **Parameter names:** only `nodes` and `elements` are treated as selections. `p` must be a number > 1; other values are checked against the tuple of allowed values you give in `params`.
- **Stateful quantities** (values that change between calls at the same design) break finite-difference checks and MMA. Keep responses pure.

---
Next: [Add a manual sensitivity](add-a-manual-sensitivity.md) · [Back to contents](../README.md)
