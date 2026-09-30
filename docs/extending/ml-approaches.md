# Machine-learning approaches

This page describes how neural networks and other machine-learning methods can be combined with PyTO, and which part of PyTO each idea plugs into. It is for researchers planning ML work on topology optimization. It gives ideas and structure only, with no tested code.

The key property that makes PyTO a good base for this: **the whole chain from density to objective is differentiable in PyTorch**. `fe.solve(x)` returns a tensor connected to `x`, and every objective and constraint, including [expressions](../code/expressions.md) and [user functions](../code/user-defined-functions.md), is built from torch operations. A neural network placed *in front of* `x` is therefore trained with ordinary `loss.backward()`, and the adjoint FE solve happens inside the backward pass automatically.

## 1. Neural reparameterization (TOuNN-style)
**Idea** (Chandrasekhar & Suresh, *TOuNN: Topology Optimization using Neural Networks*, Struct. Multidisc. Optim. 2021; code: [github.com/UW-ERSL/TOuNN](https://github.com/UW-ERSL/TOuNN)):
- Don't optimize one density per element. Instead, a neural network $\rho = \text{NN}_\theta(\mathbf{p})$ maps a **point's coordinates** to its density.
- The design variables are the network weights $\theta$.
- The network is trained by minimizing a loss built from the FE analysis.

**Why it's interesting:**
- The design is a continuous function of space. It can be sampled on a finer grid than the analysis mesh, for crisp high-resolution boundaries.
- Smoothness and length scale come from the network and its input encoding (e.g. Fourier features) rather than from a density filter.
- Symmetry and extrusion can be built into the inputs: reflect a coordinate for symmetry, drop a coordinate for extrusion.
- It uses standard ML optimizers (Adam) and tooling.

**How it maps onto PyTO:**

| TOuNN step | PyTO piece |
|---|---|
| sample points | `mesh.elem_centers` (as a float64 torch tensor), normalized to about [−1, 1] |
| network | any `torch.nn.Module` with a sigmoid output (one density per point); keep it float64 like the rest of PyTO |
| FE analysis | `fe.solve(rho, MaterialModel.SIMP)`; differentiable, with the adjoint in the backward pass |
| objective | `compute_objective_and_gradient(to_params, sol, rho, fe, KE, MaterialModel.SIMP)`, or any response or expression; `compliance_sign(fe)` for prescribed-value problems |
| volume constraint | a penalty in the loss: $L = J/J_0 + \alpha\,(\bar\rho/V^* - 1)^2$, with $\alpha$ increased over the epochs (continuation), as in TOuNN; or an augmented Lagrangian |
| penalization continuation | raise the SIMP exponent over the epochs with `set_SIMP_STRUCTURAL_PENALTY` (`autodiff/material_model.py`), which TOuNN also does |
| optimizer | `torch.optim.Adam(net.parameters(), lr=...)` |
| plotting | `fe.mesh.setPseudoDensity(rho.detach())`, then any plot method |

Outline of the training loop (a structural sketch, not a tested example):
```python
net = MyNet().double()                                     # coordinates -> density in (0, 1)
opt = torch.optim.Adam(net.parameters(), lr=1e-2)
pts = torch.as_tensor(normalize(fe.mesh.elem_centers))
for epoch in range(n_epochs):
    opt.zero_grad()
    rho = net(pts).flatten()
    sol = fe.solve(rho, MaterialModel.SIMP)
    J = compute_objective_and_gradient(to_params, sol, rho, fe, KE, MaterialModel.SIMP)
    loss = J / J0 + alpha * (rho.mean() / V - 1.0) ** 2
    loss.backward()                                        # adjoint FE solve happens here
    opt.step()
    alpha = min(alpha_max, alpha * alpha_growth)
```
**Where to put it:**
- **As a new driver.** Follow the contract in [Add an optimization method](add-an-optimization-method.md) (return `solution, history, success, message, n_fea`, keep the density on the mesh, call `iteration_callback`). Then register it so it appears in benchmark sweeps and the GUI.
- **Constraints** other than volume need their own penalty or Lagrangian terms. `compute_constraint_and_gradient` returns every constraint in the `g ≤ 0` form, and `torch.relu(g) ** 2` is a simple penalty.

**Pitfalls:**
- float32 networks mixed with float64 FE: convert once, at the network output.
- Forgetting `.detach()` when storing history or plotting (memory keeps growing).
- A density filter is usually unnecessary, and can fight the network's own smoothness. Build the drivers' `H` in only if you want it.
- For the final design, threshold or project the network output, then evaluate the 0/1 design once more ([Thresholding and evaluation](../postprocessing/thresholding-and-evaluation.md)).

## 2. Surrogate models and warm starts
**Idea:** train a model on many solved problems to predict an optimized density directly from the problem description (loads, supports, volume fraction). Use the prediction as the **starting design** for MMA, which then needs far fewer iterations and still gives a physically optimized result.

**PyTO pieces:**
- **Data:** the benchmark runner (`topopt_run_benchmarks.py`) writes densities, histories and summaries for many problems. Generate variations (load positions, volume fractions) with the problem builders in `examples_benchmarks/`.
- **Warm start:** MMA currently starts from a uniform design. Add a starting-design argument to the driver, where `x0` is created in `topopt/drivers/mma.py`, and pass the prediction.
- **Pitfall:** voxel meshes differ in size between problems. Predict on a fixed grid and interpolate to `mesh.elem_centers`, or use point-based models as in TOuNN.

## 3. Learned filters and projections
**Idea:** replace the fixed density filter or the Heaviside projection with a small trainable module, for example one that learns manufacturable feature shapes.

**PyTO pieces:** the filter and projection sit in `chain()` in `topopt/drivers/mma.py` ([Add a filter](add-a-filter.md)). A torch module there is differentiated by autograd like everything else. Train it offline, freeze it during optimization, and keep the manual-gradient path off.

## 4. ML in post-processing
PyTO already uses one: STL recovery (`io/topopt_stl_recovery.py`) trains a small 3D CNN on the voxel density to smooth it before extracting the surface ([Export](../postprocessing/export.md)). Similar models can:
- remove small floating features;
- enforce minimum wall thickness on the final geometry;
- upsample a coarse design to a finer one.

## 5. Learned physics
**Idea:** replace or accelerate the FE solve with a learned model: a neural operator, or a learned preconditioner.

**PyTO pieces:** a new solver following [Add a physics](add-a-physics.md), or a preconditioner in `solve/`. **Caution:** the gradient then comes from the learned model, not the true physics. Check against the real FE solve regularly, for example with `check_gradient` against a normal solver, and always evaluate final designs with the real solver.

---
Next: [Testing your extension](testing-your-extension.md) · [Back to contents](../README.md)
