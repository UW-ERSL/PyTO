# Sensitivities

This page explains how the gradient of an objective with respect to the design is computed: the adjoint method, how PyTO gets it from automatic differentiation, and how it is chained through the filter. It is short and practical.

## The problem
An objective $f(\mathbf{u}, \tilde{\mathbf{x}})$ depends on the design directly, and through the solution of $\mathbf{K}(\tilde{\mathbf{x}})\mathbf{u} = \mathbf{F}(\tilde{\mathbf{x}})$. With $N$ design variables, computing $d\mathbf{u}/d\tilde{x}_e$ for each one would take $N$ solves. The **adjoint method** needs one.

## Adjoint method
Differentiate the state equation: $\mathbf{K}\,\frac{d\mathbf{u}}{d\tilde{x}_e} = \frac{\partial \mathbf{F}}{\partial \tilde{x}_e} - \frac{\partial \mathbf{K}}{\partial \tilde{x}_e}\mathbf{u}$. Then solve **one** extra system, $\mathbf{K}^T\boldsymbol{\lambda} = \partial f / \partial \mathbf{u}$:

$$\frac{df}{d\tilde{x}_e} = \frac{\partial f}{\partial \tilde{x}_e} + \boldsymbol{\lambda}^T\left(\frac{\partial \mathbf{F}}{\partial \tilde{x}_e} - \frac{\partial \mathbf{K}}{\partial \tilde{x}_e}\mathbf{u}\right)$$

- $\partial\mathbf{K}/\partial\tilde{x}_e$ touches only element $e$: $E'(\tilde{x}_e)\,\mathbf{K}^0_e$. So the last term is cheap: $-E'(\tilde{x}_e)\,\boldsymbol{\lambda}_e^T\mathbf{K}^0_e\mathbf{u}_e$ per element.
- **Compliance** $f = \mathbf{F}^T\mathbf{u}$ is **self-adjoint**: $\boldsymbol{\lambda} = \mathbf{u}$, no extra solve, and $df/d\tilde{x}_e = -E'(\tilde{x}_e)\,\mathbf{u}_e^T\mathbf{K}^0_e\mathbf{u}_e$ (plus a term for design-dependent loads).
- **Prescribed-value problems** (driven by fixed temperatures, not loads) have the opposite sign behaviour. PyTO maximizes conduction for them (`compliance_sign`).

## Automatic differentiation through the solve
PyTO doesn't hand-derive most gradients. The forward computation $\mathbf{x} \to \tilde{\mathbf{x}} \to \mathbf{K} \to \mathbf{u} \to f$ is written in PyTorch, and the linear solve is a custom autograd function (`SparseLinearSolve` in `autodiff/sparse_solve.py`):
- **forward:** solve $\mathbf{K}\mathbf{u} = \mathbf{F}$;
- **backward:** given $\bar{\mathbf{u}} = \partial f / \partial \mathbf{u}$, solve $\mathbf{K}^T\boldsymbol{\lambda} = \bar{\mathbf{u}}$ (the adjoint system), and return $\bar{\mathbf{F}} = \boldsymbol{\lambda}$ and $\bar{K}_{ij} = -\lambda_i u_j$ for the stored entries.

Autograd then carries these back through the assembly, the material law, the projection and the filter. The result is exactly the adjoint gradient above, for **any** objective written with torch operations. With PARDISO, the forward factorization is reused for the adjoint solve, since $\mathbf{K}$ is symmetric.

Checks: `tests/test_sparse_solve_gradcheck.py` verifies the primitive with `torch.autograd.gradcheck`, and `check_gradient` compares any formulation with finite differences ([Gradients](../code/gradients.md)).

## Chain rule through filter and projection
MMA's variables are the raw densities $\mathbf{x}$. With $\bar{\mathbf{x}} = \mathbf{H}\mathbf{x} / \mathbf{H}_s$ (filter) and $\tilde{\mathbf{x}} = P(\bar{\mathbf{x}})$ (projection):

$$\frac{df}{dx_i} = \sum_e \frac{H_{ei}}{H_{s,e}}\; P'(\bar{x}_e)\; \frac{df}{d\tilde{x}_e}$$

Autograd applies this automatically. The manual gradient path (`manual_step` in `drivers/mma.py`) applies it by hand: multiply by $P'$, divide by $H_s$, multiply by $\mathbf{H}^T$.

## Cost
Per MMA iteration:
- one forward solve;
- one adjoint solve for each objective or constraint that depends on the solution. Volume and mass need none, and compliance needs none with manual gradients.

The measured backward/forward time ratio is 0.4–0.8 with PARDISO on 10k–40k-dof problems ([Validation](../reference/validation.md)).

---
Next: [Stress](stress.md) · [Back to contents](../README.md)
