# Stress

This page explains how PyTO measures stress for optimization: relaxed element stress, the von Mises stress, the p-norm aggregate, and the adaptive "max stress" constraint. It is short and practical.

## Element stress
Stress is evaluated at each element's centre, from the strain $\boldsymbol{\varepsilon}_e = \mathbf{B}\mathbf{u}_e$:
- $\mathbf{B}$ is the strain-displacement matrix at the centre, scaled by $2/h$ for element size $h$;
- $\mathbf{D}$ is the isotropic elasticity matrix.

For thermo-structural solvers the free thermal strain is removed first: $\boldsymbol{\varepsilon}_e - \alpha(\bar{T}_e - T_{ref})[1,1,1,0,0,0]^T$.

## Relaxation
With SIMP, grey and void elements would still show solid-material stresses (the singularity problem). PyTO scales the stress with a relaxation factor:

$$\boldsymbol{\sigma}_e = \left(E_v + (1 - E_v)\,\tilde{x}_e^{\,q}\right)\mathbf{D}\boldsymbol{\varepsilon}_e, \qquad q = 0.5$$

Void elements then carry (almost) no stress, and the stress constraint doesn't prevent material from being removed.

## von Mises and p-norm
$$\sigma^{vm}_e = \sqrt{\tfrac12\left[(\sigma_{xx}-\sigma_{yy})^2 + (\sigma_{yy}-\sigma_{zz})^2 + (\sigma_{zz}-\sigma_{xx})^2\right] + 3(\tau_{xy}^2 + \tau_{yz}^2 + \tau_{xz}^2)}$$

A constraint on every element would be one constraint per element. PyTO aggregates them into one smooth measure:

$$P = \Big(\sum_e (\sigma^{vm}_e)^{p}\Big)^{1/p}, \qquad p = 6$$

- $P \ge \max_e \sigma^{vm}_e$, and it gets closer to the maximum as $p$ grows, but less smooth.
- `StressPNorm()` is $P$; `StressFailureFactor()` is $P / \sigma_{yield}$.

## MaxStress: adaptive normalization
Because $P$ overestimates the maximum, a limit on $P$ is conservative. `MaxStress() <= limit` (the `MAX_VONMISES_STRESS` constraint) multiplies $P$ by a factor $c$, updated every iteration toward $\max_e\sigma^{vm}_e / P$ (after Le et al. 2010; see also De Leon et al. 2015):

$$c_{k+1} = 0.25\,\frac{\max_e \sigma^{vm}_e}{P} + 0.75\,c_k$$

The constraint then tracks the real peak stress. Because $c$ changes between iterations, the constraint is **stateful**: it can't have a fixed manual gradient, and finite-difference checks reset $c$ first.

## Gradient
The p-norm is differentiated by autograd, or by the manual adjoint formula ([Add a manual sensitivity](../extending/add-a-manual-sensitivity.md#worked-example-the-p-norm-stress-gradient)). The two agree to about 1e-14. One adjoint solve per iteration is needed.

## Practical advice
- Start with a **loose** stress limit, or a compliance objective plus a stress constraint. A tight stress limit from the first iteration can make MMA struggle.
- Stress concentrates at re-entrant corners and at point loads. Spread loads over faces, and expect the optimizer to round corners.
- Check the final design's stress on the solid part (`final_design_summary`), not the p-norm.

## References
- Le, Norato, Bruns, Ha, Tortorelli, *Stress-based topology optimization for continua*, SMO 41 (2010).
- De Leon, Alexandersen, Fonseca, Sigmund, *Stress-constrained topology optimization for compliant mechanism design*, SMO 52 (2015).
- Duysinx & Bendsøe, *Topology optimization of continuum structures with local stress constraints*, IJNME 43 (1998).

---
Next: [Thermo-structural coupling](thermo-structural.md) · [Back to contents](../README.md)
