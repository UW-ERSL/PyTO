"""Gradcheck the core autodiff primitive: SparseLinearSolve.

This is deliberately isolated from any FEA/mesh machinery -- it verifies
the hand-written adjoint in torch_spsolve.SparseLinearSolve.backward()
(solve A^T y = grad_x; grad_b = y; grad_A[i,j] = -y_i * x_j) against
torch.autograd.gradcheck's numerical Jacobian on a small random sparse
SPD system, independent of any physics.
"""
import numpy as np
import torch

from pyto.autodiff.sparse_solve import SparseLinearSolve, Solvers


def _random_spd_sparse(n, density=0.4, seed=0):
    rng = np.random.default_rng(seed)
    dense = rng.uniform(-1.0, 1.0, size=(n, n))
    dense[rng.uniform(size=(n, n)) > density] = 0.0
    spd = dense @ dense.T + n * np.eye(n)  # guarantee SPD and well-conditioned
    idx = np.nonzero(spd)
    indices = torch.tensor(np.array(idx), dtype=torch.long)
    values = torch.tensor(spd[idx], dtype=torch.float64, requires_grad=True)
    return indices, values, spd.shape


def test_sparse_linear_solve_gradcheck():
    torch.manual_seed(0)
    n = 8
    indices, values, shape = _random_spd_sparse(n)
    b = torch.randn(n, dtype=torch.float64, requires_grad=True)

    def f(values, b):
        mtrx = torch.sparse_coo_tensor(indices, values, shape).coalesce()
        return SparseLinearSolve.apply(mtrx, b, "spsolve", {})

    assert torch.autograd.gradcheck(f, (values, b), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_dpcg_is_not_implemented_and_warns_before_falling_back():
    # The torch path has no DPCG; it must say so, not silently substitute a direct solve.
    import pytest
    from pyto.autodiff.sparse_solve import solve
    indices, values, shape = _random_spd_sparse(6)
    A = torch.sparse_coo_tensor(indices, values.detach(), shape)
    with pytest.warns(UserWarning, match="not implemented"):
        solve(A, torch.ones(6, dtype=torch.float64), solver=Solvers.DPCG)


# ---------------------------------------------------------------- plan S8: PARDISO factor reuse in the adjoint
def test_pardiso_adjoint_reuses_factors_and_matches_spsolve(structural_problem):
    import gc
    import weakref
    import numpy as np
    import torch
    pytest = __import__("pytest")
    sp = pytest.importorskip("pypardiso")
    import pyto.autodiff.sparse_solve as S
    import pyto.solve.numpy_backend as lin
    from pyto.autodiff.material_model import MaterialModel
    from pyto.physics.structural.hex_structural_fea import HexStructuralFEA
    mesh, mat, bc, ebf, _ = structural_problem
    x0 = torch.tensor(0.3 + 0.6 * np.random.default_rng(0).random(mesh.num_elems), dtype=torch.float64)

    def grads(solver):
        fe = HexStructuralFEA(mesh=mesh, mat_prop=mat, bc=bc, solver=solver)
        x = x0.clone().requires_grad_(True)
        u = fe.solve(x, MaterialModel.SIMP)
        f1, f2 = (u ** 2).sum(), u.abs().max() + u[::7].sum()
        g1 = torch.autograd.grad(f1, x, retain_graph=True)[0]      # two backward passes on one graph,
        g2 = torch.autograd.grad(f2, x)[0]                          # as MMA does for objective + constraints
        return g1, g2

    created = []
    original_init = S._PardisoFactors.__init__

    def spy(self, A):
        original_init(self, A)
        created.append(weakref.ref(self))

    S._PardisoFactors.__init__ = spy
    try:
        p1, p2 = grads(lin.Solvers.PARDISO)
    finally:
        S._PardisoFactors.__init__ = original_init
    s1, s2 = grads(lin.Solvers.SPSOLVE)
    assert torch.allclose(p1, s1, rtol=1e-8, atol=1e-12 * s1.abs().max())
    assert torch.allclose(p2, s2, rtol=1e-8, atol=1e-12 * s2.abs().max())
    gc.collect()
    assert created and all(r() is None for r in created)            # factors freed with the graph
