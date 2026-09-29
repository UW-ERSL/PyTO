"""Library of named, differentiable responses (plan S1, notes/6_GUI_OPTIMIZATION_FORMULATION_AND_AD_PLAN.md).

A *response* is a physical quantity computed from the FE solution and the design, written with torch operations so
autograd differentiates it through the FE solve. Objectives and constraints are built from responses (directly, or
through the expression compiler in pyto.topopt.expressions); the software differentiates, the user never writes a
gradient.

Every response function has the signature

    fn(sol, x, fe_solver, **params) -> torch scalar

where `sol` is what fe_solver.solve returned (displacement for structural and thermo-structural solvers, temperature
for thermal ones) and `x` is the physical (filtered) density. Selections are passed as index arrays: `nodes` for
node-based responses, `elements` for region responses; None means the whole model.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from pyto.autodiff.material_model import MaterialModel

STRUCTURAL, THERMAL, COUPLED = "structural", "thermal", "thermo-structural"
AGGREGATES = ("mean", "sum", "pnorm")
DEFAULT_P = 8.0


@dataclass(frozen=True)
class Response:
    name: str
    fn: Callable
    physics: frozenset
    params: dict = field(default_factory=dict)   # parameter name -> allowed values (tuple) or a type description (str)
    unit: str = "-"
    depends_on_solution: bool = True             # False: no adjoint solve (volume, mass)
    smooth: bool = True
    description: str = ""


def physics_of(fe_solver) -> str:
    """'structural', 'thermal' or 'thermo-structural' for an FE solver instance."""
    from pyto.physics.thermal.hex_thermal_fea import HexThermalFEA
    from pyto.physics.thermoelastic.thermostructural_fea import ThermoStructuralFEA
    if isinstance(fe_solver, ThermoStructuralFEA):
        return COUPLED
    if isinstance(fe_solver, HexThermalFEA):
        return THERMAL
    return STRUCTURAL


def _index(ix, device):
    return None if ix is None else torch.as_tensor(np.asarray(ix, dtype=np.int64).reshape(-1), device=device)


def aggregate(values: torch.Tensor, how: str = "mean", p: float = DEFAULT_P) -> torch.Tensor:
    """Reduce per-node/per-element values to one scalar.

    mean, sum, or pnorm = (sum |v|^p)^(1/p), a smooth upper bound of max |v| (the true max is not differentiable).
    """
    if how == "mean":
        return values.mean()
    if how == "sum":
        return values.sum()
    if how == "pnorm":
        scale = values.detach().abs().max().clamp_min(1e-300)   # scale for overflow safety; exact math unchanged
        return scale * ((values.abs() / scale) ** p).sum() ** (1.0 / p)
    raise ValueError(f"aggregate must be one of {AGGREGATES}, got {how!r}")


# ------------------------------------------------------------------------------------------------ response functions
def compliance(sol, x, fe_solver):
    """Structural compliance u^T K u (thermal: the thermal compliance), SIMP material law as in the MMA driver."""
    from pyto.autodiff.qoi.compliance import compute_compliance_torch
    return compute_compliance_torch(sol, x, fe_solver, fe_solver.elem_stiff_torch[0].to(sol.dtype), MaterialModel.SIMP)


def volume_fraction(sol, x, fe_solver, elements=None):
    """Mean physical density over the region (the whole model by default)."""
    e = _index(elements, x.device)
    return x.mean() if e is None else x[e].mean()


def mass(sol, x, fe_solver):
    from pyto.autodiff.qoi.mass import compute_mass_torch
    return compute_mass_torch(x, fe_solver)


def displacement(sol, x, fe_solver, nodes=None, component="magnitude", aggregate_by="mean", p=DEFAULT_P):
    """Displacement at the selected nodes: component x, y, z (signed) or magnitude, then aggregated."""
    u = sol.reshape(-1, 3)
    n = _index(nodes, sol.device)
    if n is not None:
        u = u[n]
    if component == "magnitude":
        v = torch.sqrt((u ** 2).sum(dim=1) + 1e-300)
    else:
        v = u[:, "xyz".index(component)]
    return aggregate(v, aggregate_by, p)


def temperature(sol, x, fe_solver, nodes=None, aggregate_by="mean", p=DEFAULT_P):
    """Temperature at the selected nodes (thermal, or the thermal field of a thermo-structural solver)."""
    T = fe_solver.temperature if physics_of(fe_solver) == COUPLED else sol
    n = _index(nodes, T.device)
    return aggregate(T if n is None else T[n], aggregate_by, p)


def reaction_force(sol, x, fe_solver, nodes=None, component="y"):
    """Sum of the support reaction component over the selected nodes: (K(x) u - f) on those dofs.

    K u is formed from the stored nonzeros with index_add (not torch.sparse.mm, whose backward w.r.t. K is dense;
    see core/bc.py apply_dirichlet_bc_torch)."""
    K = fe_solver.stiff_mtrx.coalesce()
    idx, val = K.indices(), K.values()
    Ku = torch.zeros(sol.shape[0], dtype=sol.dtype, device=sol.device).index_add(0, idx[0], val * sol[idx[1]])
    R = (Ku - fe_solver.total_force).reshape(-1, 3)[:, "xyz".index(component)]
    n = _index(nodes, sol.device)
    return R.sum() if n is None else R[n].sum()


def strain_energy(sol, x, fe_solver, elements=None):
    """Strain energy 1/2 u^T K u of the region (half the compliance for the whole model)."""
    from pyto.autodiff.material_model import get_structural_material_model_scaling_torch
    KE = fe_solver.elem_stiff_torch[0].to(sol.dtype)
    edof = torch.as_tensor(fe_solver.mesh.edofMat, dtype=torch.long, device=sol.device)
    u_e = sol[edof]
    ce = torch.einsum("ei,ij,ej->e", u_e, KE, u_e) * get_structural_material_model_scaling_torch(x, MaterialModel.SIMP)
    e = _index(elements, sol.device)
    return 0.5 * (ce.sum() if e is None else ce[e].sum())


def stress_pnorm(sol, x, fe_solver):
    """p-norm of the element von Mises stress over the model (smooth stand-in for the maximum)."""
    from pyto.autodiff.qoi.stress import compute_pnorm_stress_autograd
    return compute_pnorm_stress_autograd(sol, x, fe_solver)[0]


def stress_failure_factor(sol, x, fe_solver):
    """p-norm von Mises stress / yield strength (<= 1 means the smoothed peak stress is below yield)."""
    return stress_pnorm(sol, x, fe_solver) / fe_solver.mat_prop.yield_strength


# ------------------------------------------------------------------------------------------------ registry
_ALL = frozenset({STRUCTURAL, THERMAL, COUPLED})
_MECH = frozenset({STRUCTURAL, COUPLED})
_AGG = {"aggregate_by": AGGREGATES, "p": "float > 1 (pnorm only)"}

RESPONSES = {r.name: r for r in (
    Response("Compliance", compliance, _ALL, unit="J",
             description="Compliance u^T K u (stiffness measure; thermal: thermal compliance). Lower = stiffer."),
    Response("VolumeFraction", volume_fraction, _ALL, {"elements": "element selection"}, "-", depends_on_solution=False,
             description="Mean material density over a region (the whole model by default)."),
    Response("Mass", mass, _MECH, unit="kg", depends_on_solution=False, description="Total mass."),
    Response("Displacement", displacement, _MECH,
             {"nodes": "node selection", "component": ("x", "y", "z", "magnitude"), **_AGG}, "m",
             description="Displacement at selected nodes; pnorm is a smooth upper bound of the maximum."),
    Response("Temperature", temperature, frozenset({THERMAL, COUPLED}), {"nodes": "node selection", **_AGG}, "K",
             description="Temperature at selected nodes; pnorm is a smooth upper bound of the maximum."),
    Response("ReactionForce", reaction_force, _MECH, {"nodes": "node selection", "component": ("x", "y", "z")}, "N",
             description="Sum of support reactions over selected (fixed) nodes."),
    Response("StrainEnergy", strain_energy, frozenset({STRUCTURAL}), {"elements": "element selection"}, "J",
             description="Strain energy of a region."),
    # Stress: structural only. The differentiable stress uses the total strain; for thermo-structural problems the
    # thermal strain would have to be subtracted first (HexStructuralFEA.postprocess does that, the torch path not yet).
    Response("StressPNorm", stress_pnorm, frozenset({STRUCTURAL}), unit="Pa",
             description="p-norm of von Mises stress (smooth stand-in for the maximum stress)."),
    Response("StressFailureFactor", stress_failure_factor, frozenset({STRUCTURAL}), unit="-",
             description="p-norm von Mises stress / yield strength; <= 1 keeps the smoothed peak below yield."),
)}


def available_responses(physics: str) -> list:
    """Names of the responses valid for a physics ('structural', 'thermal', 'thermo-structural')."""
    return [name for name, r in RESPONSES.items() if physics in r.physics]


def evaluate(name: str, sol, x, fe_solver, **params) -> torch.Tensor:
    """Evaluate a registered response, checking that it applies to this solver's physics."""
    if name not in RESPONSES:
        raise KeyError(f"Unknown response {name!r}; available: {', '.join(RESPONSES)}")
    r = RESPONSES[name]
    phys = physics_of(fe_solver)
    if phys not in r.physics:
        raise ValueError(f"Response {name} is not available for {phys} problems (valid for: {', '.join(sorted(r.physics))}).")
    return r.fn(sol, x, fe_solver, **params)
