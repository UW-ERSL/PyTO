"""Safe expression compiler for user-defined objectives and constraints (plan S2).

A user writes a quantity as a formula over the built-in responses (pyto.autodiff.qoi.responses), e.g.

    Displacement(Tip, y, mean) / 1e-3 + 0.1 * VolumeFraction()
    StressFailureFactor() * 2 + sqrt(Compliance() / 50)
    smooth_max(Temperature(HotFace, pnorm) / 500, Displacement(Tip, magnitude, pnorm) / 1e-3)

compile_expression() parses it with Python's `ast` module against a whitelist and builds a torch closure
fn(sol, x, fe_solver) that autograd differentiates through the FE solve. Nothing is ever passed to eval/exec: names are
looked up in fixed tables, and anything outside the whitelist (attribute access, subscripts, lambdas, imports,
comprehensions, strings, ...) is rejected with a message pointing at the offending part. That makes expressions safe
to store in project files.

Whitelist:
  numbers; + - * / ** ; unary + -; parentheses
  response calls: any name in RESPONSES, with positional or keyword parameters (see POSITIONAL)
  functions: sqrt, exp, log, smooth_max(a, b, ..., p=8)  (abs/min/max are rejected: not differentiable everywhere)
  constants: YieldStrength, YoungsModulus, Density (taken from the FE solver's material), and user constants
  inside response calls, bare words are selection names (from `selections`, plus All) or option values (x, y, mean, ...)
"""
import ast
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from pyto.autodiff.qoi.responses import RESPONSES, physics_of

# Positional parameter order of each response (keywords are accepted too; `selection` and `aggregate` are aliases).
POSITIONAL = {
    "Displacement": ("nodes", "component", "aggregate_by", "p"),
    "Temperature": ("nodes", "aggregate_by", "p"),
    "ReactionForce": ("nodes", "component"),
    "VolumeFraction": ("elements",),
    "StrainEnergy": ("elements",),
}
_ALIASES = {"selection": None, "aggregate": "aggregate_by"}      # None: resolved to nodes/elements per response
_SELECTION_PARAMS = ("nodes", "elements")
_OPTION_WORDS = {"x", "y", "z", "magnitude", "mean", "sum", "pnorm"}
_MATERIAL_CONSTANTS = {"YieldStrength": "yield_strength", "YoungsModulus": "youngs_modulus", "Density": "mass_density"}


class ExpressionError(ValueError):
    """An expression is not valid; the message names the problem and where it is."""


def _smooth_max(values, p=8.0):
    """Smooth maximum of several (positive) quantities: p-norm, an upper bound of the true max."""
    v = torch.stack([torch.as_tensor(t, dtype=torch.float64).reshape(()) for t in values])
    scale = v.detach().abs().max().clamp_min(1e-300)
    return scale * ((v.abs() / scale) ** p).sum() ** (1.0 / p)


_FUNCTIONS = {"sqrt": torch.sqrt, "exp": torch.exp, "log": torch.log}
_REJECTED_FUNCTIONS = {
    "abs": "abs() is not differentiable at 0; use Displacement(..., magnitude, ...) or a squared quantity",
    "max": "max() is not differentiable; use smooth_max(a, b, ...) or a pnorm aggregate",
    "min": "min() is not differentiable; rewrite as a constraint, or use -smooth_max(-a, -b)",
}
_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
           ast.Div: lambda a, b: a / b, ast.Pow: lambda a, b: a ** b}


@dataclass
class CompiledExpression:
    text: str
    fn: object                                   # fn(sol, x, fe_solver) -> torch scalar
    responses: set = field(default_factory=set)  # response names used

    @property
    def depends_on_solution(self) -> bool:
        return any(RESPONSES[r].depends_on_solution for r in self.responses)

    @property
    def physics(self) -> set:
        """Physics for which every response in the expression is available."""
        out = {"structural", "thermal", "thermo-structural"}
        for r in self.responses:
            out &= set(RESPONSES[r].physics)
        return out

    @property
    def is_plain_compliance(self) -> bool:
        return self.text.replace(" ", "") == "Compliance()"

    def __call__(self, sol, x, fe_solver):
        return self.fn(sol, x, fe_solver)


def compile_expression(text: str, selections: dict = None, constants: dict = None) -> CompiledExpression:
    """Compile a formula into a differentiable function fn(sol, x, fe_solver).

    selections: {name: {"nodes": index array, "elements": index array}} (either key may be missing).
    constants:  {name: float} extra named constants.
    """
    if not isinstance(text, str) or not text.strip():
        raise ExpressionError("The expression is empty.")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as e:
        raise ExpressionError(f"Syntax error in {text!r}: {e.msg} (column {e.offset}).") from None
    selections = dict(selections or {})
    constants = dict(constants or {})
    used = set()

    def where(node):
        seg = ast.get_source_segment(text.strip(), node)
        return f"'{seg}'" if seg else "here"

    def build(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ExpressionError(f"Only numbers are allowed as literals, not {where(node)}.")
            val = float(node.value)
            return lambda sol, x, fe: val
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            op, left, right = _BINOPS[type(node.op)], build(node.left), build(node.right)
            return lambda sol, x, fe: op(left(sol, x, fe), right(sol, x, fe))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            inner, sign = build(node.operand), (-1.0 if isinstance(node.op, ast.USub) else 1.0)
            return lambda sol, x, fe: sign * inner(sol, x, fe)
        if isinstance(node, ast.Name):
            if node.id in constants:
                val = float(constants[node.id])
                return lambda sol, x, fe: val
            if node.id in _MATERIAL_CONSTANTS:
                attr = _MATERIAL_CONSTANTS[node.id]
                return lambda sol, x, fe: float(getattr(fe.mat_prop, attr))
            if node.id in RESPONSES:
                raise ExpressionError(f"{node.id} is a response: call it, e.g. {node.id}().")
            raise ExpressionError(f"Unknown name {where(node)}. Known constants: "
                                  f"{', '.join(sorted(set(_MATERIAL_CONSTANTS) | set(constants))) or 'none'}.")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ExpressionError(f"Only plain function names can be called, not {where(node.func)}.")
            name = node.func.id
            if name in RESPONSES:
                return build_response(node, name)
            if name in _FUNCTIONS:
                if len(node.args) != 1 or node.keywords:
                    raise ExpressionError(f"{name}() takes exactly one argument.")
                f, arg = _FUNCTIONS[name], build(node.args[0])
                return lambda sol, x, fe: f(torch.as_tensor(arg(sol, x, fe), dtype=torch.float64))
            if name == "smooth_max":
                if len(node.args) < 2:
                    raise ExpressionError("smooth_max() needs at least two arguments.")
                p = 8.0
                for kw in node.keywords:
                    if kw.arg != "p" or not isinstance(kw.value, ast.Constant):
                        raise ExpressionError("smooth_max() only takes the keyword p=<number>.")
                    p = float(kw.value.value)
                parts = [build(a) for a in node.args]
                return lambda sol, x, fe: _smooth_max([f(sol, x, fe) for f in parts], p)
            if name in _REJECTED_FUNCTIONS:
                raise ExpressionError(_REJECTED_FUNCTIONS[name] + ".")
            raise ExpressionError(f"Unknown function {name}(). Responses: {', '.join(RESPONSES)}; "
                                  f"functions: {', '.join([*_FUNCTIONS, 'smooth_max'])}.")
        raise ExpressionError(f"{type(node).__name__} is not allowed in an expression ({where(node)}).")

    def param_value(node, pname):
        """Literal parameter of a response call: selection name, option word or number."""
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            return -float(node.operand.value)
        if isinstance(node, ast.Name):
            return node.id
        raise ExpressionError(f"Parameter {pname} must be a name or a number, not {where(node)}.")

    def build_response(node, name):
        used.add(name)
        order = POSITIONAL.get(name, ())
        if len(node.args) > len(order):
            raise ExpressionError(f"{name}() takes at most {len(order)} positional parameters "
                                  f"({', '.join(order) or 'none'}).")
        raw = {pname: param_value(a, pname) for pname, a in zip(order, node.args)}
        for kw in node.keywords:
            key = kw.arg
            if key in _ALIASES:
                key = _ALIASES[key] or next((p for p in order if p in _SELECTION_PARAMS), None)
            if key not in order:
                raise ExpressionError(f"{name}() has no parameter {kw.arg!r} (parameters: {', '.join(order) or 'none'}).")
            raw[key] = param_value(kw.value, key)
        params = {}
        allowed = RESPONSES[name].params
        for key, val in raw.items():
            if key in _SELECTION_PARAMS:
                if val == "All":
                    params[key] = None
                elif val not in selections:
                    raise ExpressionError(f"Unknown selection {val!r} in {name}(). Defined selections: "
                                          f"{', '.join(['All', *selections])}.")
                elif selections[val].get(key) is None:
                    kind = "node" if key == "nodes" else "element"
                    raise ExpressionError(f"Selection {val!r} has no {kind}s, which {name}() needs.")
                else:
                    params[key] = np.asarray(selections[val][key])
            elif key == "p":
                if not isinstance(val, float) or val <= 1:
                    raise ExpressionError(f"p must be a number > 1 in {name}().")
                params[key] = val
            else:
                options = allowed.get(key)
                if isinstance(options, tuple) and val not in options:
                    raise ExpressionError(f"{name}() parameter {key} must be one of {', '.join(options)}, got {val!r}.")
                params[key] = val
        fn = RESPONSES[name].fn
        return lambda sol, x, fe: fn(sol, x, fe, **params)

    root = build(tree.body)

    def fn(sol, x, fe_solver):
        phys = physics_of(fe_solver)
        missing = [r for r in used if phys not in RESPONSES[r].physics]
        if missing:
            raise ExpressionError(f"{', '.join(sorted(missing))} not available for {phys} problems.")
        out = root(sol, x, fe_solver)
        return torch.as_tensor(out, dtype=sol.dtype).reshape(())

    fn.__name__ = f"expression[{text.strip()}]"
    return CompiledExpression(text=text.strip(), fn=fn, responses=used)
