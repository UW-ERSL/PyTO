"""Automatic differentiation vs analytical sensitivities, side by side, for MMA and OC.

For every benchmark problem of the sweep (topopt_run_benchmarks) where analytical sensitivities apply, run
MMA with GRADIENT_METHOD.AUTODIFF and then with GRADIENT_METHOD.ANALYTICAL; after all MMA problems, do the same
with OC on the problems OC can solve. Pareto and LevelSet are not run.

Fair comparison:
  * every run is a fresh Python process (like restarting the kernel): no warm caches, no leftover memory;
  * both runs of a pair use the same problem, solver (the benchmark's choice: PARDISO, or DPCG above
    DIRECT_SOLVER_DOF_CUTOFF; or --solver), iteration limit and settings; only to_params.Gradient differs;
  * the timer covers the optimizer call only (not meshing / set-up); time per FE solve = time / #FEAs.

Run from the repository root:
    PYTHONPATH=src python -m pyto.examples_benchmarks.gradient_comparison                     # everything (hours)
    PYTHONPATH=src python -m pyto.examples_benchmarks.gradient_comparison --max-iters 20     # quick timing
    PYTHONPATH=src python -m pyto.examples_benchmarks.gradient_comparison --problems MBBBeam LBracketTopLoad_Stress_Vol
Writes Results/GradientComparison_<date>/results.csv and summary.md.
"""
import argparse
import contextlib
import csv
import datetime
import io
import multiprocessing as mp
import os
import platform
import time

ENUMS = ("StructuralTOExamples", "ThermalTOExamples", "ThermoStructuralTOExamples")


def _sweep_problems():
    """(enum name, member name) of every problem in the benchmark sweep, in sweep order."""
    import pyto.examples_benchmarks.topopt_run_benchmarks as R
    lists = [v for k, v in vars(R).items() if k.startswith("benchmarks_") and k != "benchmarks_not_in_sweep"]
    out = []
    for problems in lists:
        for p in problems:
            key = (type(p).__name__, p.name)
            if key not in out:
                out.append(key)
    return out


def _problem(enum_name, name):
    import pyto.examples_benchmarks.topopt_run_benchmarks as R
    return getattr(getattr(R, enum_name), name)


def _build(enum_name, name, solver_name):
    from pyto.examples_benchmarks.topopt_run_benchmarks import build_benchmark
    from pyto.solve.solvers import Solvers
    with contextlib.redirect_stdout(io.StringIO()):
        return build_benchmark(_problem(enum_name, name), solver=getattr(Solvers, solver_name) if solver_name else None)


def _check(enum_name, name, solver_name):
    """Which of MMA/OC can run this problem with analytical sensitivities (worker process)."""
    from pyto.autodiff.qoi.responses import physics_of
    from pyto.topopt.common import TO_QOI
    from pyto.topopt.manual_sensitivities import manual_support
    fe, to_params, _ = _build(enum_name, name, solver_name)
    ok, why = manual_support(to_params, fe)
    classical = (to_params.Objective[0] == TO_QOI.COMPLIANCE and len(to_params.Constraints) == 1
                 and to_params.Constraints[0][0] == TO_QOI.VOLUME_FRACTION)
    oc_ok = ok and classical and physics_of(fe) != "thermo-structural"   # OC 2-cycles on thermal expansion loads
    return {"analytical": ok, "reason": why, "oc": oc_ok, "ndof": int(fe.bc.num_dofs), "solver": fe.solver.name}


def _run(enum_name, name, method, gradient, max_iters, solver_name):
    """One optimization (worker process): timings and results."""
    from pyto.topopt.common import GRADIENT_METHOD
    from pyto.topopt.drivers.mma import topopt_mma
    from pyto.topopt.drivers.oc import topopt_optimality_criteria
    t0 = time.perf_counter()
    fe, to_params, _ = _build(enum_name, name, solver_name)
    build_s = time.perf_counter() - t0
    to_params.Gradient = GRADIENT_METHOD[gradient]
    iters = max_iters or to_params.MaxIterations
    binarize = not to_params.HeavisideProjection                 # as the benchmark sweep does
    with contextlib.redirect_stdout(io.StringIO()):
        t0 = time.perf_counter()
        if method == "MMA":
            u, history, success, message, n_fea = topopt_mma(fe, to_params=to_params, maxMMAIterations=iters,
                                                             binarize_topology=binarize, print_progress=False)
        else:
            u, history, success, message, n_fea = topopt_optimality_criteria(
                fe, to_params=to_params, maxIterations=iters, binarize_topology=binarize, print_progress=False)
        opt_s = time.perf_counter() - t0
    return {"problem": name, "method": method, "gradient": gradient, "solver": fe.solver.name,
            "ndof": int(fe.bc.num_dofs), "iterations": int(n_fea), "time_s": opt_s, "s_per_iter": opt_s / max(n_fea, 1),
            "build_s": build_s, "objective": float(history["objective"][-1]),
            "volfrac": float(history["volfrac"][-1]), "success": bool(success), "message": message}


def _isolated(fn, *args):
    """Run fn(*args) in a fresh spawned process and return its result (exceptions come back as dicts)."""
    with mp.get_context("spawn").Pool(1, maxtasksperchild=1) as pool:
        try:
            return pool.apply(fn, args)
        except Exception as e:  # report and continue with the next case
            return {"error": f"{type(e).__name__}: {e}"}


def _pairs_table(rows, method):
    lines = [f"### {method}", "",
             "| Problem | dofs | Solver | FEAs AD / AN | Time AD (s) | Time AN (s) | s/iter AD | s/iter AN | Speed-up AD/AN | "
             "Objective AD | Objective AN | Rel. diff |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tot = {"AUTODIFF": 0.0, "ANALYTICAL": 0.0}
    for problem in dict.fromkeys(r["problem"] for r in rows if r["method"] == method):
        pair = {r["gradient"]: r for r in rows if r["method"] == method and r["problem"] == problem}
        ad, an = pair.get("AUTODIFF"), pair.get("ANALYTICAL")
        if not ad or not an or "error" in ad or "error" in an:
            err = (ad or {}).get("error") or (an or {}).get("error") or "missing run"
            lines.append(f"| {problem} | | | | | | | | | | | error: {err} |")
            continue
        tot["AUTODIFF"] += ad["time_s"]
        tot["ANALYTICAL"] += an["time_s"]
        rel = abs(an["objective"] - ad["objective"]) / max(abs(ad["objective"]), 1e-300)
        lines.append(f"| {problem} | {ad['ndof']} | {ad['solver']} | {ad['iterations']} / {an['iterations']} | "
                     f"{ad['time_s']:.1f} | {an['time_s']:.1f} | {ad['s_per_iter']:.3f} | {an['s_per_iter']:.3f} | "
                     f"{ad['s_per_iter'] / an['s_per_iter']:.2f} | {ad['objective']:.6g} | {an['objective']:.6g} | "
                     f"{rel:.1e} |")
    if tot["ANALYTICAL"] > 0:
        lines.append(f"| **Total** | | | | **{tot['AUTODIFF']:.1f}** | **{tot['ANALYTICAL']:.1f}** | | | "
                     f"**{tot['AUTODIFF'] / tot['ANALYTICAL']:.2f}** | | | |")
    return lines + [""]


def _write(out_dir, rows, skipped, args):
    os.makedirs(out_dir, exist_ok=True)
    keys = ["problem", "method", "gradient", "solver", "ndof", "iterations", "time_s", "s_per_iter", "build_s",
            "objective", "volfrac", "success", "message", "error"]
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    import torch
    md = ["# Automatic differentiation vs analytical sensitivities", "",
          f"*Date:* {datetime.datetime.now():%Y-%m-%d %H:%M}  |  *Machine:* {platform.node()}, {os.cpu_count()} CPUs, "
          f"torch threads {torch.get_num_threads()}  |  *Iterations:* {args.max_iters or 'each problem’s MaxIterations'}"
          f"  |  *Solver:* {args.solver or 'benchmark default'}", "",
          "Each run is a separate process; both runs of a pair use the same solver and settings. Time = optimizer "
          "call only. **Speed-up AD/AN** = seconds per iteration with AD ÷ with analytical: above 1 means "
          "analytical is faster. **Rel. diff** compares the final objectives (the two gradients agree to about "
          "1e-12, so differences come only from round-off accumulated over the iterations).", ""]
    for method in ("MMA", "OC"):
        if any(r["method"] == method for r in rows):
            md += _pairs_table(rows, method)
    if skipped:
        md += ["### Not compared (analytical sensitivities not available)", "", "| Problem | Reason |", "|---|---|"]
        md += [f"| {p} | {why} |" for p, why in skipped] + [""]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(md))
    return os.path.join(out_dir, "summary.md")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--problems", nargs="*", help="problem names, or Enum.Name (default: the whole sweep)")
    ap.add_argument("--max-iters", type=int, default=None, help="iteration limit (default: each problem's)")
    ap.add_argument("--solver", choices=["PARDISO", "SPSOLVE", "DPCG"], default=None,
                    help="linear solver for both modes (default: the benchmark's choice)")
    ap.add_argument("--methods", nargs="*", default=["MMA", "OC"], choices=["MMA", "OC"])
    ap.add_argument("--out", default=None, help="output folder (default Results/GradientComparison_<date>)")
    args = ap.parse_args(argv)

    problems = _sweep_problems()
    if args.problems:                        # names, or Enum.Name for names used twice (MBBBeam)
        chosen = []
        for n in args.problems:
            if "." in n:
                enum_name, member = n.split(".", 1)
                match = [(enum_name, member)] if enum_name in ENUMS and hasattr(_problem_enum(enum_name), member) else []
            else:
                match = [p for p in problems if p[1] == n] or [(e, n) for e in ENUMS if hasattr(_problem_enum(e), n)]
            if not match:
                raise SystemExit(f"Unknown problem: {n}")
            chosen.append(match[0])
        problems = chosen
    out_dir = args.out or os.path.join("Results", f"GradientComparison_{datetime.date.today()}")

    rows, skipped, eligible = [], [], {}
    for enum_name, name in problems:
        info = _isolated(_check, enum_name, name, args.solver)
        if "error" in info:
            skipped.append((name, info["error"]))
        elif not info["analytical"]:
            skipped.append((name, info["reason"]))
        else:
            eligible[(enum_name, name)] = info
        print(f"check {name}: {'analytical OK' if (enum_name, name) in eligible else 'skipped'}"
              f"{' (OC too)' if eligible.get((enum_name, name), {}).get('oc') else ''}", flush=True)

    for method in args.methods:
        for (enum_name, name), info in eligible.items():
            if method == "OC" and not info["oc"]:
                continue
            for gradient in ("AUTODIFF", "ANALYTICAL"):
                r = _isolated(_run, enum_name, name, method, gradient, args.max_iters, args.solver)
                r.setdefault("problem", name)
                r.setdefault("method", method)
                r.setdefault("gradient", gradient)
                rows.append(r)
                print(f"{method:3s} {gradient:10s} {name:40s} " + (f"error {r['error']}" if "error" in r else
                      f"{r['iterations']:4d} FEAs {r['time_s']:8.1f} s  {r['s_per_iter']:.3f} s/it  "
                      f"J={r['objective']:.6g}"), flush=True)
            _write(out_dir, rows, skipped, args)          # keep the report current during long runs
    path = _write(out_dir, rows, skipped, args)
    print(f"wrote {path}")
    return rows


def _problem_enum(enum_name):
    import pyto.examples_benchmarks.topopt_run_benchmarks as R
    return getattr(R, enum_name)


if __name__ == "__main__":
    main()
