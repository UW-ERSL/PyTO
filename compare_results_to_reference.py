"""Compare a benchmark sweep in Results/ with Results/Results_Reference and write a Markdown report.

Usage (from the repository root):
    python compare_results_to_reference.py                     # newest Results_<date> folder
    python compare_results_to_reference.py Results_2026-09-23  # a specific folder

Writes Results/COMPARISON_<folder>_vs_Reference.md. Re-run it after a sweep finishes to refresh the report.
Only the standard library is used.
"""
import csv
import glob
import os
import sys
import time
from collections import Counter, defaultdict

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
REFERENCE = "Results_Reference"
METHOD_ORDER = ["DENSITYMMA", "DENSITYOCM", "PARETO", "LEVELSET"]
METHOD_LABEL = {
    "DENSITYMMA": "MMA (autograd)",
    "DENSITYOCM": "OCM (autograd)",
    "PARETO": "Pareto (hand-derived)",
    "LEVELSET": "LevelSet (hand-derived)",
}
SAME_TOL = 0.01      # objectives within 1% count as the same result
ITER_TOL = 0.10      # iteration counts within 10% count as similar
MAX_ITERS = 150      # iteration cap used by the benchmark runner (TOParams.MaxIterations)
VF_TOL = 0.02        # volume fractions differing by more than this make objectives not directly comparable

# Plain-language notes for failures we understand. Keys are substrings of the error message.
KNOWN_ERRORS = {
    "coo_matrix": "Known bug: the body-force setup passes a SciPy sparse matrix to torch.tensor (hex_structural_fea.py, __init__).",
    "vf": "The optimizer stopped before reaching the target volume fraction.",
    "Volume fraction": "The optimizer stopped before reaching the target volume fraction.",
    "Maximum iterations": "Hit the iteration limit before converging.",
    "can't allocate memory": "Ran out of memory.",
    "under-constrained": "Supports leave a rigid-body mode free while the load excites it.",
}


def load(folder):
    """{(category, method, problem): row} for every *_summary.csv under Results/<folder>."""
    rows = {}
    for f in glob.glob(os.path.join(RESULTS, folder, "**", "*_summary.csv"), recursive=True):
        parts = os.path.relpath(f, os.path.join(RESULTS, folder)).split(os.sep)
        category, method = parts[0], parts[1]
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                rows[(category, method, r["name"])] = r
    return rows


def num(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def ok(row):
    return row is not None and row.get("success") == "True"


def fmt(v, digits=4):
    return "-" if v is None else f"{v:.{digits}g}"


def image(folder, category, method, problem):
    rel = os.path.join(folder, category, method, f"{problem}.png")
    return rel.replace(os.sep, "/") if os.path.exists(os.path.join(RESULTS, rel)) else None


def explain(err):
    if not err or err == "No errors.":
        return ""
    for key, text in KNOWN_ERRORS.items():
        if key in err:
            return text
    return err[:140]


def compare(ref, new, category="", method=""):
    """Return a dict describing how one AutoPyTO row differs from its Reference row."""
    out = {"verdict": "", "diff": None, "iters": "", "note": ""}
    r_ok, n_ok = ok(ref), ok(new)
    if ref is None:
        out["verdict"] = "No Reference row"
        out["note"] = "Not in the Reference sweep."
        return out
    if new is None:
        out["verdict"] = "Not run"
        if category == "ThermoStructural" and method in ("PARETO", "LEVELSET"):
            out["note"] = "Skipped by design: Pareto/LevelSet have hand-derived sensitivities for structural or thermal only."
        else:
            out["note"] = "Missing from this sweep (still running, not reached yet, or the process was killed)."
        return out
    if not r_ok and not n_ok:
        out["verdict"] = "Both failed"
    elif r_ok and not n_ok:
        out["verdict"] = "Reference only OK"
    elif n_ok and not r_ok:
        out["verdict"] = "AutoPyTO only OK"
    if not (r_ok and n_ok):
        out["note"] = explain(new.get("errorMsg")) if not n_ok else "Reference reported: " + explain(ref.get("errorMsg"))
        return out

    if category == "ThermoStructural":
        # Redefined in Plan B (B0) to match the papers (load, volume, material, temperatures), and the old
        # results used the component-major getHMatrix bug (B2): the Reference numbers describe other problems.
        out["verdict"] = "Not comparable"
        out["note"] = ("Problem redefined to match the papers and the thermal-force ordering bug fixed "
                       "(THERMOELASTIC_PLAN_B_LOG.md B0/B2); compare with Results/Publications/PyTO_comparison instead.")
        return out

    ro, no = num(ref, "objective"), num(new, "objective")
    rv, nv = num(ref, "volfrac"), num(new, "volfrac")
    out["diff"] = None if not ro else (no - ro) / abs(ro)
    if rv is not None and nv is not None and abs(rv - nv) > VF_TOL:
        out["verdict"] = "Not comparable"
        out["note"] = f"Volume fraction differs ({rv:.2f} vs {nv:.2f}), so the objectives measure different designs."
    elif out["diff"] is None:
        out["verdict"] = "Not comparable"
    elif abs(out["diff"]) <= SAME_TOL:
        out["verdict"] = "Same"
    elif out["diff"] < 0:
        out["verdict"] = "AutoPyTO better"
    else:
        out["verdict"] = "Reference better"

    ri, ni = num(ref, "#FEAs"), num(new, "#FEAs")
    # The 150-iteration cap applies to MMA and OCM only; Pareto and LevelSet count iterations differently.
    if method in ("DENSITYMMA", "DENSITYOCM") and ni is not None and ni >= MAX_ITERS:
        out["capped"] = True
        cap = f"Reached the {MAX_ITERS}-iteration limit without converging (Reference stopped after {ri:.0f})." if ri else f"Reached the {MAX_ITERS}-iteration limit."
        out["note"] = (out["note"] + " " + cap).strip()
    if ri and ni:
        if abs(ni - ri) <= ITER_TOL * ri:
            out["iters"] = "Similar"
        else:
            out["iters"] = "AutoPyTO fewer" if ni < ri else "Reference fewer"
    return out


def badge(verdict):
    return {
        "Same": "🟰 Same",
        "AutoPyTO better": "✅ AutoPyTO better",
        "Reference better": "⚠️ Reference better",
        "Reference only OK": "❌ AutoPyTO failed",
        "AutoPyTO only OK": "✅ AutoPyTO only OK",
        "Both failed": "❌ Both failed",
        "Not comparable": "❔ Not comparable",
        "Not run": "⏳ Not run",
        "No Reference row": "🆕 New",
    }.get(verdict, verdict)


def main():
    folders = sorted(d for d in os.listdir(RESULTS) if d.startswith("Results_20") and os.path.isdir(os.path.join(RESULTS, d)))
    today = sys.argv[1] if len(sys.argv) > 1 else folders[-1]
    ref, new = load(REFERENCE), load(today)
    keys = sorted(set(ref) | set(new), key=lambda k: (k[0], METHOD_ORDER.index(k[1]) if k[1] in METHOD_ORDER else 9, k[2]))
    # ThermoStructural results reuse problem names (e.g. MBBBeam) for different problems; the category key keeps them apart.
    comp = {k: compare(ref.get(k), new.get(k), k[0], k[1]) for k in keys}
    cats = sorted({k[0] for k in keys})

    L = []
    w = L.append
    w(f"# AutoPyTO vs Reference: benchmark comparison")
    w("")
    w(f"*AutoPyTO results:* `Results/{today}`  |  *Reference results:* `Results/{REFERENCE}`  |  *Generated:* {time.strftime('%Y-%m-%d %H:%M')}  ")
    w(f"*Regenerate:* `python compare_results_to_reference.py {today}`")
    w("")
    w("> **This is a snapshot.** If the sweep is still running, some rows show as **Not run**. Re-run the command above when it finishes.")
    w("")
    w("## How to read this report")
    w("")
    w("- **Reference** is the saved set of results in `Results_Reference` (generated in Feb 2026 by an earlier version of the code). **AutoPyTO** is the current code, run today.")
    w("- **Objective** is the quantity each problem minimises (for most problems, compliance: how much the structure flexes; lower means stiffer). **Lower is better.** The objective is only compared when both runs succeeded and reached about the same volume fraction.")
    w(f"- **Diff %** = (AutoPyTO - Reference) / Reference. Negative means AutoPyTO found a lower (better) objective. Differences within ±{SAME_TOL:.0%} are called **Same**.")
    w(f"- **Iterations** are the number of finite-element analyses (`#FEAs` in the CSV). Fewer means the optimizer converged faster. Counts within ±{ITER_TOL:.0%} are called **Similar**.")
    w("- **Time** is wall-clock seconds on whatever machine ran each sweep, so it is only a rough guide.")
    w("- **Error / note** shows the failure message when a run failed, or a plain-language explanation when the result is not directly comparable.")
    w("- **MMA and OCM** compute gradients with automatic differentiation (PyTorch autograd); **Pareto and LevelSet** use hand-derived sensitivities.")
    w("- Methods: MMA = `DENSITYMMA`, OCM = `DENSITYOCM`.")
    w("")

    # Scoreboard
    w("## Scoreboard")
    w("")
    w("Counts of problem/method pairs, by how the AutoPyTO result compares with the Reference.")
    w("")
    verdicts = ["AutoPyTO better", "Same", "Reference better", "Not comparable", "Reference only OK", "AutoPyTO only OK", "Both failed", "Not run", "No Reference row"]
    w("| Method | " + " | ".join(badge(v) for v in verdicts) + " |")
    w("|---|" + "---|" * len(verdicts))
    for m in METHOD_ORDER + ["ALL"]:
        c = Counter(comp[k]["verdict"] for k in keys if m == "ALL" or k[1] == m)
        w(f"| {'**All methods**' if m == 'ALL' else METHOD_LABEL[m]} | " + " | ".join(str(c.get(v, 0)) or "0" for v in verdicts) + " |")
    w("")
    it = Counter(comp[k]["iters"] for k in keys if comp[k]["iters"])
    w(f"**Iterations (where both succeeded):** AutoPyTO used fewer in **{it.get('AutoPyTO fewer', 0)}**, "
      f"similar in **{it.get('Similar', 0)}**, more in **{it.get('Reference fewer', 0)}** cases.")
    w("")

    capped = [k for k in keys if comp[k].get("capped")]
    both = [k for k in keys if ok(ref.get(k)) and ok(new.get(k))]
    n_good = sum(comp[k]["verdict"] in ("Same", "AutoPyTO better") for k in both)
    w("## Key findings")
    w("")
    w(f"- **Objective quality:** of {len(both)} problem/method pairs that succeeded in both sweeps, AutoPyTO matched or beat the Reference in **{n_good}**.")
    mma = [k for k in both if k[1] == "DENSITYMMA" and comp[k]["iters"]]
    more = [k for k in mma if comp[k]["iters"] == "Reference fewer"]
    w(f"- **Iterations:** MMA needed noticeably more iterations than the Reference in **{len(more)} of {len(mma)}** cases. OCM, Pareto and LevelSet used the same number as the Reference.")
    if capped:
        w(f"- **Did not converge:** **{len(capped)}** runs hit the {MAX_ITERS}-iteration limit (listed below). Their objectives are worse than, or not comparable to, the Reference.")
    w("")

    # Needs attention
    fails = [k for k in keys if new.get(k) is not None and not ok(new[k])]
    worse = sorted([k for k in keys if comp[k]["verdict"] == "Reference better"], key=lambda k: -(comp[k]["diff"] or 0))
    noncomp = [k for k in keys if comp[k]["verdict"] == "Not comparable"]
    w("## Needs attention")
    w("")
    if fails:
        w("### Failed today")
        w("")
        w("| Category | Method | Problem | Reference | AutoPyTO error |")
        w("|---|---|---|---|---|")
        for k in fails:
            r = ref.get(k)
            w(f"| {k[0]} | {METHOD_LABEL.get(k[1], k[1])} | {k[2]} | {'no Reference row' if r is None else ('OK' if ok(r) else 'failed')} | {explain(new[k].get('errorMsg')) or comp[k]['note'] or '-'} |")
        w("")
    if capped:
        w("### Stopped at the iteration limit (did not converge)")
        w("")
        w("| Category | Method | Problem | Reference objective | AutoPyTO objective | Reference iters | AutoPyTO iters |")
        w("|---|---|---|---|---|---|---|")
        for k in capped:
            w(f"| {k[0]} | {METHOD_LABEL.get(k[1], k[1])} | {k[2]} | {fmt(num(ref.get(k), 'objective')) if ok(ref.get(k)) else '-'} | {fmt(num(new[k], 'objective'))} | {fmt(num(ref.get(k), '#FEAs'), 3) if ref.get(k) else '-'} | {fmt(num(new[k], '#FEAs'), 3)} |")
        w("")
    if worse:
        w("### Reference is better by more than 1%")
        w("")
        w("| Category | Method | Problem | Reference | AutoPyTO | Diff % |")
        w("|---|---|---|---|---|---|")
        for k in worse:
            w(f"| {k[0]} | {METHOD_LABEL.get(k[1], k[1])} | {k[2]} | {fmt(num(ref[k], 'objective'))} | {fmt(num(new[k], 'objective'))} | {comp[k]['diff']:+.1%} |")
        w("")
    if noncomp:
        w("### Not directly comparable")
        w("")
        w("| Category | Method | Problem | Why |")
        w("|---|---|---|---|")
        for k in noncomp:
            w(f"| {k[0]} | {METHOD_LABEL.get(k[1], k[1])} | {k[2]} | {comp[k]['note'] or '-'} |")
        w("")

    # Detailed tables
    w("## Detailed comparison")
    w("")
    w("Every section has a table of numbers, followed by a collapsible gallery of the saved topology images (Reference on the left, AutoPyTO on the right).")
    w("")
    for cat in cats:
        w(f"### {cat}")
        w("")
        for m in METHOD_ORDER:
            ks = [k for k in keys if k[0] == cat and k[1] == m]
            if not ks:
                continue
            w(f"#### {METHOD_LABEL[m]}")
            w("")
            w("| Problem | Ref objective | AutoPyTO objective | Diff % | Better result | Ref iters | AutoPyTO iters | Fewer iterations | Ref vf | AutoPyTO vf | Ref time (s) | AutoPyTO time (s) | Error / note |")
            w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for k in ks:
                r, n, c = ref.get(k), new.get(k), comp[k]
                d = "-" if c["diff"] is None else f"{c['diff']:+.1%}"
                note = c["note"]
                if n is not None and not ok(n) and not note:
                    note = n.get("errorMsg", "")[:140]
                w(f"| {k[2]} | {fmt(num(r, 'objective')) if ok(r) else '-'} | {fmt(num(n, 'objective')) if ok(n) else '-'} | {d} | {badge(c['verdict'])} | "
                  f"{fmt(num(r, '#FEAs'), 3) if ok(r) else '-'} | {fmt(num(n, '#FEAs'), 3) if ok(n) else '-'} | {c['iters'] or '-'} | "
                  f"{fmt(num(r, 'volfrac'), 3) if ok(r) else '-'} | {fmt(num(n, 'volfrac'), 3) if ok(n) else '-'} | "
                  f"{fmt(num(r, 'time (s)'), 3) if ok(r) else '-'} | {fmt(num(n, 'time (s)'), 3) if ok(n) else '-'} | {note or '-'} |")
            w("")
            pics = [(k[2], image(REFERENCE, k[0], k[1], k[2]), image(today, k[0], k[1], k[2])) for k in ks]
            pics = [p for p in pics if p[1] or p[2]]
            if pics:
                w("<details><summary>Topology images</summary>")
                w("")
                w("<table>")
                w("<tr><th>Problem</th><th>Reference</th><th>AutoPyTO</th></tr>")
                for name, a, b in pics:
                    ca = f'<img src="{a}" width="260">' if a else "<i>no image</i>"
                    cb = f'<img src="{b}" width="260">' if b else "<i>no image</i>"
                    w(f"<tr><td>{name}</td><td>{ca}</td><td>{cb}</td></tr>")
                w("</table>")
                w("")
                w("</details>")
                w("")

    out = os.path.join(RESULTS, f"COMPARISON_{today}_vs_Reference.md")
    with open(out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"wrote {out}  ({len(keys)} problem/method pairs)")


if __name__ == "__main__":
    main()
