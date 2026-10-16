"""Print exp0 tables as Markdown. Usage: python -m experiments.summarize_exp0 results/exp0

Cells show the mean over seeds, with [min, max] across seeds where the spread matters.
"""

import json
import math
import sys
from pathlib import Path


def finite(xs):
    return [x for x in xs if not math.isnan(x)]


def mean(xs):
    xs = finite(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def cell(xs, spread: bool) -> str:
    xs = finite(xs)
    if not xs:
        return "–"
    s = f"{mean(xs):.2f}"
    return f"{s} [{min(xs):.2f}, {max(xs):.2f}]" if spread and len(xs) > 1 else s


def table(runs, names, title, get, spread=True, note=""):
    print(f"\n## {title}\n")
    if note:
        print(note + "\n")
    print("| noise at r_j, scale | acc | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 2) + "|")
    for i, row in enumerate(runs[0]["interventions"]):
        rows = [r["interventions"][i] for r in runs]
        acc = mean([x["acc"] for x in rows])
        cells = [cell([get(x["reports"][n]) for x in rows], spread) for n in names]
        print(f"| r_{row['layer']}, {row['scale']} | {acc:.2f} | " + " | ".join(cells) + " |")


def main() -> None:
    runs = [json.loads(p.read_text()) for p in sorted(Path(sys.argv[1]).glob("seed*.json"))]
    names = list(runs[0]["clean"])
    print(f"Seeds: {[r['seed'] for r in runs]}; clean acc: {[round(r['clean_acc'], 3) for r in runs]}")
    print(f"Environment: {runs[0].get('env', 'not recorded')}")

    print("\n## Clean (mean [min, max] over seeds)\n\n| report | AUROC | ECE |\n|---|---|---|")
    for n in names:
        print(f"| {n} | {cell([r['clean'][n]['auroc'] for r in runs], True)} "
              f"| {cell([r['clean'][n]['ece'] for r in runs], True)} |")

    table(runs, names, "Intervened: sensitivity (as pre-registered)", lambda x: x["sensitivity"])
    table(runs, names, "Intervened: AUROC", lambda x: x["auroc"], spread=False)
    table(runs, names, "Intervened: ECE", lambda x: x["ece"], spread=False)

    print("\n# Post-hoc analyses (added after the independent review; not pre-registered)")
    flips = runs[0]["interventions"]
    print("\n## Flip composition (seed means)\n\n| noise at r_j, scale | correct→wrong | wrong→correct |\n|---|---|---|")
    for i, row in enumerate(flips):
        d = [r["interventions"][i]["reports"][names[0]]["directional"] for r in runs]
        print(f"| r_{row['layer']}, {row['scale']} | {mean([x['n_down'] for x in d]):.0f} "
              f"| {mean([x['n_up'] for x in d]):.0f} |")
    table(runs, names, "Balanced directional sensitivity", lambda x: x["directional"]["balanced"],
          note="Mean of P(report falls | correct→wrong) and P(report rises | wrong→correct). "
               "Null: 0.5 for a report that always falls or moves symmetrically at random; 0 if unchanged.")
    table(runs, names, "Rises on wrong→correct flips", lambda x: x["directional"]["up"])
    table(runs, names, "Spearman ρ(Δreport, Δ log p(target))", lambda x: x["rho_dlogp"],
          note="Over all eval inputs. 0 for a report that does not change.")


if __name__ == "__main__":
    main()
