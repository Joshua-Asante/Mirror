"""Print exp0 tables (mean over seeds) as Markdown. Usage: python -m experiments.summarize_exp0 results/exp0"""

import json
import math
import sys
from pathlib import Path


def mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    runs = [json.loads(p.read_text()) for p in sorted(Path(sys.argv[1]).glob("seed*.json"))]
    names = list(runs[0]["clean"])
    print(f"Seeds: {[r['seed'] for r in runs]}; clean acc: {[round(r['clean_acc'], 3) for r in runs]}\n")

    print("## Clean\n\n| report | AUROC | ECE |\n|---|---|---|")
    for n in names:
        print(f"| {n} | {mean([r['clean'][n]['auroc'] for r in runs]):.3f} "
              f"| {mean([r['clean'][n]['ece'] for r in runs]):.3f} |")

    for metric in ("sensitivity", "auroc", "ece"):
        print(f"\n## Intervened: {metric}\n")
        print("| noise at r_j, scale | acc | flips | " + " | ".join(names) + " |")
        print("|---" * (len(names) + 3) + "|")
        for i, row in enumerate(runs[0]["interventions"]):
            rows = [r["interventions"][i] for r in runs]
            acc = mean([x["acc"] for x in rows])
            flips = mean([x["reports"][names[0]]["n_flip"] for x in rows])
            cells = [f"{mean([x['reports'][n][metric] for x in rows]):.2f}" for n in names]
            print(f"| r_{row['layer']}, {row['scale']} | {acc:.2f} | {flips:.0f} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
