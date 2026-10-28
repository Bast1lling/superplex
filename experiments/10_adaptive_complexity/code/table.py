"""Markdown summary table of runs (as in the READMEs of 09 / 10).

    python table.py <run> [<run> ...]       run as in compare.py; per-class K columns for the classes in CLASSES
"""

import sys

import numpy as np

from compare import load

CLASSES = ("indoor-plant", "table", "chair", "lamp", "wall", "rug", "floor")


def main(runs):
    print("| run | SQs | vIoU | prec | rec | plausible | over | " + " | ".join(f"K {c}" for c in CLASSES) + " |")
    print("|---" * (7 + len(CLASSES)) + "|")
    for r in runs:
        s, p = load(r)
        ks = []
        for c in CLASSES:
            k = [K for n, K in zip(p["name"], p["K"]) if n == c]
            ks.append(f"{np.mean(k):.1f}" if k else "–")
        extra = f" (purity {s['purity_mean']:.2f}, no SQ {s['instances_without_sq']:.2f})" if "purity_mean" in s else ""
        print(f"| {r}{extra} | {s['sqs']} | {s['viou']:.3f} | {s['precision']:.3f} | {s['recall']:.3f} | "
              f"{s['plausible']:.2f} | {s['over']:.2f} | " + " | ".join(ks) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
