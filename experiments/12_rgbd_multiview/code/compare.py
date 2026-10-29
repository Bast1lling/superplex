"""Compare runs against the 3D GT upper bound on the same instances.

    python compare.py <exp>:<run> [...]          e.g. 14:room0_f400   (12:<scene>_gt3d must exist)

Per run: mean vIoU / precision / recall over its fitted instances, and the upper bound (12:<scene>_gt3d) on exactly
those instances; `rel` = vIoU / upper bound.
"""

import json
import sys

import numpy as np

from fit import EXP_DIRS, EXPERIMENTS


def load(spec):
    exp, name = spec.split(":")
    d = EXPERIMENTS / EXP_DIRS[int(exp)] / "results" / "runs" / name
    return json.loads((d / "metrics.json").read_text()), json.loads((d / "config.json").read_text())


def row(spec):
    m, cfg = load(spec)
    ref, _ = load(f"12:{cfg['scene'].replace('_', '')}_gt3d")
    keep = [i for i, r in enumerate(m["per_instance"]) if r["fitted"] and r["gt_vox"] > 0]
    mean = lambda rows, k: np.mean([rows[i][k] for i in keep])
    v, u = mean(m["per_instance"], "iou"), mean(ref["per_instance"], "iou")
    return {"run": spec, "fitted": len(keep), "viou": v, "precision": mean(m["per_instance"], "precision"),
            "recall": mean(m["per_instance"], "recall"), "upper": u, "rel": v / u}


def main(specs):
    print("| run | instances | vIoU | prec | rec | upper bound (same instances) | rel |")
    print("|---|---|---|---|---|---|---|")
    for s in specs:
        r = row(s)
        print(f"| {r['run']} | {r['fitted']} | {r['viou']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | "
              f"{r['upper']:.3f} | {r['rel']:.2f} |")


if __name__ == "__main__":
    main(sys.argv[1:])
