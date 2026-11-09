"""Table of runs: python compare.py run1 run2 ... [--keys k1,k2] (default: all results/*.json). test + train_eval."""
import json
import sys
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"
DEFAULT = ["pair_iou", "pair_iou_v0", "pair_iou_v0_seen", "pair_iou_v0_partial", "pair_iou_v1only", "occ_iou", "occ_iou@0.3",
           "found", "found@0.3", "dec_recall@0.5", "enc_recall@0.5", "mask_cov", "r_inst_acc", "r_depth_absrel",
           "depth_absrel", "viou", "viou@0.3", "viou_target", "predicted", "targets"]
args = [a for a in sys.argv[1:] if not a.startswith("--")]
keys = DEFAULT
for a in sys.argv[1:]:
    if a.startswith("--keys="):
        keys = a[7:].split(",")
runs = args or sorted(p.stem for p in RES.glob("*.json") if "per_frame" in json.loads(p.read_text()))
rows = []
for r in runs:
    d = json.loads((RES / f"{r}.json").read_text())
    for split in ("test", "train_eval"):
        if d.get(split):
            rows.append((r if split == "test" else f"  ({r} train)", d[split]))
ks = [k for k in keys if any(k in m for _, m in rows)]
print("| run | " + " | ".join(ks) + " |")
print("|---" * (len(ks) + 1) + "|")
for n, m in rows:
    print(f"| {n} | " + " | ".join(f"{m[k]:.3f}" if k in m and abs(m[k]) < 10 else (f"{m[k]:.0f}" if k in m else "–") for k in ks) + " |")
