"""Markdown table of run summaries: python compare.py run1 run2 ... (default: all runs)."""

import json
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "results" / "runs"
cols = ["sqs", "sqs_per_instance", "viou", "precision", "recall", "coverage", "point_miou"]
names = sys.argv[1:] or sorted(p.name for p in RUNS.iterdir() if (p / "metrics.json").exists())
print("| run | " + " | ".join(cols) + " | K by size | vIoU by size |")
print("|---" * (len(cols) + 3) + "|")
for n in names:
    s = json.loads((RUNS / n / "metrics.json").read_text())["summary"]
    f = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    print(f"| {n} | " + " | ".join(f(s.get(c, float("nan"))) for c in cols) +
          f" | {' / '.join(f'{v:.1f}' for v in s['K_by_size'])} | {' / '.join(f'{v:.2f}' for v in s['viou_by_size'])} |")
