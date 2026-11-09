"""Final metrics of finished runs side by side (results/<name>.json, test set = evaluation frames).

    uv run python compare.py [prefix]        (default f16_) -> markdown table on stdout
"""

import json
import sys
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"
COLS = [("recall@0.5", "proposal recall@0.5", ("recall@0.5", "enc_recall@0.5")),
        ("best_iou", "proposal best IoU", ("best_iou", "enc_best_iou")),
        ("dec_recall@0.5", "dec. recall@0.5", ("dec_recall@0.5",)),
        ("pair_iou", "pair IoU", ("pair_iou",)),
        ("found", "found 0.5", ("found",)), ("found@0.3", "found 0.3", ("found@0.3",)),
        ("predicted", "predicted 0.5", ("predicted",)), ("occ_iou@0.3", "occ. IoU 0.3", ("occ_iou@0.3",)),
        ("occ", "occ. IoU 0.5", ("occ_iou",)), ("mask_cov", "mask cov.", ("mask_cov",))]


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "f16_"
    rows = []
    for p in sorted(RES.glob(f"{prefix}*.json"), key=lambda p: p.stat().st_mtime):
        d = json.loads(p.read_text())
        if "test" not in d:
            continue
        t, a = d["test"], d["args"]
        kind = "topk" if a.get("topk") else "dense" if a.get("dense") else "patch"
        cells = []
        for _, _, keys in COLS:
            k = next((k for k in keys if k in t), None)
            cells.append(f"{t[k]:.3f}" if k and isinstance(t[k], float) and k != "predicted" else (f"{t[k]:.0f}" if k else "–"))
        rows.append(f"| {p.stem} | {kind} | " + " | ".join(cells) + f" | {t.get('visible', t.get('targets', 0)):.0f} |")
    print("| run | model | " + " | ".join(c[1] for c in COLS) + " | visible |")
    print("|---|---|" + "---|" * len(COLS) + "---|")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
