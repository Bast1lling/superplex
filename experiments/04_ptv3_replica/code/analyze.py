"""How well do PTv3 tokens / features respect Replica instances? Sonata (pretrained) vs. PTv3 from scratch (random init).

Per stage (finest 0 -> coarsest 4):
    purity     share of input points whose token's majority instance is their own instance
    coverage   share of GT instances that are the majority instance of at least one token
               (an instance without such a token cannot get a query initialized on it)
Per-point features ("full", Sonata's recommended concat of the 3 coarsest stages):
    AUC        nearby point pairs (< 0.5 m): does cosine similarity rank same-instance pairs above
               different-instance pairs at the same distance? 0.5 = no instance information, 1 = perfect.

    cd experiments/04_ptv3_replica && uv run python code/analyze.py
"""

import json

import numpy as np
import torch

from encoder import RESULTS_DIR, encode, list_scenes, load_model

SCENES = ["room_0", "office_3", "frl_apartment_0", "hotel_0", "apartment_1"]


def purity_coverage(inst, assign):
    n_tok, n_inst = int(assign.max()) + 1, int(inst.max()) + 1
    counts = torch.zeros(n_tok * n_inst, device=inst.device).index_add_(
        0, assign * n_inst + inst, torch.ones_like(inst, dtype=torch.float)).view(n_tok, n_inst)
    major = counts.argmax(1)
    purity = counts.max(1).values.sum().item() / len(inst)
    coverage = len(torch.unique(major)) / len(torch.unique(inst))
    return purity, coverage


def pair_auc(coord, inst, feat, n_anchor=4000, radius=0.5, seed=0):
    g = torch.Generator(device=coord.device).manual_seed(seed)
    a = torch.randint(len(coord), (n_anchor,), generator=g, device=coord.device)
    b = torch.randint(len(coord), (200_000,), generator=g, device=coord.device)
    d = torch.cdist(coord[a], coord[b])
    ia, ib = torch.nonzero(d < radius, as_tuple=True)
    ia, ib = a[ia], b[ib]
    keep = ia != ib
    ia, ib = ia[keep], ib[keep]
    if len(ia) > 2_000_000:
        sel = torch.randperm(len(ia), generator=g, device=coord.device)[:2_000_000]
        ia, ib = ia[sel], ib[sel]
    f = torch.nn.functional.normalize(feat, dim=1)
    sim = torch.cat([(f[ia[i:i + 100_000]] * f[ib[i:i + 100_000]]).sum(1) for i in range(0, len(ia), 100_000)])
    same = inst[ia] == inst[ib]
    # Distance-controlled AUC: same-instance pairs are closer on average, and any feature that depends on position
    # (even a random network) would exploit that. So compute the AUC within narrow distance bins, then average.
    dist = (coord[ia] - coord[ib]).norm(dim=1)
    aucs, weights = [], []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (dist >= lo) & (dist < hi)
        if same[m].any() and (~same[m]).any():
            aucs.append(_auc(sim[m], same[m]))
            weights.append(m.sum().item())
    auc = float(np.average(aucs, weights=weights))
    return auc, _auc(sim, same), len(sim), same.float().mean().item()


BINS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]


def _auc(score, pos):
    """P(score_pos > score_neg) via ranks (Mann-Whitney U)."""
    rank = torch.empty_like(score).scatter_(0, score.argsort(),
                                            torch.arange(1, len(score) + 1, device=score.device, dtype=score.dtype))
    n_pos, n_neg = pos.sum().item(), (~pos).sum().item()
    return (rank[pos].sum().item() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    rows = []
    for pretrained in (True, False):
        name = "sonata" if pretrained else "ptv3_scratch"
        model = load_model(pretrained).cuda()
        for scene in SCENES:
            out = encode(model, scene)
            inst = out["instance"]
            stage_stats = []
            for s, st in enumerate(out["stages"]):
                pur, cov = purity_coverage(inst, st["assign"])
                stage_stats.append({"stage": s, "tokens": len(st["feat"]), "purity": pur, "coverage": cov})
            auc, auc_raw, n_pairs, frac_same = pair_auc(out["coord"], inst, out["full"])
            rows.append({"model": name, "scene": scene, "stages": stage_stats, "auc": auc, "auc_raw": auc_raw, "pairs": n_pairs, "frac_same": frac_same})
            print(f"{name:12s} {scene:16s} AUC {auc:.3f} (raw {auc_raw:.3f})  " + "  ".join(
                f"s{x['stage']}: {x['tokens']} tok pur {x['purity']:.3f} cov {x['coverage']:.2f}" for x in stage_stats), flush=True)
        del model
        torch.cuda.empty_cache()

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "analysis.json").write_text(json.dumps(rows, indent=1))
    lines = ["# Instance structure of PTv3 tokens and features\n",
             "Input: 2 cm grid sample of the 1 cm Replica clouds. Stage s pools by 2^s (stage 4 ~ 32 cm cells). "
             "Purity / coverage depend only on the pooling (identical for both models); AUC compares features.\n",
             "| scene | " + " | ".join(f"s{s} tokens / purity / coverage" for s in range(5)) + " |",
             "|---|" + "---|" * 5]
    for r in rows:
        if r["model"] == "sonata":
            lines.append(f"| {r['scene']} | " + " | ".join(
                f"{x['tokens']:,} / {x['purity']:.3f} / {x['coverage']:.2f}" for x in r["stages"]) + " |")
    lines += ["\n## Same-instance AUC of per-point features (pairs < 0.5 m apart)\n", "Distance-controlled: AUC within distance bins of 5-10 cm, averaged (raw AUC in brackets is confounded by distance).\n", "| scene | sonata | ptv3_scratch |", "|---|---|---|"]
    for scene in SCENES:
        a = {r["model"]: (r["auc"], r["auc_raw"]) for r in rows if r["scene"] == scene}
        lines.append(f"| {scene} | " + " | ".join(f"{a[m][0]:.3f} ({a[m][1]:.3f})" for m in ("sonata", "ptv3_scratch")) + " |")
    (RESULTS_DIR / "analysis.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
