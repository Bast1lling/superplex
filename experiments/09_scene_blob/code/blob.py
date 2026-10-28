"""SQ decomposition of the whole scene WITHOUT semantics: the targets are all solid voxels (one blob, split only into
its connected components), decomposed with experiment 10's quality-driven top-down / bottom-up search.

    python blob.py --name <run> [--set key=value ...]      (keys: experiment 10's decomp.CFG)

Evaluation against the GT instances (never used by the fit): every SQ is assigned to the instance owning most of
the solid voxels inside it. Reported: SQ purity (share of the voxels inside an SQ that belong to its instance),
per-instance K and strict vIoU of the assigned SQs (same scorer as exp 10, so directly comparable to the
instance-aware decompositions), and the share of instances that get no SQ.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP.parent / "10_adaptive_complexity" / "code"))
import decomp  # noqa: E402
import partfit as pf  # noqa: E402

CFG = dict(decomp.CFG) | dict(scene="room_0", method="both", q=0.8, w_sib=0.1)


class BlobScene(pf.Scene):
    """All solid voxels as targets; groups = connected components of the solid (26-neighborhood)."""

    def __init__(self, scene, min_comp=8, **kw):
        super().__init__(scene, planar=False, **kw)
        solid = np.argwhere(self.L_np >= 0)
        comps = [c for c in pf.components(solid) if len(c) >= min_comp]
        self.blob_targets = [solid[c] for c in comps]
        G = np.full(self.L_np.shape, -1, np.int64)
        for g, v in enumerate(self.blob_targets):
            G[tuple(v.T)] = g
        self.G = torch.from_numpy(G).to(pf.DEV)


@torch.no_grad()
def assign(S, sqs, chunk=1 << 15):
    """SQ -> majority GT instance of the solid voxels inside it, and its purity."""
    from sq import sdf_all
    par = {k: torch.from_numpy(np.asarray(sqs[k])).float().to(pf.DEV) for k in ("trans", "scale", "eps", "rot")}
    solid = np.argwhere(S.L_np >= 0)
    lab = torch.from_numpy(S.L_np[tuple(solid.T)]).to(pf.DEV)
    x = torch.from_numpy(S.centers(solid).astype(np.float32)).to(pf.DEV)
    M, I = len(sqs["trans"]), S.n_inst
    hist = torch.zeros(M, I, device=pf.DEV)
    for a in range(0, len(x), chunk):
        inside = (sdf_all(par, x[a:a + chunk]) < 0).float()          # (n, M)
        oh = torch.nn.functional.one_hot(lab[a:a + chunk], I).float()  # (n, I)
        hist += inside.T @ oh
    tot = hist.sum(1)
    inst = hist.argmax(1)
    purity = hist.max(1).values / tot.clamp(min=1)
    return inst.cpu().numpy(), purity.cpu().numpy(), tot.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    cfg = decomp.parse_set(dict(CFG), a.set)
    pf.W["w_sib"] = cfg["w_sib"]
    torch.manual_seed(cfg["seed"])
    t0 = time.time()
    S = BlobScene(cfg["scene"])
    groups = list(range(len(S.blob_targets)))
    print(f"{len(groups)} components, {sum(len(t) for t in S.blob_targets)} voxels", flush=True)
    S.targets_inst, S.targets = S.targets, S.blob_targets  # decomp works on S.targets
    parts, group, log = list(S.targets), groups, []
    if cfg["method"] in ("topdown", "both"):
        parts, group = decomp.topdown(S, parts, group, cfg, log)
    if cfg["method"] == "bottomup":
        parts, group = decomp.cells(S, parts, group, cfg["cell"])
    if cfg["method"] in ("bottomup", "both"):
        parts, group = decomp.bottomup(S, parts, group, cfg, log)
    sqs = decomp.finalize(S, parts, group, groups, cfg, log)
    S.targets = S.targets_inst
    inst, purity, vox = assign(S, sqs)
    sqs = sqs | {"component": sqs["instance"], "instance": inst, "purity": purity}
    summ, per = decomp.save(a.name, cfg, S, {k: v for k, v in sqs.items() if k not in ("component", "purity")},
                            log, t0, exp=EXP)
    out = EXP / "results" / "runs" / a.name
    np.savez_compressed(out / "sqs.npz", **sqs, instance_ids=S.sc["instance_ids"])
    w = vox / max(vox.sum(), 1)
    summ |= {"components": len(groups), "purity_mean": float(purity.mean()), "purity_vol": float((purity * w).sum()),
             "pure80": float((purity >= 0.8).mean()), "instances_without_sq": float((np.array(per["K"]) == 0).mean())}
    m = json.loads((out / "metrics.json").read_text())
    m["summary"] = summ
    m["sq_purity"] = purity.tolist()
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    print(a.name, json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
