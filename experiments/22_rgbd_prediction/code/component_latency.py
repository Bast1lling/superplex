"""Training-step latency of the top-k model with single components changed (results/component_latency.md).

    uv run python component_latency.py [--steps 60]

Plain step time (forward + loss + backward + AdamW, batch 8, full training set, after 10 warm-up steps) and peak
memory, per variant of the default top-k configuration (d 128, 1 encoder layer, 1 decoder layer, K 300, Chamfer, focal
existence). "no masks" removes the mask heads and the mask / Dice losses (patched here, there is no switch).
"""

import argparse
import itertools
import json
import time

import torch

import train as T
import dense as D
import model as M

VARIANTS = {
    "default": {},
    "dense only (no decoder)": {"dense": True},
    "+ volumetric pair loss (--loss both)": {"loss": "both"},
    "+ union vs targets (--union target)": {"union": "target"},
    "+ union vs depth (--union observed)": {"union": "observed"},
    "decoder 3 layers": {"dec_layers": 3},
    "decoder 6 layers": {"dec_layers": 6},
    "K 100": {"k": 100},
    "K 900": {"k": 900},
    "encoder 2 layers": {"layers": 2},
    "encoder 4 layers": {"layers": 4},
    "width 256": {"d": 256},
    "width 256, encoder 2 layers": {"d": 256, "layers": 2},
    "no masks": {"no_mask": True},
}


def no_mask_patches():
    heads_fwd, pair = M.Heads.forward, T.pair_losses

    def heads(self, q, mp, ref):
        mask_q = self.mask_q
        self.mask_q = None
        try:
            out = {"scale": torch.nn.functional.softplus(self.scale(q)) + 0.005,
                   "shape": 0.1 + 1.8 * torch.sigmoid(self.shape(q)),
                   "rotate": M.SuperFlexHead.quat2mat(torch.nn.functional.normalize(self.rot(q), dim=-1)),
                   "trans": ref + self.trans(q), "exist": torch.sigmoid(self.exist(q))[..., 0], "cls": self.cls(q),
                   "mask": q.new_zeros(*q.shape[:2], 1, 1)}
        finally:
            self.mask_q = mask_q
        return out

    def pairs(pred, batch, fb, qs, tg, n):
        B = len(batch["targets"])
        cat = {k: torch.cat([t[k] for t in batch["targets"]]) for k in ("scale", "exponents", "rotation", "translation", "cls")}
        tgt = T.surface(cat["scale"][tg], cat["exponents"][tg], cat["rotation"][tg], cat["translation"][tg], n)
        ps, pe, pr = pred["scale"][fb, qs], pred["shape"][fb, qs], pred["rotate"][fb, qs]
        return {"cls": T.seg_mean(torch.nn.functional.cross_entropy(pred["cls"][fb, qs], cat["cls"][tg], reduction="none"), fb, B),
                "cd": T.seg_mean(T.chamfer(T.surface(ps, pe, pr, pred["trans"][fb, qs], n), tgt), fb, B),
                "forced": T.seg_mean(T.chamfer(T.surface(ps, pe, pr, cat["translation"][tg], n), tgt), fb, B)}

    return (heads_fwd, pair), heads, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    a = ap.parse_args()
    n_classes = len(json.loads((T.FRAMES / "classes.json").read_text()))
    loader = torch.utils.data.DataLoader(T.Frames(T.TRAIN), batch_size=8, shuffle=True, drop_last=True, collate_fn=T.collate,
                                         num_workers=6, pin_memory=True, persistent_workers=True)
    it = itertools.chain.from_iterable(itertools.repeat(loader))
    (orig_heads, orig_pairs), heads, pairs = no_mask_patches()
    lines = [f"\n# Component latency {time.strftime('%Y-%m-%d %H:%M')}, {torch.cuda.get_device_name()}, batch 8, "
             f"{a.steps} steps", "", "| variant | step ms | Δ vs default | peak GB | parameters |", "|---|---|---|---|---|"]
    base = None
    for name, v in VARIANTS.items():
        T.LOSS, T.UNION, T.MASK = v.get("loss", "cd"), v.get("union", "none"), "instance"
        M.Heads.forward, T.pair_losses = (heads, pairs) if v.get("no_mask") else (orig_heads, orig_pairs)
        torch.manual_seed(0)
        kw = dict(d=v.get("d", 128), layers=v.get("layers", 1))
        model = (M.DenseModel(n_classes, **kw) if v.get("dense") else
                 M.TopKModel(n_classes, dec_layers=v.get("dec_layers", 1), k=v.get("k", 300), **kw)).to(T.DEV)
        losses = D.losses if v.get("dense") else D.topk_losses
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        torch.cuda.reset_peak_memory_stats()

        def step():
            batch = T.to_device(next(it))
            loss, _ = losses(model(batch["dino"], batch["feats"], batch["idx"], batch["depth"]), batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        for _ in range(10):
            step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.steps):
            step()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / a.steps
        base = base or dt
        lines.append(f"| {name} | {dt * 1e3:.0f} | {(dt - base) * 1e3:+.0f} | {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} | "
                     f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M |")
        print(lines[-1], flush=True)
    M.Heads.forward, T.pair_losses = orig_heads, orig_pairs
    with open(T.EXP / "results" / "component_latency.md", "a") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
