"""Per-frame training cache, part 3: the one-to-many assignment of `dense.assign`, the heatmap target of
`dense.heatmap` and the visible fraction of every target SQ (`train.visible_fraction`), which depend only on the frame
and its targets (train.Frames loads them, so they are not recomputed every step).

    uv run python cache_assign.py [--select] [--skip names] [--only names]
                                              -> FRAMES/<scene>/<frame>_assign.npz: owner (42 * 75,) int16, heat
                                                 (42 * 75,) float32, visible (targets,) bool, f (targets,) float32
                                                 (grid samples), f_cube (cube samples, `train.surface`),
                                                 pixown (168, 300) int16 (`train.pixel_owner`), ownz (168, 300)
                                                 float16 the owner's entry depth along the pixel ray

--select: the frames of results/frame_selection.json (`select_frames.py`) instead of every 10th frame; existing
assign files with f are kept. Rerun whenever the targets in <frame>_view.npz change (or train.OWNER_RULE: the
pixel owner decides the visible targets, the patch owners and f).
"""

import json
import sys

import numpy as np
import torch

import train as T
import dense as D


def main():
    if "--select" in sys.argv:
        sel = json.loads(T.SELECTION.read_text())["frames"]
        ds = T.Frames(T.TRAIN, select=sel)
        ds.items = [p for p in ds.items if not (p.parent / p.name.replace("_view", "_assign")).exists()
                    or "f_cube" not in np.load(p.parent / p.name.replace("_view", "_assign"))]
    else:
        ds = T.Frames(T.TRAIN + T.TEST)
    name = lambda p: f"{p.parent.name}_{p.name[:6]}"
    for flag, keep in (("--skip", False), ("--only", True)):  # comma-separated frame names, e.g. room_1_000130
        if flag in sys.argv:
            names = set(sys.argv[sys.argv.index(flag) + 1].split(","))
            ds.items = [p for p in ds.items if (name(p) in names) == keep]
    loader = torch.utils.data.DataLoader(ds, batch_size=8, collate_fn=T.collate, num_workers=6)
    n = 0
    for batch in loader:
        for t in batch["targets"]:
            t.pop("f", None)  # recompute, not reuse an older cache
        batch.pop("pixown", None), batch.pop("ownz", None)
        batch = T.to_device(batch)
        batch.pop("owner", None), batch.pop("heat", None)  # (to_device recomputed pixown and f: not in old caches)
        for b, t in enumerate(batch["targets"]):
            owner, visible = D.assign(batch, b, t)
            scene, fid = batch["name"][b].rsplit("_", 1)
            np.savez(T.FRAMES / scene / f"{fid}_assign.npz", owner=owner.cpu().numpy().astype(np.int16),
                     heat=D.heatmap(owner).cpu().numpy(), visible=visible.cpu().numpy(),
                     f=t["f"].cpu().numpy().astype(np.float32),
                     f_cube=T.visible_fraction(t, batch["pixown"][b:b + 1], batch["depth"][b:b + 1],
                                               ownz=T.zslice(batch, b), kind="cube").cpu().numpy().astype(np.float32),
                     pixown=batch["pixown"][b].cpu().numpy().astype(np.int16),
                     ownz=batch["ownz"][b].cpu().numpy().astype(np.float16))
            n += 1
    print(n, "frames")


if __name__ == "__main__":
    main()
