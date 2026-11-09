"""Two root-cause numbers for the target chair. How much do the sheets weigh in IoU, compared to their share of the
surface? And what does the voxel loss say when the back sheet of its fit is trimmed to the real panel?

    python trim_test.py <folder of diagnose.py's fits>
"""

import json
import sys

from pathlib import Path

import numpy as np
import torch
from diagnose import regions, term
from make_sq import make_sq
from scipy.spatial import cKDTree
from variants import DATA, DEV, TARGET, PredictionHandler, load_chair, scaled_distance

chair = load_chair(TARGET)
parts = regions()
all_surface = np.load(DATA / TARGET / "pointcloud.npz")["points"].astype(np.float32)
q, occ = chair["q"], chair["occ"]
for name, x in parts.items():
    near = cKDTree(x).query(q[occ])[0] < 0.012
    n = int(near.sum())
    print(f"{name}: {len(x) / len(all_surface):.1%} of the surface, {n} of {occ.sum()} occupied query points ({n / occ.sum():.1%}), "
          f"perfect fit otherwise but this sheet missing: IoU {1 - n / occ.sum():.3f}")

z = dict(np.load(sys.argv[1] + "/voxel_radial.npz"))
K = len(z["scale"])
thin = np.sort(z["scale"], 1)[:, 0]
j = int(np.argmax(np.where(thin < 0.01, -np.abs(z["rotation"][np.arange(K), 1, z["scale"].argmin(1)]), -9)))  # thin, normal not along y
axes = z["rotation"][j]
print("back sheet", j, "center", z["translation"][j].round(3), "half", z["scale"][j].round(3))
print("its axes, world:", axes.T.round(2).tolist())
panel_bottom = float(np.percentile(parts["back"][:, 1], 1))
panel_top = float(parts["back"][:, 1].max())
print(f"real panel y {panel_bottom:.3f} .. {panel_top:.3f}")
along = int(np.argmax(np.abs(axes[1])))
sign = np.sign(axes[1, along])
extent = z["translation"][j, 1] + sign * axes[1, along] * np.array([-1, 1]) * z["scale"][j, along]
print(f"sheet y {extent.min():.3f} .. {extent.max():.3f} along its axis {along}")


def make(params):
    handler = PredictionHandler({"names": np.arange(1), "pc": chair["surf"][None], "assign_matrix": np.zeros((1, 1, K)), "exist": np.ones((1, K)),
                                 "scale": params["scale"][None], "translation": params["translation"][None], "rotation": params["rotation"][None],
                                 "exponents": params["exponents"][None], "tapering": np.zeros((1, K, 2)), "bending": np.zeros((1, K, 6))})
    return make_sq([chair], handler, "vox", "radial", 0.005, 50.0)


def losses(params):
    sq = make(params)
    with torch.no_grad():
        return {name: float(term(sq, name)[0]) for name in ("positive", "free")}, sq


full, sq_full = losses(z)
trimmed = {key: value.copy() for key, value in z.items()}
top = extent.max()
half = (top - panel_bottom) / 2 / abs(axes[1, along])
trimmed["translation"][j] += axes[:, along] * (sign if False else 1) * np.sign(axes[1, along]) * (z["scale"][j, along] - half)
trimmed["scale"][j, along] = half
cut, sq_cut = losses(trimmed)
print("full   ", {k: round(v, 5) for k, v in full.items()}, "sum", round(sum(full.values()), 5))
print("trimmed", {k: round(v, 5) for k, v in cut.items()}, "sum", round(sum(cut.values()), 5))


def covered(params, x):
    R, t, s, e = [torch.tensor(params[k], device=DEV, dtype=torch.float32) for k in ("rotation", "translation", "scale", "exponents")]
    x = torch.tensor(x, device=DEV)
    return (scaled_distance((x[None] - t[:, None]) @ R, s[:, None], e[:, 0, None], e[:, 1, None]).amin(0) < 0.01).cpu().numpy()


lost = covered(z, all_surface) & ~covered(trimmed, all_surface)
print("surface samples lost by trimming:", int(lost.sum()), "of", len(all_surface))
if lost.sum():
    x = all_surface[lost]
    print("  where: x", x[:, 0].min().round(2), x[:, 0].max().round(2), "y", x[:, 1].min().round(2), x[:, 1].max().round(2), "z", x[:, 2].min().round(2), x[:, 2].max().round(2))
    print("  in the seat region:", int((cKDTree(parts["seat"]).query(x)[0] < 0.005).sum()))
gain = covered(z, q) & ~covered(trimmed, q)
print("query points lost:", int(gain.sum()), "occupied", int((gain & occ).sum()), "free", int((gain & ~occ).sum()))


# is it the samples, or the distance? The same two states under the scaled distance, and the distance of the panel's own samples
def back_distance(sq):
    x = (torch.tensor(parts["back"], device=DEV) - sq.normalization_translation[0]) / sq.normalization_scale[0]
    with torch.no_grad():
        return sq.sdf_batch(x.T[None].contiguous())[0].amin(0).clamp_min(0).mean().item()


report = {"sheet": j, "panel": [panel_bottom, panel_top], "sheet_y": [float(extent.min()), float(extent.max())], "lost_surface": int(lost.sum())}
for distance in ("radial", "scaled"):
    row = []
    for params in (z, trimmed):
        sq = make(params)
        sq.distance = distance
        with torch.no_grad():
            row.append({name: float(term(sq, name)[0]) for name in ("positive", "free")} | {"back": back_distance(sq)})
    print(distance, "full", {k: round(v, 4) for k, v in row[0].items()}, "sum", round(row[0]["positive"] + row[0]["free"], 3),
          "| trimmed", {k: round(v, 4) for k, v in row[1].items()}, "sum", round(row[1]["positive"] + row[1]["free"], 3))
    report[distance] = row

R, t, s, e = [torch.tensor(z[k], device=DEV, dtype=torch.float32)[j:j + 1] for k in ("rotation", "translation", "scale", "exponents")]
x = torch.tensor(q, device=DEV)
inside = (scaled_distance((x[None] - t[:, None]) @ R, s[:, None], e[:, 0, None], e[:, 1, None])[0] < 0).cpu().numpy()
report["sheet_query"] = {"free": int((inside & ~occ).sum()), "occupied": int((inside & occ).sum())}
np.savez(sys.argv[1] + "/voxel_radial_trimmed.npz", **trimmed, lost=all_surface[lost], air=q[inside & ~occ])
(Path(sys.argv[1]) / "trim.json").write_text(json.dumps(report))
