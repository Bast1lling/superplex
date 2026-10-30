"""Exports the view fits (optimize.py --views) for views.html into viewer/data/<scene>_<res>cm/views_<setting>/
(not versioned): cameras, images and labels, back-projected points, the carved grid and the SQs. Run export.py first
(GT points and the 3D reference come from there).

    python export_views.py [--res 0.02] [scene ...]
"""

import colorsys
import json

import numpy as np
from PIL import Image

from export import HERE, deformed_mesh, majority_instance, surface_voxels
import data  # noqa: E402  (export.py put code/ on the path)
import views  # noqa: E402


def color(i):
    """Same colors as the viewer: golden-ratio hues, gray for -1."""
    if i < 0:
        return np.array([136, 136, 136])
    return np.array(colorsys.hls_to_rgb((i * 0.618034) % 1, 0.55, 0.6)) * 255


def export(scene, setting, pts, max_points=1_000_000, seed=0):
    frames, depth_source = (views.SETTINGS | views.SCENE_SETTINGS)[setting]
    labelled = setting in views.SETTINGS
    out = HERE / "data" / f"{scene}_{data.RES * 100:g}cm" / f"views_{setting}"
    (out / "img").mkdir(parents=True, exist_ok=True)
    vs = views.load_views(scene, frames, pts, depth_source)
    for v in vs:
        Image.fromarray(v["rgb"]).save(out / "img" / f"{v['frame']}.jpg", quality=85)
        lut = np.stack([color(i) for i in range(-1, pts["inst"].max() + 1)]).astype(np.uint8)
        Image.fromarray(lut[v["label"] + 1]).save(out / "img" / f"{v['frame']}_label.png")
    vp = views.view_points(vs, labelled)
    rgb = np.concatenate([v["rgb"].reshape(-1, 3)[(v["depth"].reshape(-1) > 0) & ((v["label"].reshape(-1) >= 0) | (not labelled))]
                          for v in vs])
    keep = np.random.default_rng(seed).permutation(len(vp["pos"]))[:max_points]
    (out / "points.bin").write_bytes(vp["pos"][keep].astype("<f4").tobytes() + vp["inst"][keep].astype("<i2").tobytes()
                                     + rgb[keep].astype("u1").tobytes())
    grid = views.carve(vs, pts["inst"].max() + 1, labelled=labelled)
    L = grid.label_np
    idx = np.argwhere(surface_voxels(L))
    unk = np.argwhere(surface_voxels(np.where(L == data.DEEP, 0, -1)))  # the boundary of the never-seen space
    lab = np.concatenate([L[tuple(idx.T)], np.full(len(unk), data.DEEP)])
    (out / "voxels.bin").write_bytes(np.concatenate([idx, unk]).astype("<i2").tobytes() + lab.astype("<i2").tobytes())
    z = dict(np.load(data.result_path(scene, f"views_{setting}")))
    if not labelled:  # label-free fits: each SQ colored by the GT instance it covers most
        z["instance"] = majority_instance(z, data.voxelize(pts))
    meta = {"frames": [v["frame"] for v in vs], "c2w": [v["c2w"].tolist() for v in vs],
            "camera": {"fx": views.FX, "fy": views.FY, "cx": views.CX, "cy": views.CY, "w": vs[0]["depth"].shape[1],
                       "h": vs[0]["depth"].shape[0]},
            "points": len(keep), "voxels": len(lab), "lo": grid.lo_np.tolist(), "res": grid.res,
            "sqs": {"count": len(z["trans"]), "mesh": deformed_mesh(z, out / "sqs.bin"), "iou": float(z["iou"].mean()),
                    "instances": len(z["iou"])}}
    (out / "scene.json").write_text(json.dumps(meta))
    print(f"{scene} {setting}: {len(vs)} views, {meta['points']} points, {meta['voxels']} voxels, {meta['sqs']['count']} SQs")


if __name__ == "__main__":
    for scene in data.cli():
        pts = data.load_points(scene)
        for setting in views.SETTINGS | views.SCENE_SETTINGS:
            export(scene, setting, pts)
