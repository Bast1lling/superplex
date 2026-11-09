"""Per-frame training cache, part 1: DINOv3 tokens, the /4 view (RGB, depth, GT instance per pixel) and the target SQs
in the camera frame. Part 2 (Sonata features) is `cache_sonata.py`, in experiment 04's environment.

    uv run python cache_frames.py [scene ...]      -> FRAMES/<scene>/<frame>_{dino.npy, view.npz}, FRAMES/classes.json
    uv run python cache_frames.py --select         the frames of results/frame_selection.json (`select_frames.py`)

Every 10th frame of the 8 NICE-SLAM scenes. Rows 4..676 of the 1200 x 680 image are kept (42 x 75 patches of
16 px); at /4 that is a 168 x 300 grid, 4 x 4 pixels per patch.
- dino: DINOv3 ViT-L/16 last layer patch tokens (42, 75, 1024) float16 (computed in bfloat16: float16 overflows to
  NaN inside the ViT), ImageNet normalization, frozen.
- view: rgb (168, 300, 3) u8, depth (168, 300) f16 (z, m), label (168, 300) i16 = GT instance index (experiment 12's
  `gt`, -1 unlabelled), c2w; targets = every SQ of `results/targets_winding` that intersects the camera frustum
  (a vertex of its mesh projects into the kept image in front of the camera, z > 0.1 m): scale, exponents, rotation,
  translation in the camera frame (x right, y down, z forward), its instance and class (index into classes.json).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP.parent / "16_superflex_objective" / "code"))
from sfopt import PredictionHandler, gt  # noqa: E402
from views import CAM, RENDERS, load_view  # noqa: E402
from fit_targets import SCENES  # noqa: E402

FRAMES = Path("/home/sebastian/Documents/Uni/Thesis/data/replica_rgbd_frames")
STRIDE, ROW0, ROWS, W = 10, 4, 672, 1200  # full-resolution crop rows ROW0 .. ROW0 + ROWS
DOWN = 4
DINO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def class_vocab():
    names = sorted({n for s in SCENES for n in gt(s)["names"]})
    (FRAMES / "classes.json").write_text(json.dumps(names))
    return {n: i for i, n in enumerate(names)}


def sq_meshes(scene):
    """Mesh vertices of every existing target SQ (for the frustum test) and its parameters."""
    z = np.load(EXP / "results" / "targets_winding" / scene / "sqs.npz")
    mesher = PredictionHandler.__new__(PredictionHandler)
    verts, rows = [], []
    for b, k in enumerate(z["instance"]):
        for j in np.flatnonzero(z["exist"][b] > 0.5):
            v, _ = mesher._superquadric_mesh(z["scale"][b, j], z["exponents"][b, j], z["rotation"][b, j],
                                             z["translation"][b, j], np.zeros(2), np.zeros(6), 8)
            verts.append(v)
            rows.append((z["scale"][b, j], z["exponents"][b, j], z["rotation"][b, j], z["translation"][b, j], k))
    return verts, rows


def frustum(verts, w2c):
    out = []
    for v in verts:
        c = v @ w2c[:3, :3].T + w2c[:3, 3]
        u = CAM["fx"] * c[:, 0] / c[:, 2] + CAM["cx"]
        r = CAM["fy"] * c[:, 1] / c[:, 2] + CAM["cy"]
        out.append(((c[:, 2] > 0.1) & (u >= 0) & (u < W) & (r >= ROW0) & (r < ROW0 + ROWS)).any())
    return np.array(out)


@torch.no_grad()
def main():
    from transformers import AutoModel
    FRAMES.mkdir(parents=True, exist_ok=True)
    vocab = class_vocab()
    dino = AutoModel.from_pretrained(DINO, torch_dtype=torch.bfloat16).cuda().eval()
    skip = 1 + dino.config.num_register_tokens
    select = None
    if "--select" in sys.argv:
        sys.argv.remove("--select")
        select = json.loads((EXP / "results" / "frame_selection.json").read_text())["frames"]
    for scene in sys.argv[1:] or (list(select) if select else SCENES):
        out = FRAMES / scene
        out.mkdir(exist_ok=True)
        names = gt(scene)["names"]
        verts, rows = sq_meshes(scene)
        root = RENDERS / scene.replace("_", "") / "results"
        n = len(list(root.glob("frame*.jpg")))
        for f in (select[scene] if select else range(0, n, STRIDE)):
            if not (out / f"{f:06d}_dino.npy").exists():
                img = np.asarray(Image.open(root / f"frame{f:06d}.jpg"))[ROW0:ROW0 + ROWS]
                x = torch.tensor((img / 255 - MEAN) / STD, dtype=torch.bfloat16, device="cuda").permute(2, 0, 1)[None]
                tok = dino(pixel_values=x).last_hidden_state[0, skip:].reshape(ROWS // 16, W // 16, -1).float()
                assert torch.isfinite(tok).all() and tok.abs().max() < 6e4, (scene, f)
                np.save(out / f"{f:06d}_dino.npy", tok.cpu().numpy().astype(np.float16))
            if (out / f"{f:06d}_view.npz").exists():
                continue
            v = load_view(scene, f, DOWN)
            r = slice(ROW0 // DOWN, (ROW0 + ROWS) // DOWN)
            w2c = np.linalg.inv(v["c2w"])
            keep = frustum(verts, w2c)
            sel = [rows[i] for i in np.flatnonzero(keep)]
            Rw = w2c[:3, :3]
            np.savez_compressed(
                out / f"{f:06d}_view.npz", rgb=(v["rgb"][r] * 255).round().astype(np.uint8),
                depth=v["depth"][r].astype(np.float16), label=v["label"][r].astype(np.int16), c2w=v["c2w"],
                scale=np.array([s[0] for s in sel], np.float32).reshape(-1, 3),
                exponents=np.array([s[1] for s in sel], np.float32).reshape(-1, 2),
                rotation=np.array([Rw @ s[2] for s in sel], np.float32).reshape(-1, 3, 3),
                translation=np.array([Rw @ s[3] + w2c[:3, 3] for s in sel], np.float32).reshape(-1, 3),
                instance=np.array([s[4] for s in sel], np.int16),
                cls=np.array([vocab[names[s[4]]] for s in sel], np.int16))
            print(scene, f, f"{len(sel)} target SQs", flush=True)


if __name__ == "__main__":
    main()
