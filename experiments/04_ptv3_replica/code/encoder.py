"""PTv3 / Sonata encoder on Replica scenes.

Uses the standalone PTv3 implementation from the Sonata repo (third_party/sonata) and a torch_scatter stand-in
(third_party/shims). Scenes are the cached 1 cm point clouds of experiment 01 (xyz, rgb, normal, instance).
"""

from pathlib import Path
import sys

import numpy as np
import torch

EXP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(EXP / "third_party/shims"), str(EXP / "third_party/sonata"), str(EXP.parent / "01_scene_viewer/code")]
import sonata  # noqa: E402
from scenes import list_scenes, load_points, load_semantics  # noqa: E402,F401

WEIGHTS = Path("/home/sebastian/Documents/Uni/Thesis/data/models/sonata")
RESULTS_DIR = EXP / "results"
NO_FLASH = dict(enable_flash=False, enc_patch_size=[1024] * 5)  # flash-attn is not installed


def load_model(pretrained: bool = True, **overrides) -> torch.nn.Module:
    """Sonata (pretrained, encoder-only PTv3) or the same architecture randomly initialized."""
    cfg = {**NO_FLASH, **overrides}
    if pretrained:
        return sonata.model.load("sonata", repo_id="facebook/sonata", download_root=str(WEIGHTS), custom_config=cfg)
    ckpt = sonata.model.load("sonata", repo_id="facebook/sonata", download_root=str(WEIGHTS), custom_config=cfg,
                             ckpt_only=True)
    return sonata.model.PointTransformerV3(**ckpt["config"])  # same architecture, PTv3's own initialization


def sonata_config() -> dict:
    return sonata.model.load("sonata", repo_id="facebook/sonata", download_root=str(WEIGHTS), ckpt_only=True)["config"]


def transform(grid_size: float = 0.02):
    """Sonata's default inference pipeline (center shift, grid sampling, color normalization)."""
    return sonata.transform.Compose([
        dict(type="CenterShift", apply_z=True),
        dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train",
             return_grid_coord=True, return_inverse=True),
        dict(type="NormalizeColor"),
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "grid_coord", "color", "inverse", "instance"),
             feat_keys=("coord", "color", "normal")),
    ])


def scene_input(scene: str, grid_size: float = 0.02, device: str = "cuda") -> dict:
    """Grid-sampled network input; keeps the GT instance id of each sampled point ("instance")."""
    p = load_points(scene)
    point = {"coord": p["pos"].astype(np.float32), "color": p["col"].astype(np.float32),
             "normal": p["nrm"].astype(np.float32), "instance": p["instance"].astype(np.int64)}
    point = transform(grid_size)(point)
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in point.items()}


@torch.no_grad()
def encode(model, scene: str, grid_size: float = 0.02, seed: int = 0) -> dict:
    """Run the encoder on a whole scene and unroll its pooling hierarchy.

    Returns the network input points (coord, color, instance) and, per stage from finest (0) to coarsest,
    the token coords, token features and the token index of every input point ("assign").
    "full" is Sonata's recommended per-point feature: the last two coarse stages concatenated onto stage 2,
    then copied to the input resolution.
    """
    torch.manual_seed(seed)
    model.eval()
    inp = scene_input(scene, grid_size)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        point = model(dict(inp))
    chain = [point]
    while "pooling_parent" in chain[-1].keys():
        chain.append(chain[-1]["pooling_parent"])
    chain = chain[::-1]  # finest first
    assign = [torch.arange(len(chain[0].feat), device="cuda")]
    for s in range(1, len(chain)):
        assign.append(chain[s]["pooling_inverse"][assign[-1]])
    stage = [{"coord": c.coord.float(), "feat": c.feat.float(), "assign": a} for c, a in zip(chain, assign)]
    n = len(stage)
    full =torch.cat([stage[s]["feat"][stage[s]["assign"]] for s in (n - 3, n - 2, n - 1)], 1)
    return {"coord": inp["coord"], "color": inp["color"], "instance": inp["instance"], "stages": stage, "full": full}


def stages(point) -> list:
    """Walk the pooling hierarchy of an encoder output from coarsest to finest: [(n_points, channels, grid_size)]."""
    out = []
    while True:
        out.append((len(point.feat), point.feat.shape[1], float(point.grid_size) if "grid_size" in point.keys() else None))
        if "pooling_parent" not in point.keys():
            return out
        point = point["pooling_parent"]
