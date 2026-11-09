"""make_sq on its own."""

import numpy as np
import torch
from omegaconf import OmegaConf
from variants import DEV, Variant


def make_sq(chairs, handler, loss="iou", distance="scaled", tau=0.01, w_free=1.0, cap=None):
    external = {"points": torch.tensor(np.stack([c["surf"] for c in chairs]), device=DEV),
                "points_iou": torch.tensor(np.stack([c["q"] for c in chairs]), device=DEV),
                "occupancies": torch.tensor(np.stack([c["occ"] for c in chairs]), device=DEV)}
    sq = Variant(pred_handler=handler, indices=list(range(len(chairs))), device=DEV, external_data=external,
                 cfg=OmegaConf.create({"tapering": False, "bending": False, "pruning": False}))
    return sq.setup(loss, distance, tau, w_free, cap)
