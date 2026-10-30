"""depth: metric depth of a rendered frame from its RGB image with Depth Pro (Apple, HF apple/DepthPro-hf), given the
true focal length (Depth Pro predicts depth for its own focal estimate; rescaled to the true 600 px). Predictions on
the views' pixel grid are cached in Thesis/data/replica_mono_depth/depthpro/ (shared with experiments 13 / 15).
"""

import numpy as np
import torch
from PIL import Image

from data import DATA

CACHE = DATA / "replica_mono_depth" / "depthpro"


@torch.no_grad()
def depthpro(image, focal):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    cache = str(DATA / "models" / "hf")
    proc = AutoImageProcessor.from_pretrained("apple/DepthPro-hf", cache_dir=cache)
    net = AutoModelForDepthEstimation.from_pretrained("apple/DepthPro-hf", cache_dir=cache, dtype=torch.float16).cuda().eval()
    out = net(**proc(images=image, return_tensors="pt").to("cuda", torch.float16))
    post = proc.post_process_depth_estimation(out, target_sizes=[(image.height, image.width)])[0]
    return (post["predicted_depth"].float() * focal / float(post["focal_length"])).cpu().numpy()


def predict(scene, frame, image_path, focal, down):
    """Depth Pro depth of one frame on the pixel grid of the downsampled views (every `down`-th pixel)."""
    f = CACHE / scene / f"{frame:06d}.npy"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        np.save(f, depthpro(Image.open(image_path), focal)[down // 2::down, down // 2::down].astype(np.float32))
    return np.load(f)
