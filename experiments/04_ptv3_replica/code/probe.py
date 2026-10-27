"""Probe PTv3 / Sonata on Replica: parameters, tokens per stage, time and memory (inference and one training step).

    cd experiments/04_ptv3_replica && uv run python code/probe.py
"""

import json
import time

import torch

from encoder import list_scenes, load_model, scene_input, sonata_config, stages, RESULTS_DIR


def measure(model, scene, train, grid=0.02, amp=True):
    try:
        return _measure(model, scene, train, grid, amp)
    except torch.OutOfMemoryError:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return {"scene": scene, "mode": "train" if train else "infer", "grid": grid, "points_in": None,
                "seconds": None, "peak_gb": None, "stages": [], "oom": True}


def _measure(model, scene, train, grid, amp):
    point = scene_input(scene, grid)
    n_in = len(point["coord"])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp), torch.set_grad_enabled(train):
        out = model(point)
        if train:
            out.feat.float().pow(2).mean().backward()  # dummy loss on the coarsest tokens
    torch.cuda.synchronize()
    row = {"scene": scene, "mode": "train" if train else "infer", "grid": grid, "points_in": n_in, "oom": False,
           "seconds": time.perf_counter() - t, "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
           "stages": stages(out)}
    model.zero_grad(set_to_none=True)
    return row


def main():
    cfg = sonata_config()
    print("Sonata config:", {k: cfg[k] for k in ("enc_depths", "enc_channels", "stride", "enc_mode", "in_channels")})
    rows = []
    for pretrained in (True, False):
        model = load_model(pretrained).cuda()
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        for train, grid in ((False, 0.02), (True, 0.02), (True, 0.04)):
            model.train(train)
            for scene in ("room_0", "office_3", "frl_apartment_0", "apartment_0"):
                measure(model, scene, train, grid)  # warm-up (allocator, kernels)
                row = measure(model, scene, train, grid)
                row.update(model="sonata" if pretrained else "ptv3_scratch", params_m=n_params)
                rows.append(row)
                if row["oom"]:
                    print(f"{row['model']:12s} {row['mode']:5s} grid {grid} {scene:16s} OOM", flush=True)
                    continue
                print(f"{row['model']:12s} {row['mode']:5s} grid {grid} {scene:16s} in {row['points_in']:>9,}  "
                      f"{row['seconds']:6.2f} s  {row['peak_gb']:5.1f} GB  stages {[s[0] for s in row['stages']]}", flush=True)
        del model
        torch.cuda.empty_cache()
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "probe.json").write_text(json.dumps({"config": cfg, "rows": rows}, indent=1, default=str))


if __name__ == "__main__":
    main()
