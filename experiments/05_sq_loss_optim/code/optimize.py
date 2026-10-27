"""Direct optimization of free SQ parameters with the voted-label CE + swallow hinge (see config.yaml).

    uv run python experiments/05_sq_loss_optim/code/optimize.py [--config path] [--name run] [--set key=value ...]
    e.g. --set loss.sdf=gauge_lb --set optim.steps=1000

Writes results/runs/<name>/{config.yaml, metrics.json, snapshots.npz} and logs to trackio (results/large_files/trackio).
"""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

EXP = Path(__file__).resolve().parent.parent
RESULTS = EXP / "results"
os.environ.setdefault("TRACKIO_DIR", str(RESULTS / "large_files" / "trackio"))  # keep trackio data inside the experiment
import trackio  # noqa: E402

sys.path[:0] = [str(EXP.parent / "02_sq_sdf/code"), str(EXP.parent / "01_scene_viewer/code")]
from scenes import load_points, load_semantics  # noqa: E402
import sdf as sdf_lib  # noqa: E402

DEVICE = "cuda"
PROJECT = "sq_loss_optim"


# ---------------------------------------------------------------- config

def load_config(path: Path, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(path.read_text())
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = yaml.safe_load(value)
    return cfg


# ---------------------------------------------------------------- data

def load_scene(cfg: dict) -> dict:
    """Labeled 1 cm points in the frame of the chosen token version, instance labels remapped to 0..K-1."""
    d = cfg["data"]
    pts = load_points(d["scene"])
    tok = np.load(Path(d["tokens_dir"]) / f"{d['scene']}.npz")
    v = d["version"]
    R, t = tok["R"][v], tok["t"][v]
    pos = pts["pos"] @ R.T + t
    inst = pts["instance"].astype(np.int64)
    if d["drop_unlabeled"]:
        l2 = np.array(load_semantics(d["scene"])["id_to_label"])
        keep = l2[np.minimum(inst, len(l2) - 1)] > 0
        pos, inst = pos[keep], inst[keep]
    ids, y = np.unique(inst, return_inverse=True)
    tokens = tok["coord"][tok["offsets"][v]:tok["offsets"][v + 1]]
    return {"pos": torch.from_numpy(pos).float().to(DEVICE), "y": torch.from_numpy(y).to(DEVICE),
            "instance_ids": ids, "tokens": tokens, "R": R, "t": t}


def anchors(tokens: np.ndarray, cell: float) -> np.ndarray:
    """Merge positions (tokens or scene points) into cells of the given size (mean position per occupied cell)."""
    key = np.floor(tokens / cell).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    sums = np.zeros((inv.max() + 1, 3))
    np.add.at(sums, inv, tokens)
    return (sums / np.bincount(inv)[:, None]).astype(np.float32)


class DistanceGrid:
    """Unsigned distance to the nearest scene point on a regular grid (EDT of occupied cells), trilinear lookup."""

    def __init__(self, pos: torch.Tensor, res: float, pad: float):
        from scipy.ndimage import distance_transform_edt
        p = pos.cpu().numpy()
        self.lo = p.min(0) - pad
        dims = np.ceil((p.max(0) + pad - self.lo) / res).astype(int) + 1
        occ = np.zeros(dims, bool)
        idx = np.clip(np.round((p - self.lo) / res).astype(int), 0, dims - 1)
        occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        dist = distance_transform_edt(~occ).astype(np.float32) * res  # (X, Y, Z)
        self.grid = torch.from_numpy(dist).permute(2, 1, 0)[None, None].contiguous().to(DEVICE)  # (1, 1, Z, Y, X)
        self.lo_t = torch.from_numpy(self.lo).float().to(DEVICE)
        self.span = torch.from_numpy((dims - 1) * res).float().to(DEVICE)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        g = (x - self.lo_t) / self.span * 2 - 1
        out = torch.nn.functional.grid_sample(self.grid, g.view(1, -1, 1, 1, 3), mode="bilinear",
                                              padding_mode="border", align_corners=True)
        return out.view(x.shape[:-1])


def fibonacci_sphere(m: int) -> torch.Tensor:
    i = torch.arange(m, dtype=torch.float32) + 0.5
    z = 1 - 2 * i / m
    r = (1 - z * z).sqrt()
    phi = math.pi * (3 - math.sqrt(5)) * i
    return torch.stack([r * phi.cos(), r * phi.sin(), z], 1)


def random_rotation(g: torch.Generator) -> torch.Tensor:
    q = torch.nn.functional.normalize(torch.randn(4, device=DEVICE, generator=g), dim=0)
    w, x, y, z = q
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]).view(3, 3)


def surface_samples(sq: dict, dirs: torch.Tensor) -> torch.Tensor:
    """World points (N, M, 3) on every SQ surface: radial projection of unit directions (M, 3), scaled by a."""
    a = sq["scale"][:, None]                                     # (N, 1, 3)
    D = a * dirs[None]                                           # (N, M, 3)
    G = sdf_lib.gauge(D, a, sq["eps"][:, 0, None], sq["eps"][:, 1, None])
    y = D / G[..., None]
    return torch.einsum("nij,nmj->nmi", sq["rot"], y) + sq["trans"][:, None]


def union_weight(cfg: dict, sq: dict, y: torch.Tensor) -> torch.Tensor:
    """(N, M) weight that a surface sample of SQ j lies outside all other SQs (no gradient)."""
    n, m = y.shape[:2]
    with torch.no_grad():
        flat = y.reshape(-1, 3)
        owner = torch.arange(n, device=DEVICE).repeat_interleave(m)
        chunk = max(1024, (1 << 26) // n)  # bound the (N, chunk) intermediates for large N
        out = []
        for i in range(0, len(flat), chunk):
            d = sdf_matrix(cfg, sq, flat[i:i + chunk])          # (N, chunk)
            inside = sq["existence"][:, None] * torch.sigmoid(-d / cfg["loss"]["reverse_chamfer"]["union_sharpness"])
            logout = torch.log1p(-inside.clamp(max=1 - 1e-6))
            cols = torch.arange(len(d[0]), device=DEVICE)
            out.append(logout.sum(0) - logout[owner[i:i + chunk], cols])  # exclude the sample's own SQ
        return torch.cat(out).exp().view(n, m)


def reverse_chamfer(cfg: dict, sq: dict, grid: DistanceGrid, dirs: torch.Tensor) -> torch.Tensor:
    rc = cfg["loss"]["reverse_chamfer"]
    y = surface_samples(sq, dirs)
    v = union_weight(cfg, sq, y)
    gap = torch.relu(grid(y) - rc["delta"]) / rc["delta"]
    e = sq["existence"] if rc.get("existence_grad", True) else sq["existence"].detach()  # detach: no "switch off" escape
    return (e[:, None] * v * gap).mean()


class InstanceSampler:
    """Pick an instance uniformly, then a point uniformly within it."""

    def __init__(self, y: torch.Tensor, generator: torch.Generator):
        self.order = torch.argsort(y)
        counts = torch.bincount(y)
        self.start = torch.cumsum(counts, 0) - counts
        self.counts = counts
        self.g = generator

    def __call__(self, n: int) -> torch.Tensor:
        k = torch.randint(len(self.counts), (n,), device=DEVICE, generator=self.g)
        off = (torch.rand(n, device=DEVICE, generator=self.g) * self.counts[k]).long()
        return self.order[self.start[k] + off]


# ---------------------------------------------------------------- SQ parameters

class SQs(torch.nn.Module):
    def __init__(self, cfg: dict, centers: np.ndarray):
        super().__init__()
        n, ini, par = len(centers), cfg["init"], cfg["parametrization"]
        self.eps_min, self.eps_max = par["eps"]["min"], par["eps"]["max"]
        self.scale_min, self.scale_max = par["scale"]["min"], par["scale"]["max"]
        eps0 = (torch.tensor(ini["eps"]) - self.eps_min) / (self.eps_max - self.eps_min)
        self.translation = torch.nn.Parameter(torch.from_numpy(centers).float())
        self.log_scale = torch.nn.Parameter(torch.full((n, 3), math.log(ini["scale"])))
        self.eps_raw = torch.nn.Parameter(torch.logit(eps0).expand(n, 2).clone())
        self.rotation = torch.nn.Parameter(torch.tensor([1.0, 0, 0, 0]).expand(n, 4).clone())  # quaternion (w, x, y, z)
        p = torch.tensor(ini["existence"])
        self.existence_logit = torch.nn.Parameter(torch.logit(p).expand(n).clone())

    def forward(self):
        scale = self.log_scale.exp().clamp(self.scale_min, self.scale_max)
        eps = self.eps_min + (self.eps_max - self.eps_min) * torch.sigmoid(self.eps_raw)
        w, x, y, z = torch.nn.functional.normalize(self.rotation, dim=1).unbind(1)
        rot = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                           2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                           2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)
        return {"scale": scale, "eps": eps, "rot": rot, "trans": self.translation,
                "existence": torch.sigmoid(self.existence_logit)}


def sdf_matrix(cfg: dict, sq: dict, x: torch.Tensor) -> torch.Tensor:
    """(N SQs, P points) signed distances."""
    q = sdf_lib.to_local(x, sq["rot"], sq["trans"])
    method = sdf_lib.METHODS[cfg["loss"]["sdf"]]
    d = method(q, sq["scale"][:, None], sq["eps"][:, 0, None], sq["eps"][:, 1, None])
    tr = cfg["loss"].get("truncation", {"mode": "none"})
    if tr["mode"] == "none":
        return d
    t = tr["value"]
    dt = d.clamp(-t, t) if tr["mode"] == "hard" else t * torch.tanh(d / t)
    return dt if tr.get("side", "both") == "both" else torch.where(d < 0, dt, d)


def claims(cfg: dict, sq: dict, d: torch.Tensor) -> torch.Tensor:
    mem = cfg["loss"]["membership"]
    return sq["existence"][:, None] * torch.sigmoid((mem["offset"] - d) / mem["temperature"])


# ---------------------------------------------------------------- loss

def loss_fn(cfg: dict, sq: dict, x: torch.Tensor, y: torch.Tensor, n_labels: int, grid=None, dirs=None):
    L = cfg["loss"]
    d = sdf_matrix(cfg, sq, x)                                   # (N, P)
    w = claims(cfg, sq, d)                                       # (N, P)
    onehot = torch.nn.functional.one_hot(y, n_labels).float()    # (P, K)
    votes = w @ onehot / (w.sum(1, keepdim=True) + L["vote"]["eps"])  # (N, K) voted label per SQ
    if L["vote"]["stop_gradient"]:
        votes = votes.detach()
    e0 = L["nothing"]["weight"]
    total_claim = w.sum(0)                                       # (P,)
    q_true = ((w * votes[:, y]).sum(0) + e0 / n_labels) / (total_claim + e0)
    ce = -torch.log(q_true.clamp_min(1e-12)).mean()

    h = L["swallow_hinge"]
    own = w / (total_claim + 1e-9)                               # ownership pi_pj (real SQs only)
    d_assigned = (own * d).sum(0)
    hinge = (torch.relu(-h["delta"] - d_assigned) / h["delta"]).mean()

    total = L["label_ce"]["weight"] * ce + h["weight"] * hinge
    eh = L.get("enclosure_hinge", {})
    if eh.get("weight", 0.0) > 0:
        x_out = torch.relu(d_assigned - eh["tolerance"]) / eh["delta"]
        enclosure = (eh["cap"] * torch.tanh(x_out / eh["cap"])).mean()
        total = total + eh["weight"] * enclosure
    else:
        enclosure = torch.zeros(())
    logs = {"loss/ce": ce.item(), "loss/hinge": hinge.item(), "loss/enclosure": enclosure.item(),
            "batch/q_true_mean": q_true.mean().item(), "batch/uncovered": (total_claim < 0.5).float().mean().item()}
    rc_w = L.get("reverse_chamfer", {}).get("weight", 0.0)
    if rc_w > 0:
        rc = reverse_chamfer(cfg, sq, grid, dirs)
        total = total + rc_w * rc
        logs["loss/reverse_chamfer"] = rc.item()
    logs["loss/total"] = total.item()
    return total, logs


# ---------------------------------------------------------------- full-scene evaluation

@torch.no_grad()
def evaluate(cfg: dict, sq: dict, scene: dict, grid: "DistanceGrid", chunk: int | None = None) -> dict:
    """Votes from all points, then per-point predictions and metrics."""
    x, y, K = scene["pos"], scene["y"], len(scene["instance_ids"])
    N = len(sq["trans"])
    chunk = chunk or max(4096, min(131072, (1 << 25) // N))  # bound (N, chunk) intermediates for large N
    vote_num = torch.zeros(N, K, device=DEVICE)
    vote_den = torch.zeros(N, 1, device=DEVICE)
    for i in range(0, len(x), chunk):
        w = claims(cfg, sq, sdf_matrix(cfg, sq, x[i:i + chunk]))
        vote_num.index_add_(1, y[i:i + chunk], w)
        vote_den += w.sum(1, keepdim=True)
    votes = vote_num / (vote_den + cfg["loss"]["vote"]["eps"])
    sq_label = votes.argmax(1)
    purity = votes.max(1).values

    e0, delta = cfg["loss"]["nothing"]["weight"], cfg["loss"]["swallow_hinge"]["delta"]
    usable = sq["existence"] > cfg["metrics"]["active_existence"]
    pred, covered, d_assigned, owner, d_own = [], [], [], [], []
    for i in range(0, len(x), chunk):
        d = sdf_matrix(cfg, sq, x[i:i + chunk])
        w = claims(cfg, sq, d)
        tc = w.sum(0)
        q = (w.T @ votes + e0 / K) / (tc + e0)[:, None]
        pred.append(q.argmax(1))
        covered.append(tc >= cfg["metrics"]["uncovered_claim"])
        d_assigned.append(((w / (tc + 1e-9)) * d).sum(0))
        owner.append(w.argmax(0))
        same = (sq_label[:, None] == y[None, i:i + chunk]) & usable[:, None]  # SQs voted for the point's own instance
        d_own.append(torch.where(same, d, torch.full_like(d, float("inf"))).min(0).values)
    pred, covered, d_assigned, owner, d_own = map(torch.cat, (pred, covered, d_assigned, owner, d_own))

    conf = torch.bincount(y * K + pred, minlength=K * K).view(K, K).float()
    iou = conf.diag() / (conf.sum(0) + conf.sum(1) - conf.diag()).clamp_min(1)
    owned = torch.bincount(owner[covered], minlength=N)
    active = (sq["existence"] > cfg["metrics"]["active_existence"]) & (owned >= cfg["metrics"]["active_min_points"])
    per_inst = torch.bincount(sq_label[active], minlength=K)

    # union surface of the active SQs vs. scene: share of surface samples farther than delta from any scene point
    rc_delta = cfg["loss"].get("reverse_chamfer", {}).get("delta", 0.05)
    air_share, air_gap = 0.0, 0.0
    if active.any():
        sub = {k: v[active] for k, v in sq.items()}
        ys = surface_samples(sub, fibonacci_sphere(256).to(DEVICE))
        v = union_weight(cfg, sub, ys)
        u = grid(ys)
        wsum = v.sum().clamp_min(1e-9)
        air_share = ((v * (u > rc_delta)).sum() / wsum).item()
        air_gap = ((v * u).sum() / wsum).item()
    return {
        "metrics": {
            "eval/accuracy": (pred == y).float().mean().item(),
            "eval/mean_iou": iou.mean().item(),
            "eval/uncovered": 1 - covered.float().mean().item(),
            "eval/active_sqs": int(active.sum().item()),
            "eval/instances_with_sq": (per_inst > 0).float().mean().item(),
            "eval/sqs_per_covered_instance": (per_inst[per_inst > 0].float().mean().item() if (per_inst > 0).any() else 0.0),
            "eval/purity_active": purity[active].mean().item() if active.any() else 0.0,
            "eval/swallowed": (d_assigned[covered] < -delta).float().mean().item(),
            "eval/d_assigned_median": d_assigned[covered].median().item() if covered.any() else 0.0,
            "eval/outside_owner": (d_assigned > 0.01).float().mean().item(),  # points > 1 cm outside their assigned SQ
            # threshold-free coverage: point lies inside (<= 1 cm outside) an SQ voted for its own instance
            "eval/enclosed_correct": (d_own <= 0.01).float().mean().item(),
            "eval/enclosed_correct_inst": torch.zeros(K, device=DEVICE).index_add_(0, y, (d_own <= 0.01).float())
                                          .div(torch.bincount(y, minlength=K)).mean().item(),
            "eval/existence_mean": sq["existence"].mean().item(),
            "eval/surface_in_air": air_share,
            "eval/surface_gap_mean": air_gap,
        },
        "iou": iou.cpu().numpy(), "sq_label": sq_label.cpu().numpy(), "purity": purity.cpu().numpy(),
        "active": active.cpu().numpy(), "owned": owned.cpu().numpy(),
    }


# ---------------------------------------------------------------- main loop

def run(cfg: dict, name: str):
    torch.manual_seed(cfg["seed"])
    g = torch.Generator(device=DEVICE).manual_seed(cfg["seed"])
    scene = load_scene(cfg)
    K = len(scene["instance_ids"])
    src = scene["tokens"] if cfg["anchors"]["source"] == "sonata_stage4" else scene["pos"].cpu().numpy()  # "points"
    centers = anchors(src, cfg["anchors"]["cell"])
    model = SQs(cfg, centers).to(DEVICE)
    sampler = InstanceSampler(scene["y"], g)
    rc_cfg = cfg["loss"].get("reverse_chamfer", {"grid_res": 0.02, "grid_pad": 0.5, "samples_per_sq": 64})
    grid = DistanceGrid(scene["pos"], rc_cfg["grid_res"], rc_cfg["grid_pad"])
    dirs0 = fibonacci_sphere(rc_cfg["samples_per_sq"]).to(DEVICE)
    lr = cfg["optim"]["lr"]
    opt = torch.optim.Adam([
        {"params": [model.translation], "lr": lr["translation"]},
        {"params": [model.log_scale], "lr": lr["log_scale"]},
        {"params": [model.eps_raw], "lr": lr["eps"]},
        {"params": [model.rotation], "lr": lr["rotation"]},
        {"params": [model.existence_logit], "lr": lr["existence"]},
    ])
    steps = cfg["optim"]["steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))

    out = RESULTS / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    trackio.init(project=PROJECT, name=name, config=cfg, auto_log_gpu=False, auto_log_cpu=False)
    print(f"{name}: {len(scene['pos']):,} labeled points, {K} instances, {len(centers)} SQs", flush=True)

    snaps, history = [], []
    structure = None
    if cfg.get("structure", {}).get("enabled", False):
        from structure import Structure
        structure = Structure(cfg, len(centers))

    def snapshot(step):
        with torch.no_grad():
            sq = model()
        ev = evaluate(cfg, sq, scene, grid)
        snaps.append({"step": step, **{k: v.detach().cpu().numpy() for k, v in sq.items()},
                      "sq_label": ev["sq_label"], "purity": ev["purity"], "active": ev["active"], "owned": ev["owned"]})
        return ev

    t_start = time.perf_counter()
    for step in range(steps + 1):
        if step % cfg["logging"]["snapshot_every"] == 0 or step == steps:
            ev = snapshot(step)
            if step % cfg["logging"]["eval_every"] == 0 or step == steps:
                history.append({"step": step, **ev["metrics"]})
                trackio.log(ev["metrics"], step=step)
                m = ev["metrics"]
                print(f"step {step:5d}  acc {m['eval/accuracy']:.3f}  mIoU {m['eval/mean_iou']:.3f}  "
                      f"uncovered {m['eval/uncovered']:.3f}  active {m['eval/active_sqs']}  "
                      f"swallowed {m['eval/swallowed']:.3f}", flush=True)
        if step == steps:
            break
        if structure is not None and structure.due(step):
            slog = structure.step(model, opt, scene)
            trackio.log(slog, step=step)
            print(f"step {step:5d}  structure: {slog['struct/merges']} merges, {slog['struct/splits']} splits "
                  f"-> {slog['struct/n_sqs']} SQs", flush=True)
        t = time.perf_counter()
        idx = sampler(cfg["sampling"]["points_per_step"])
        dirs = dirs0 @ random_rotation(g).T  # fresh surface sample positions every step
        loss, logs = loss_fn(cfg, model(), scene["pos"][idx], scene["y"][idx], K, grid, dirs)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        torch.cuda.synchronize()
        if step % 10 == 0:
            logs.update({"time/step_ms": 1000 * (time.perf_counter() - t), "lr/translation": sched.get_last_lr()[0]})
            trackio.log(logs, step=step)
    trackio.finish()

    elapsed = time.perf_counter() - t_start
    keys = [k for k in snaps[0] if k != "step"]
    n_max = max(len(s["trans"]) for s in snaps)  # the SQ count changes with merging / splitting: pad

    def padded(k):
        fill = {"existence": -1.0, "scale": 0.01}.get(k, 0)
        out_ = []
        for s in snaps:
            v = s[k]
            pad = np.full((n_max - len(v),) + v.shape[1:], fill, dtype=v.dtype)
            if k == "rot":
                pad[:] = np.eye(3)
            out_.append(np.concatenate([v, pad]))
        return np.stack(out_)

    np.savez_compressed(out / "snapshots.npz", step=np.array([s["step"] for s in snaps]),
                        **{k: padded(k) for k in keys},
                        instance_ids=scene["instance_ids"], R=scene["R"], t=scene["t"])
    snap, big = out / "snapshots.npz", RESULTS / "large_files" / "runs" / name / "snapshots.npz"
    big.unlink(missing_ok=True)
    if snap.stat().st_size > 1e6:  # >1MB results go to results/large_files (not versioned)
        big.parent.mkdir(parents=True, exist_ok=True)
        snap.replace(big)
    final = history[-1]
    (out / "metrics.json").write_text(json.dumps({"name": name, "seconds": elapsed, "final": final,
                                                  "history": history, "final_iou": ev["iou"].tolist()}, indent=1))
    print(f"done in {elapsed:.0f} s -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(EXP / "config.yaml"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--set", action="append", default=[], help="override, e.g. loss.sdf=gauge_lb")
    args = ap.parse_args()
    cfg = load_config(Path(args.config), args.set)
    name = args.name or f"{cfg['data']['scene']}_{cfg['loss']['sdf']}_{time.strftime('%m%d-%H%M%S')}"
    run(copy.deepcopy(cfg), name)


if __name__ == "__main__":
    main()
