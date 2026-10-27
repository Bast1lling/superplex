"""Global top-down SQ fit to a scene point cloud (see config.yaml). No instance labels in the optimization.

    uv run python experiments/07_sq_global_fit/code/global_fit.py [--config path] [--name run] [--set key=value ...]

Writes results/runs/<name>/{config.yaml, metrics.json, snapshots.npz} (snapshot format of experiment 05, so the
viewer is shared; sq_label = majority GT instance of the owned points) and logs to trackio (results/large_files/trackio).
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
os.environ.setdefault("TRACKIO_DIR", str(RESULTS / "large_files" / "trackio"))  # before experiments 05 / 06 set their own
sys.path[:0] = [str(EXP.parent / "06_sq_gt_instance_fit/code"), str(EXP.parent / "05_sq_loss_optim/code")]
import trackio  # noqa: E402
from optimize import (DEVICE, DistanceGrid, fibonacci_sphere, load_config, load_points, load_scene,  # noqa: E402,F401
                      load_semantics, random_rotation, sdf_lib, surface_samples)
from structure import fit as box_fit, matrix_to_quat  # noqa: E402
from fit import kmeans  # noqa: E402  (experiment 06)

PROJECT = "sq_global_fit"


# ---------------------------------------------------------------- SQ pool

class SQPool(torch.nn.Module):
    """max_sqs slots; the first len(t) slots start alive with the given boxes, the others are free."""

    def __init__(self, cfg: dict, t: torch.Tensor, R: torch.Tensor, a: torch.Tensor):
        super().__init__()
        n, par = cfg["pool"]["max_sqs"], cfg["parametrization"]
        self.eps_min, self.eps_max = par["eps"]["min"], par["eps"]["max"]
        self.scale_min, self.scale_max = par["scale"]["min"], par["scale"]["max"]
        eps0 = (torch.tensor(cfg["init"]["eps"], device=t.device) - self.eps_min) / (self.eps_max - self.eps_min)
        pad = lambda v: torch.cat([v, v[:1].expand(n - len(v), *v.shape[1:])])  # noqa: E731  free slots: copies
        self.translation = torch.nn.Parameter(pad(t))
        self.log_scale = torch.nn.Parameter(pad(a.clamp(self.scale_min, self.scale_max).log()))
        self.eps_raw = torch.nn.Parameter(torch.logit(eps0).expand(n, 2).clone())
        self.rotation = torch.nn.Parameter(pad(matrix_to_quat(R)))
        alive = torch.zeros(n, dtype=torch.bool, device=t.device)
        alive[:len(t)] = True
        self.register_buffer("alive", alive)

    def forward(self):
        scale = self.log_scale.exp().clamp(self.scale_min, self.scale_max)
        eps = self.eps_min + (self.eps_max - self.eps_min) * torch.sigmoid(self.eps_raw)
        w, x, y, z = torch.nn.functional.normalize(self.rotation, dim=1).unbind(1)
        rot = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                           2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                           2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)
        return {"scale": scale, "eps": eps, "rot": rot, "trans": self.translation,
                "existence": self.alive.float(), "alive": self.alive}


def load_scene_with_normals(cfg: dict) -> dict:
    """load_scene (experiment 05) + unit normals of the same points in the same frame."""
    scene = load_scene(cfg)
    d = cfg["data"]
    pts = load_points(d["scene"])
    nrm = pts["nrm"].astype(np.float32) @ scene["R"].T.astype(np.float32)
    if d["drop_unlabeled"]:
        l2 = np.array(load_semantics(d["scene"])["id_to_label"])
        inst = pts["instance"].astype(np.int64)
        nrm = nrm[l2[np.minimum(inst, len(l2) - 1)] > 0]
    scene["nrm"] = torch.from_numpy(nrm).to(DEVICE)
    return scene


def rows(sq: dict, idx: torch.Tensor) -> dict:
    return {k: sq[k][idx] for k in ("scale", "eps", "rot", "trans")}


def sdf_rows(cfg: dict, s: dict, x: torch.Tensor) -> torch.Tensor:
    """(n SQs, P points) signed distances."""
    q = sdf_lib.to_local(x, s["rot"], s["trans"])
    return sdf_lib.METHODS[cfg["loss"]["sdf"]](q, s["scale"][:, None], s["eps"][:, 0, None], s["eps"][:, 1, None])


def forward_value(cfg: dict, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-point forward value and owner (row of d) from the (n, P) SDF matrix."""
    mode = cfg["loss"]["forward"]["mode"]
    m = (d.abs() if mode == "min_abs" else d).argmin(0)
    dm = d.gather(0, m[None]).squeeze(0)
    return (torch.relu(dm) if mode == "enclosure" else dm.abs()), m


@torch.no_grad()
def union_weight(cfg: dict, s: dict, ys: torch.Tensor) -> torch.Tensor:
    """(n, S) weight that a surface sample of SQ j lies outside all other SQs (no gradient)."""
    n, m = ys.shape[:2]
    flat = ys.reshape(-1, 3)
    owner = torch.arange(n, device=DEVICE).repeat_interleave(m)
    chunk = max(1024, (1 << 25) // n)
    out = []
    for i in range(0, len(flat), chunk):
        d = sdf_rows(cfg, s, flat[i:i + chunk])
        logout = torch.log1p(-torch.sigmoid(-d / cfg["loss"]["backward"]["union_sharpness"]).clamp(max=1 - 1e-6))
        cols = torch.arange(d.shape[1], device=DEVICE)
        out.append(logout.sum(0) - logout[owner[i:i + chunk], cols])  # exclude the sample's own SQ
    return torch.cat(out).exp().view(n, m)


def loss_fn(cfg: dict, sq: dict, x: torch.Tensor, grid: DistanceGrid, dirs: torch.Tensor):
    F, B = cfg["loss"]["forward"], cfg["loss"]["backward"]
    s = rows(sq, torch.nonzero(sq["alive"]).squeeze(1))
    f, _ = forward_value(cfg, sdf_rows(cfg, s, x))
    fwd = f.mean()
    ys = surface_samples(s, dirs)                                                # (n, S, 3)
    v = union_weight(cfg, s, ys)
    bwd = (v * torch.relu(grid(ys) - B["delta"])).mean()
    total = F["weight"] * fwd + B["weight"] * bwd
    return total, {"loss/forward": fwd.item(), "loss/backward": bwd.item(), "loss/total": total.item(),
                   "batch/union_visible": v.mean().item()}


# ---------------------------------------------------------------- full passes

@torch.no_grad()
def point_pass(cfg: dict, sq: dict, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward value and owner slot of every point."""
    ids = torch.nonzero(sq["alive"]).squeeze(1)
    s = rows(sq, ids)
    chunk = max(4096, (1 << 25) // len(ids))
    fs, owners = [], []
    for i in range(0, len(x), chunk):
        f, m = forward_value(cfg, sdf_rows(cfg, s, x[i:i + chunk]))
        fs.append(f)
        owners.append(ids[m])
    return torch.cat(fs), torch.cat(owners)


@torch.no_grad()
def surface_pass(cfg: dict, sq: dict, grid: DistanceGrid, n_dirs: int = 256) -> dict:
    """Per slot (NaN for free slots): backward value b_j = mean_s v * relu(u - delta), visible weight, air weight."""
    ids = torch.nonzero(sq["alive"]).squeeze(1)
    s = rows(sq, ids)
    ys = surface_samples(s, fibonacci_sphere(n_dirs).to(DEVICE))
    v = union_weight(cfg, s, ys)
    u = grid(ys)
    out = {}
    for key, val in (("b", (v * torch.relu(u - cfg["loss"]["backward"]["delta"])).mean(1)),
                     ("vu", (v * u).sum(1)), ("v", v.sum(1)), ("air", (v * (u > cfg["metrics"]["air_delta"])).sum(1))):
        full = torch.full((len(sq["alive"]),), float("nan"), device=DEVICE)
        full[ids] = val
        out[key] = full
    return out


# ---------------------------------------------------------------- top-down splitting

def clusters(p: torch.Tensor, voxel: float, min_points: int) -> list[torch.Tensor]:
    """Indices (into p) of the 26-connected components of the points' voxels with >= min_points, largest first."""
    from scipy.ndimage import label
    q = p.cpu().numpy()
    key = np.floor((q - q.min(0)) / voxel).astype(np.int64)
    occ = np.zeros(key.max(0) + 1, bool)
    occ[key[:, 0], key[:, 1], key[:, 2]] = True
    lab, _ = label(occ, structure=np.ones((3, 3, 3), bool))
    pl = lab[key[:, 0], key[:, 1], key[:, 2]]
    size = np.bincount(pl)
    order = np.argsort(pl, kind="stable")
    start = np.cumsum(size) - size
    return [torch.from_numpy(order[start[c]:start[c] + size[c]]).to(p.device)
            for c in np.argsort(-size) if size[c] >= min_points]


def dbscan6d(pos: torch.Tensor, nrm: torch.Tensor, eps: float, w: float, min_pts: int) -> np.ndarray:
    """DBSCAN on [x, w * n]: core points have >= min_pts neighbors (incl. themselves) within eps; clusters are the
    connected components of core points; border points join a core neighbor's cluster; noise gets -1."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    f = torch.cat([pos, w * torch.nn.functional.normalize(nrm, dim=1)], 1).cpu().numpy()
    P = len(f)
    pairs = cKDTree(f).query_pairs(eps, output_type="ndarray")
    core = np.bincount(pairs.ravel(), minlength=P) + 1 >= min_pts
    cc = pairs[core[pairs[:, 0]] & core[pairs[:, 1]]]
    _, c = connected_components(coo_matrix((np.ones(len(cc)), (cc[:, 0], cc[:, 1])), shape=(P, P)), directed=False)
    c[~core] = -1
    bp = pairs[core[pairs[:, 0]] != core[pairs[:, 1]]]
    b_core = np.where(core[bp[:, 0]], bp[:, 0], bp[:, 1])
    b_pt = np.where(core[bp[:, 0]], bp[:, 1], bp[:, 0])
    c[b_pt] = c[b_core]
    return c


class Splitter:
    """Every `every` steps: (1) judge the splits proposed at the previous check, (2) propose new ones.

    Proposal "halve": every alive SQ j (not cooling down) whose owned points have a mean forward error > tolerance is
    split (largest total error first while free slots last): 2-means of its owned points, box fit per half; the halves
    inherit the parent's Adam second moments.
    Proposal "residual": every alive SQ j owning >= min_points points with forward error > residual_threshold gets a
    new SQ f, box-fitted to the largest connected cluster C of those points (voxels of cluster_voxel); j is untouched.
    Judged on C: gain = mean_C f(shadow) - mean_C f(main) - b_f - (b_j(main) - b_j(shadow)). Verification: a shadow copy trains on the same batches without the
    proposals; with R the points j owned at the proposal,
        L_main  = mean_R f(main)   + (b_j + b_f) / 2      L_shadow = mean_R f(shadow) + b_j(shadow)
    keep if L_main < L_shadow - tolerance, else restore slots j and f from the shadow and let j cool down."""

    NAMES = ("translation", "log_scale", "eps_raw", "rotation")

    def __init__(self, cfg: dict, model: SQPool, opt, g: torch.Generator):
        self.cfg, self.g, self.model, self.opt = cfg, g, model, opt
        self.cooldown = torch.zeros(len(model.alive), dtype=torch.long, device=DEVICE)
        self.check = 0
        self.pending = []  # (parent j, new slot f, region R)
        self.shadow, self.shadow_opt = None, None
        if cfg["split"]["verify"]:
            self.shadow = copy.deepcopy(model)
            self.shadow_opt = torch.optim.Adam([{"params": [getattr(self.shadow, n)], "lr": gr["lr"]}
                                                for n, gr in zip(self.NAMES, opt.param_groups)])

    def due(self, step: int) -> bool:
        s = self.cfg["split"]
        last = s["stop"] + (s["every"] if s["verify"] else 0)
        return s["enabled"] and s["start"] <= step <= last and (step - s["start"]) % s["every"] == 0

    def _params(self, m):
        return [getattr(m, n) for n in self.NAMES]

    @torch.no_grad()
    def _copy(self, src, src_opt, dst, dst_opt, idx=None):
        """Copy parameters, alive mask and Adam state (all slots, or the given slots) from src to dst."""
        for ps, pd in zip(self._params(src), self._params(dst)):
            if idx is None:
                pd.data.copy_(ps.data)
            else:
                pd.data[idx] = ps.data[idx]
            ss = src_opt.state.get(ps)
            if not ss:
                continue
            sd = dst_opt.state.setdefault(pd, {})
            for key, v in ss.items():
                if idx is None or key == "step":
                    sd[key] = v.clone()
                else:
                    sd[key][idx] = v[idx]
        if idx is None:
            dst.alive.copy_(src.alive)
        else:
            dst.alive[idx] = src.alive[idx]

    @staticmethod
    def _spawn(model, opt, j: int, f: int, pts: torch.Tensor, ini: dict):
        """Slot f becomes a new SQ box-fitted to pts, with parent j's eps and Adam second moments."""
        t, Rm, a = box_fit(pts, ini, model.scale_min)
        model.translation.data[f] = t
        model.rotation.data[f] = matrix_to_quat(Rm[None])[0]
        model.log_scale.data[f] = a.clamp(model.scale_min, model.scale_max).log()
        model.eps_raw.data[f] = model.eps_raw.data[j]
        model.alive[f] = True
        for p in (model.translation, model.log_scale, model.eps_raw, model.rotation):
            st = opt.state.get(p)
            if st:
                st["exp_avg"][f] = 0
                st["exp_avg_sq"][f] = st["exp_avg_sq"][j]

    @property
    def shadow_active(self) -> bool:
        return bool(self.pending)

    def train_shadow(self, x: torch.Tensor, grid: DistanceGrid, dirs: torch.Tensor):
        for gs, gm in zip(self.shadow_opt.param_groups, self.opt.param_groups):
            gs["lr"] = gm["lr"]
        loss, _ = loss_fn(self.cfg, self.shadow(), x, grid, dirs)
        self.shadow_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.shadow_opt.step()

    @torch.no_grad()
    def step(self, step: int, scene: dict, grid: DistanceGrid) -> dict:
        S, ini, tol = self.cfg["split"], self.cfg["init"], self.cfg["split"]["tolerance"]
        model, opt, x = self.model, self.opt, scene["pos"]
        self.check += 1
        f_main, owner = point_pass(self.cfg, model(), x)
        kept, undone, gains, touched = 0, 0, [], torch.zeros_like(model.alive)
        if self.pending:
            b_main = surface_pass(self.cfg, model(), grid)["b"]
            f_sh, _ = point_pass(self.cfg, self.shadow(), x)
            b_sh = surface_pass(self.cfg, self.shadow(), grid)["b"]
            restore, kept_parent, rejected_parent = [], set(), set()
            for j, f, R, kind in self.pending:
                if kind == "residual":  # on the cluster: better explained points - new surface - parent surface change
                    gain = (f_sh[R].mean() - f_main[R].mean() - b_main[f] - (b_main[j] - b_sh[j])).item()
                else:
                    gain = (f_sh[R].mean() + b_sh[j] - f_main[R].mean() - (b_main[j] + b_main[f]) / 2).item()
                gains.append(gain)
                if gain > tol:
                    kept += 1
                    kept_parent.add(j)
                    continue
                undone += 1
                if kind == "residual":  # the parent stays as trained; only the new SQ is removed
                    model.alive[f] = False
                    rejected_parent.add(j)
                    continue
                restore += [j, f]
                self.cooldown[j] = self.check + S["retry_cooldown"]
                touched[j] = touched[f] = True
            for j in rejected_parent - kept_parent:
                self.cooldown[j] = self.check + S["retry_cooldown"]
                touched[j] = True
            if restore:
                self._copy(self.shadow, self.shadow_opt, model, opt, torch.tensor(restore, device=DEVICE))
            self.pending = []
        if self.shadow is not None:
            self._copy(model, opt, self.shadow, self.shadow_opt)  # shadow = state without the new proposals

        n_split = 0
        if step <= S["stop"]:
            N = len(model.alive)
            count = torch.bincount(owner, minlength=N)
            residual = S.get("proposal", "halve") == "residual"
            if residual:  # candidates: SQs owning >= min_points badly explained points; ranked by their total error
                bad = f_main > S["residual_threshold"]
                n_bad = torch.bincount(owner[bad], minlength=N)
                err = torch.zeros(N, device=DEVICE).index_add_(0, owner[bad], f_main[bad])
                cand = model.alive & ~touched & (n_bad >= S["min_points"]) & (self.cooldown <= self.check)
            else:
                err = torch.zeros(N, device=DEVICE).index_add_(0, owner, f_main)
                mean_err = err / count.clamp_min(1)
                cand = model.alive & ~touched & (count >= S["min_points"]) & (mean_err > tol) & (self.cooldown <= self.check)
            cand = torch.nonzero(cand).squeeze(1)
            cand = cand[torch.argsort(err[cand], descending=True)]
            free = torch.nonzero(~model.alive).squeeze(1)
            order = torch.argsort(owner)
            start = torch.cumsum(count, 0) - count
            new_rows = []
            free_iter = iter(free.tolist())
            for j in cand.tolist():
                R = order[start[j]:start[j] + count[j]]
                if residual:  # parent untouched; a new SQ per connected cluster of its bad points (largest first)
                    B_ = R[f_main[R] > S["residual_threshold"]]
                    for C in clusters(x[B_], S["cluster_voxel"], S["min_points"])[:S["max_clusters_per_parent"]]:
                        f = next(free_iter, None)
                        if f is None:
                            break
                        C = B_[C]
                        self._spawn(model, opt, j, f, x[C], ini)
                        n_split += 1
                        if self.shadow is not None:
                            self.pending.append((j, f, C, "residual"))
                    continue
                f = next(free_iter, None)
                if f is None:
                    break
                P = x[R]
                if len(P) > ini["kmeans_max_points"]:
                    P = P[torch.randperm(len(P), device=DEVICE, generator=self.g)[:ini["kmeans_max_points"]]]
                lab = kmeans(P, 2, S["kmeans_iters"], self.g)
                parts = [P[lab == 0], P[lab == 1]]
                if min(len(q) for q in parts) < 4:
                    continue
                for slot, Q in zip((j, f), parts):
                    t, Rm, a = box_fit(Q, ini, model.scale_min)
                    model.translation.data[slot] = t
                    model.rotation.data[slot] = matrix_to_quat(Rm[None])[0]
                    model.log_scale.data[slot] = a.clamp(model.scale_min, model.scale_max).log()
                model.eps_raw.data[f] = model.eps_raw.data[j]
                model.alive[f] = True
                new_rows.append((j, f))
                n_split += 1
                if self.shadow is not None:
                    self.pending.append((j, f, R, "halve"))
            for p in self._params(model):  # halves inherit the parent's second moments
                st = opt.state.get(p)
                if st and new_rows:
                    jj = torch.tensor([r[0] for r in new_rows], device=DEVICE)
                    ff = torch.tensor([r[1] for r in new_rows], device=DEVICE)
                    st["exp_avg"][jj] = 0
                    st["exp_avg"][ff] = 0
                    st["exp_avg_sq"][ff] = st["exp_avg_sq"][jj]
        out = {"split/splits": n_split, "split/kept": kept, "split/undone": undone,
               "split/n_sqs": int(model.alive.sum().item())}
        if gains:
            gq = torch.tensor(gains).quantile(torch.tensor([0.1, 0.5, 0.9])).tolist()
            out.update({"split/gain_p10": gq[0], "split/gain_median": gq[1], "split/gain_p90": gq[2],
                        "split/gain_positive": sum(x > 0 for x in gains) / len(gains)})
        return out


# ---------------------------------------------------------------- evaluation

def bin_names(bins: list[int]) -> list[str]:
    edges = ["0"] + [f"{b // 1000}k" for b in bins]
    return [f"{a}-{b}" for a, b in zip(edges, edges[1:])] + [f"{edges[-1]}+"]


@torch.no_grad()
def evaluate(cfg: dict, sq: dict, scene: dict, grid: DistanceGrid) -> dict:
    """Geometry on all points + GT diagnostics: every SQ takes the majority GT instance of its owned points."""
    x, y = scene["pos"], scene["y"]
    n_inst, N = len(scene["instance_ids"]), len(sq["alive"])
    f, owner = point_pass(cfg, sq, x)
    sp = surface_pass(cfg, sq, grid)
    alive = sq["alive"]
    v_sum = sp["v"][alive].sum().clamp_min(1e-9)

    nk = torch.bincount(owner * n_inst + y, minlength=N * n_inst).view(N, n_inst).float()
    owned = nk.sum(1)
    major = nk.argmax(1)
    used = alive & (owned > 0)
    sq_purity = nk.max(1).values / owned.clamp_min(1)
    pred = major[owner]
    conf = torch.bincount(y * n_inst + pred, minlength=n_inst * n_inst).view(n_inst, n_inst).float()
    iou = conf.diag() / (conf.sum(0) + conf.sum(1) - conf.diag()).clamp_min(1)
    per_inst = torch.bincount(major[used], minlength=n_inst).float()
    count = torch.bincount(y, minlength=n_inst).float()

    m = {
        "eval/n_sqs": int(alive.sum().item()),
        "eval/forward_mean": f.mean().item(),
        "eval/forward_median": f.median().item(),
        "eval/forward_p90": f.quantile(0.9).item() if len(f) < (1 << 24) else f[::4].quantile(0.9).item(),
        "eval/backward_mean": (sp["vu"][alive].sum() / v_sum).item(),
        "eval/surface_in_air": (sp["air"][alive].sum() / v_sum).item(),
        "eval/oracle_miou": iou.mean().item(),
        "eval/point_purity": (pred == y).float().mean().item(),
        "eval/sq_purity_mean": sq_purity[used].mean().item(),
        "eval/instances_without_sq": (per_inst == 0).float().mean().item(),
        "eval/sqs_per_instance": per_inst.mean().item(),
    }
    b = torch.bucketize(count, torch.tensor(cfg["metrics"]["size_bins"], device=DEVICE, dtype=torch.float), right=True)
    for i, name in enumerate(bin_names(cfg["metrics"]["size_bins"])):
        sel = b == i
        if sel.any():
            m[f"bins/oracle_iou_{name}"] = iou[sel].mean().item()
            m[f"bins/sqs_{name}"] = per_inst[sel].mean().item()
    return {"metrics": m, "iou": iou.cpu().numpy(), "sq_label": major.cpu().numpy(),
            "purity": sq_purity.cpu().numpy(), "active": alive.cpu().numpy(),
            "owned": owned.long().cpu().numpy()}


# ---------------------------------------------------------------- main loop

def run(cfg: dict, name: str):
    torch.manual_seed(cfg["seed"])
    g = torch.Generator(device=DEVICE).manual_seed(cfg["seed"])
    scene = load_scene_with_normals(cfg)
    x_all = scene["pos"]
    t0 = time.perf_counter()
    ini, smin = cfg["init"], cfg["parametrization"]["scale"]["min"]
    if ini.get("method", "kmeans") == "dbscan6d":
        D = ini["dbscan"]
        c = dbscan6d(x_all, scene["nrm"], D["eps"], D["normal_weight"], D["min_pts"])
        size = np.bincount(c[c >= 0])
        boxes = [box_fit(x_all[torch.from_numpy(np.nonzero(c == k)[0]).to(DEVICE)], ini, smin)
                 for k in np.nonzero(size >= D["min_cluster"])[0]]
        print(f"DBSCAN 6D: {len(boxes)} clusters >= {D['min_cluster']} points, noise {(c < 0).mean():.2f}, "
              f"in small clusters {np.isin(c, np.nonzero(size < D['min_cluster'])[0]).mean():.2f}", flush=True)
    else:
        P = x_all[torch.randperm(len(x_all), device=DEVICE, generator=g)[:ini["kmeans_max_points"]]]
        lab = kmeans(P, ini.get("n_sqs", 1), ini.get("kmeans_iters", 25), g)
        boxes = [box_fit(P[lab == m], ini, smin) for m in range(ini.get("n_sqs", 1)) if (lab == m).sum() >= 4]
    model = SQPool(cfg, *(torch.stack(v) for v in zip(*boxes))).to(DEVICE)
    B = cfg["loss"]["backward"]
    grid = DistanceGrid(x_all, B["grid_res"], B["grid_pad"])
    print(f"{name}: {len(x_all):,} points, {cfg['pool']['max_sqs']} slots; grid {tuple(grid.grid.shape[2:])} "
          f"in {time.perf_counter() - t0:.1f} s", flush=True)
    dirs0 = fibonacci_sphere(B["samples_per_sq"]).to(DEVICE)
    lr = cfg["optim"]["lr"]
    opt = torch.optim.Adam([
        {"params": [model.translation], "lr": lr["translation"]},
        {"params": [model.log_scale], "lr": lr["log_scale"]},
        {"params": [model.eps_raw], "lr": lr["eps"]},
        {"params": [model.rotation], "lr": lr["rotation"]},
    ])
    steps = cfg["optim"]["steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    splitter = Splitter(cfg, model, opt, g)

    out = RESULTS / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    trackio.init(project=PROJECT, name=name, config=cfg, auto_log_gpu=False, auto_log_cpu=False)

    snaps, history = [], []

    def snapshot(step):
        with torch.no_grad():
            sq = model()
        ev = evaluate(cfg, sq, scene, grid)
        snaps.append({"step": step, **{k: sq[k].detach().cpu().numpy() for k in ("scale", "eps", "rot", "trans", "existence")},
                      "sq_label": ev["sq_label"], "purity": ev["purity"], "active": ev["active"], "owned": ev["owned"]})
        return ev

    t_start = time.perf_counter()
    for step in range(steps + 1):
        is_eval = step % cfg["logging"]["eval_every"] == 0 or step == steps
        if step % cfg["logging"]["snapshot_every"] == 0 or is_eval:
            ev = snapshot(step) if step % cfg["logging"]["snapshot_every"] == 0 or step == steps else \
                evaluate(cfg, model(), scene, grid)
            if is_eval:
                history.append({"step": step, **ev["metrics"]})
                trackio.log(ev["metrics"], step=step)
                m = ev["metrics"]
                print(f"step {step:5d}  SQs {m['eval/n_sqs']:4d}  fwd {100 * m['eval/forward_mean']:.2f} cm  "
                      f"bwd {100 * m['eval/backward_mean']:.2f} cm  in air {m['eval/surface_in_air']:.3f}  "
                      f"oracle mIoU {m['eval/oracle_miou']:.3f}  purity {m['eval/point_purity']:.3f}", flush=True)
        if step == steps:
            break
        if splitter.due(step):
            slog = splitter.step(step, scene, grid)
            trackio.log(slog, step=step)
            gain = (f"  gain p10/50/90 {1000 * slog['split/gain_p10']:.1f} / {1000 * slog['split/gain_median']:.1f} / "
                    f"{1000 * slog['split/gain_p90']:.1f} mm") if "split/gain_median" in slog else ""
            print(f"step {step:5d}  split: {slog['split/kept']} kept, {slog['split/undone']} undone, "
                  f"{slog['split/splits']} new -> {slog['split/n_sqs']} SQs{gain}", flush=True)
        t = time.perf_counter()
        idx = torch.randint(len(x_all), (cfg["sampling"]["points_per_step"],), device=DEVICE, generator=g)
        dirs = dirs0 @ random_rotation(g).T
        loss, logs = loss_fn(cfg, model(), x_all[idx], grid, dirs)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if splitter.shadow_active:  # same batch, same learning rate, without the trial splits
            splitter.train_shadow(x_all[idx], grid, dirs)
        sched.step()
        torch.cuda.synchronize()
        if step % 10 == 0:
            logs.update({"time/step_ms": 1000 * (time.perf_counter() - t), "lr/translation": sched.get_last_lr()[0]})
            trackio.log(logs, step=step)
    trackio.finish()

    elapsed = time.perf_counter() - t_start
    keys = [k for k in snaps[0] if k != "step"]
    np.savez_compressed(out / "snapshots.npz", step=np.array([s["step"] for s in snaps]),
                        **{k: np.stack([s[k] for s in snaps]) for k in keys},
                        instance_ids=scene["instance_ids"], R=scene["R"], t=scene["t"])
    snap, big = out / "snapshots.npz", RESULTS / "large_files" / "runs" / name / "snapshots.npz"
    big.unlink(missing_ok=True)
    if snap.stat().st_size > 1e6:  # >1MB results go to results/large_files (not versioned)
        big.parent.mkdir(parents=True, exist_ok=True)
        snap.replace(big)
    (out / "metrics.json").write_text(json.dumps({"name": name, "seconds": elapsed, "final": history[-1],
                                                  "history": history, "final_iou": ev["iou"].tolist()}, indent=1))
    print(f"done in {elapsed:.0f} s -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(EXP / "config.yaml"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--set", action="append", default=[], help="override, e.g. split.tolerance=0.01")
    args = ap.parse_args()
    cfg = load_config(Path(args.config), args.set)
    name = args.name or f"{cfg['data']['scene']}_global_{time.strftime('%m%d-%H%M%S')}"
    run(copy.deepcopy(cfg), name)


if __name__ == "__main__":
    main()
