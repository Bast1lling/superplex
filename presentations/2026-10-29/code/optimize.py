"""optimize: SuperFlex's test-time optimization (papers/superflex) on every GT instance, from k-means or DBSCAN boxes.

    python optimize.py [--res 0.02] [scene ...]          (needs network.py first: the network's primitive counts)
    python optimize.py --views [--res 0.02] [scene ...]  (fits from rendered RGBD views, see views.py)
    python optimize.py --whole [--res 0.02] [scene ...]  (the whole scene at once, without instance labels)
    python optimize.py --views-whole [--res 0.02] [scene ...]  (the same from 100 rendered RGBD / RGB views)

Per instance (as SuperFlex on ShapeNet): 4096 of its points (surface) and 100k query points uniform in its box + 10 %,
inside = a voxel of this instance (other instances and free voxels: outside; DEEP voxels take the instance of the
nearest solid voxel). SuperFlex's parametrization, optimizer and 1000 steps (best state per instance kept).
Loss terms (SuperFlex's): volume = -log soft IoU of the query points, surface = truncated signed distance of the
surface points, plus a bounding-box term (primitive poles inside the instance box) and 0.1 * overlap.

Results (one per variant):
- kmeans_surface / kmeans_volume / kmeans_full: k-means boxes (as many as the network predicts for the instance),
  surface term only / volume term only / all four terms; plain SQs.
- kmeans_full_deform: as kmeans_full, with SuperFlex's tapering and bending.
- dbscan_full: count and boxes from DBSCAN clusters of (position, normal) instead of the network; all terms, plain.
- views_<multi|f800|rgb_multi|rgb_f800>: as kmeans_full, but everything from rendered views (100 views / frame 800;
  RGBD, or RGB with Depth Pro's depth, see views.py): surface
  points = back-projected pixels with the instance's label, query points labelled by the grid carved from the views,
  network count from the view points. Single view: the never-seen space (DEEP) is left out of the IoU instead of
  being filled, and the bounding-box term (box of the visible points only) is dropped.
- scene_kmeans_net / scene_kmeans_dbscan: no instance labels, the whole scene is one object (all solid voxels inside):
  k-means boxes of the scene's points, as many as the network predicts summed over all instances / as DBSCAN finds
  clusters of (position in m, normal) in the scene; all terms, plain. SuperFlex's 100k query / 4096 surface points per
  object become 500k / 50k, and its normalization to unit size a fixed 2 m (its temperatures and truncation are in
  normalized units: for a 2 m object as on a typical instance, not 10x coarser for a whole room).
- views_scene_multi / views_scene_rgb_multi: the DBSCAN variant from 100 views (RGBD / RGB with Depth Pro), labels
  not used: carving without labels, all back-projected pixels, normals from the depth images.
IoU (saved per instance): hard IoU of the union of an instance's SQs on its GT query points (from the 3D GT); for the
whole-scene fits one IoU of all SQs with all solid space.
"""

import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.ndimage import distance_transform_edt
from sklearn.cluster import DBSCAN, KMeans

import network  # puts SuperFlex on the path (with its C++20 build fix)
import views
from data import DEEP, DEV, cli, load_points, result_path, voxelize
from superflex.optimization import SuperQOptimization
from superflex.utils.predictions_handler import PredictionHandler

N_SURF, N_QUERY, SLOTS, BATCH, STEPS = 4096, 100_000, 16, 16, 1000
N_SURF_SCENE, N_QUERY_SCENE, SCENE_SCALE = 50_000, 500_000, 2.0
FULL = ("volume", "surface", "bbox", "overlap")
VARIANTS = {  # name: (init, loss terms, tapering + bending)
    "kmeans_surface": ("kmeans", ("surface",), False),
    "kmeans_volume": ("kmeans", ("volume",), False),
    "kmeans_full": ("kmeans", FULL, False),
    "kmeans_full_deform": ("kmeans", FULL, True),
    "dbscan_full": ("dbscan", FULL, False),
}


def instances(pts, grid, fill_deep=True, seed=0):
    """Per instance with points: id, center, centered surface points (+ normals) and query points, inside / outside
    of the query points; without fill_deep also which query points are known (not DEEP)."""
    rng = np.random.default_rng(seed)
    L = grid.label_np
    if fill_deep:
        _, near = distance_transform_edt(L < 0, return_indices=True)
        L = np.where(L == DEEP, L[tuple(near)], L)
    out = []
    for k in range(pts["inst"].max() + 1):
        m = pts["inst"] == k
        if m.sum() < 4:
            continue
        i = rng.choice(m.sum(), N_SURF, replace=m.sum() < N_SURF)
        p, n = pts["pos"][m][i].astype(np.float64), pts["nrm"][m][i] if "nrm" in pts else None
        lo, hi = p.min(0), p.max(0)
        pad = 0.1 * (hi - lo).max()
        q = lo - pad + rng.random((N_QUERY, 3)) * (hi - lo + 2 * pad)
        v = np.round((q - grid.lo_np) / grid.res).astype(int)
        inside = ((v >= 0) & (v < L.shape)).all(1)
        v = np.minimum(np.maximum(v, 0), np.array(L.shape) - 1)
        lab = np.where(inside, L[tuple(v.T)], -1)
        if (lab == k).any():  # skip instances never solid in the grid (no IoU target)
            c = p.mean(0)
            out.append({"k": k, "c": c, "surf": p - c, "nrm": n, "q": q - c, "occ": lab == k, "known": lab != DEEP})
    return out


def kmeans_labels(d, count):
    return KMeans(count, n_init=1, random_state=0).fit_predict(d["surf"])


def dbscan_labels(d, w_nrm=1.0, eps=0.15, min_size=40):
    """DBSCAN of (position / instance size, w_nrm * normal). Clusters with >= min_size of the 4096 points, at most
    the 16 largest, renumbered by size; other points -1. No such cluster: one box for all points."""
    x = d["surf"] / (2 * np.abs(d["surf"]).max())
    lab = DBSCAN(eps=eps, min_samples=10).fit_predict(np.hstack([x, w_nrm * d["nrm"]]))
    ids, size = np.unique(lab[lab >= 0], return_counts=True)
    keep = ids[size >= min_size][np.argsort(-size[size >= min_size])][:SLOTS]
    if not len(keep):
        return np.zeros(len(lab), int)
    new = np.full(lab.max() + 2, -1)
    new[keep] = np.arange(len(keep))
    return new[lab]  # lab == -1 -> new[-1] == -1


def boxes(data, labels, slots=SLOTS):
    """One PCA box per cluster (exponents 1) in `slots` slots; unused slots exist = 0."""
    B, S = len(data), slots
    p = {"scale": np.ones((B, S, 3)), "translation": np.zeros((B, S, 3)), "rotation": np.tile(np.eye(3), (B, S, 1, 1)),
         "exponents": np.ones((B, S, 2)), "exist": np.zeros((B, S)), "tapering": np.zeros((B, S, 2)),
         "bending": np.zeros((B, S, 6))}
    for b, (d, lab) in enumerate(zip(data, labels)):
        for j in range(lab.max() + 1):
            x = d["surf"][lab == j]
            if len(x) < 4:
                continue
            mu = x.mean(0)
            _, axes = np.linalg.eigh(np.cov((x - mu).T))
            axes[:, 2] *= np.sign(np.linalg.det(axes))
            q = (x - mu) @ axes
            p["translation"][b, j] = mu + axes @ ((q.min(0) + q.max(0)) / 2)
            p["scale"][b, j] = np.maximum((q.max(0) - q.min(0)) / 2, 0.005)
            p["rotation"][b, j], p["exist"][b, j] = axes, 1.0
    return p


class Objective(SuperQOptimization):
    """SuperFlex's loss restricted to the chosen terms; with `known` set, the soft IoU over the known query points."""
    known = None

    def compute_losses(self, out):
        _, iou, surface, bbox, overlap = self._compute_losses(out["sdfs"], out["overlap"])
        if self.known is not None:
            pred = torch.sigmoid(-out["sdfs"][:, :self.M_points_iou] / 1e-3) * self.known
            occ = self.occupancies.float() * self.known
            iou = (pred * occ).sum(1) / torch.clamp((pred + occ - pred * occ).sum(1), min=1.0)
        terms = {"volume": lambda: -torch.log(iou), "surface": lambda: surface, "bbox": lambda: bbox, "overlap": lambda: overlap}
        return sum(terms[t]() for t in self.terms), iou


class SceneObjective(Objective):
    """For the whole scene: more points than SuperFlex's fixed per-object numbers, fixed normalization scale."""
    M_points_iou = property(lambda self: N_QUERY_SCENE, lambda self, v: None)  # SuperFlex assigns 100k / 4096
    M_points_surf = property(lambda self: N_SURF_SCENE, lambda self, v: None)

    def normalize(self):  # SuperFlex's normalize, with SCENE_SCALE instead of the points' extent
        t = self.points[:, -self.M_points_surf:].mean(1)
        s = torch.full((len(t),), SCENE_SCALE, device=t.device)
        self.points = (self.points - t[:, None]) / s[:, None, None]
        self.normalization_translation, self.normalization_scale = t, s
        with torch.no_grad():
            self.translation.data = (self.translation.data - t[:, None]) / s[:, None, None]
            self.raw_scale.data = torch.log(torch.clamp(self.scale() / s[:, None, None] - self.minS, min=1e-6))


@torch.no_grad()
def gt_iou(sq, data, ref):
    """Hard IoU of each instance's union of SQs on its GT query points (ref: instances() of the 3D GT)."""
    x = torch.tensor(np.stack([r["q"] + r["c"] - d["c"] for d, r in zip(data, ref)]), dtype=torch.float32, device=DEV)
    x = (x - sq.normalization_translation[:, None]) / sq.normalization_scale[:, None, None]
    occ = torch.cat([union_inside(sq, x[:, a:a + 100_000]) for a in range(0, x.shape[1], 100_000)], 1)
    gt = torch.tensor(np.stack([r["occ"] for r in ref]), device=DEV)
    return ((occ & gt).sum(1) / (occ | gt).sum(1).clamp_min(1)).cpu().numpy()


def union_inside(sq, x):
    s = sq.sdf_batch(x.transpose(1, 2).contiguous())  # (B, primitives, points)
    s[~sq.exist_mask[:, :, None].expand_as(s)] = float("inf")
    return s.min(1).values < 0


def optimize(data, params, terms, deform, ref, masked=False, cls=Objective):
    """SuperFlex's loop on one batch of instances; returns the primitives (scene frame) and the GT IoU per instance."""
    B, slots = len(data), params["exist"].shape[1]
    handler = PredictionHandler({"names": np.arange(B), "pc": np.stack([d["surf"] for d in data]),
                                 "assign_matrix": np.zeros((B, 1, slots)), **params})
    ext = {"points": torch.tensor(np.stack([d["surf"] for d in data]), dtype=torch.float32, device=DEV),
           "normals": torch.zeros(B, len(data[0]["surf"]), 3, device=DEV),
           "points_iou": torch.tensor(np.stack([d["q"] for d in data]), dtype=torch.float32, device=DEV),
           "occupancies": torch.tensor(np.stack([d["occ"] for d in data]), device=DEV)}
    sq = cls(pred_handler=handler, indices=list(range(B)), device=DEV, external_data=ext,
                   cfg=OmegaConf.create({"tapering": deform, "bending": deform, "pruning": False}))
    sq.terms = terms
    if masked:
        sq.known = torch.tensor(np.stack([d["known"] for d in data]), device=DEV).float()
    opt = torch.optim.Adam(sq.get_param_groups())
    names = ["raw_scale", "raw_exponents", "raw_rotation", "translation", "raw_tapering", "raw_bending"]
    best, best_p = torch.full((B,), float("inf"), device=DEV), {n: getattr(sq, n).detach().clone() for n in names}
    for _ in range(STEPS):
        opt.zero_grad()
        loss, _ = sq.compute_losses(sq.forward())
        loss.mean().backward()
        opt.step()
        with torch.no_grad():
            better = loss < best
            best = torch.where(better, loss, best)
            for n in names:
                best_p[n][better] = getattr(sq, n)[better]
    with torch.no_grad():
        for n in names:
            getattr(sq, n).copy_(best_p[n])
    iou = gt_iou(sq, data, ref)
    sq.update_handler(compute_meshes=False)
    s = sq.normalization_scale.cpu().numpy()
    bending = handler.bending.copy()
    bending[..., 0::2] /= s[:, None, None]  # update_handler leaves the curvature in the normalized frame
    b, j = np.nonzero(handler.exist > 0.5)
    c = np.stack([d["c"] for d in data])
    sqs = {"trans": handler.translation[b, j] + c[b], "scale": handler.scale[b, j], "eps": handler.exponents[b, j],
           "rot": handler.rotation[b, j], "tapering": handler.tapering[b, j], "bending": bending[b, j], "instance": b}
    return sqs, iou


def fit(scene, name, data, labels, terms, deform, ref, masked=False):
    """Optimizes all instances (in batches) from the boxes of their point clusters and saves the result."""
    params = boxes(data, labels)
    ks = np.array([d["k"] for d in data])
    parts, ious = [], []
    for a in range(0, len(data), BATCH):
        sl = slice(a, a + BATCH)
        sqs, iou = optimize(data[sl], {k: v[sl] for k, v in params.items()}, terms, deform, ref[sl], masked)
        sqs["instance"] = ks[sl][sqs["instance"]]
        parts.append(sqs), ious.append(iou)
    sqs = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} | {"iou": np.concatenate(ious), "iou_instance": ks}
    out = result_path(scene, name)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **sqs)
    print(f"{scene} {name}: {len(sqs['trans'])} SQs for {len(ks)} instances, mean IoU {sqs['iou'].mean():.3f} -> {out}", flush=True)


def main(scene):
    pts = load_points(scene)
    data = instances(pts, voxelize(pts))
    counts = np.bincount(np.load(result_path(scene, "superflex_robust"))["instance"], minlength=pts["inst"].max() + 1).clip(1)
    for name, (init, terms, deform) in VARIANTS.items():
        labels = [kmeans_labels(d, counts[d["k"]]) if init == "kmeans" else dbscan_labels(d) for d in data]
        fit(scene, name, data, labels, terms, deform, data)


def scene_object(pts, grid, seed=0):
    """The whole scene as one object: N_SURF_SCENE of its points (+ normals), query points in their box, inside =
    any solid or DEEP voxel."""
    rng = np.random.default_rng(seed)
    L = grid.label_np
    i = rng.choice(len(pts["pos"]), N_SURF_SCENE, replace=False)
    p, n = pts["pos"][i].astype(np.float64), pts["nrm"][i]
    lo, hi = p.min(0), p.max(0)
    q = lo + rng.random((N_QUERY_SCENE, 3)) * (hi - lo)
    v = np.round((q - grid.lo_np) / grid.res).astype(int)
    inside = ((v >= 0) & (v < L.shape)).all(1)
    lab = np.where(inside, L[tuple(np.minimum(np.maximum(v, 0), np.array(L.shape) - 1).T)], -1)
    c = p.mean(0)
    return {"c": c, "surf": p - c, "nrm": n, "q": q - c, "occ": (lab >= 0) | (lab == DEEP)}


def dbscan_count(d):
    """Clusters of (position in m, normal) with >= 20 of the scene's surface points."""
    lab = DBSCAN(eps=0.15, min_samples=10).fit_predict(np.hstack([d["surf"], d["nrm"]]))
    return int((np.unique(lab[lab >= 0], return_counts=True)[1] >= 20).sum())


def fit_scene(scene, name, d, count, ref):
    params = boxes([d], [kmeans_labels(d, count)], count)
    sqs, iou = optimize([d], params, FULL, False, [ref], cls=SceneObjective)
    sqs["instance"] = np.arange(len(sqs["trans"]))  # no instances: every SQ its own color
    out = result_path(scene, name)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **sqs, iou=iou)
    print(f"{scene} {name}: {len(sqs['trans'])} SQs (count {count}), scene IoU {iou[0]:.3f} -> {out}", flush=True)


def main_whole(scene):
    pts = load_points(scene)
    d = scene_object(pts, voxelize(pts))
    fit_scene(scene, "scene_kmeans_net", d, len(np.load(result_path(scene, "superflex_robust"))["instance"]), d)
    fit_scene(scene, "scene_kmeans_dbscan", d, dbscan_count(d), d)


def main_views_scene(scene):
    """Whole scene from 100 views, no labels: carved solid / free / DEEP, all pixels as surface points, DBSCAN count
    (normals from the depth images); evaluated against the 3D GT scene object."""
    gt = load_points(scene)
    ref = scene_object(gt, voxelize(gt))
    for setting, (frames, depth_source) in views.SCENE_SETTINGS.items():
        vs = views.load_views(scene, frames, gt, depth_source)
        d = scene_object(views.view_points(vs, labelled=False), views.carve(vs, 1, labelled=False))
        fit_scene(scene, f"views_{setting}", d, dbscan_count(d), ref)


def main_views(scene):
    gt = load_points(scene)
    ref = {d["k"]: d for d in instances(gt, voxelize(gt))}
    model = network.load_model("robust")
    for setting, (frames, depth_source) in views.SETTINGS.items():
        vs = views.load_views(scene, frames, gt, depth_source)
        pts, single = views.view_points(vs), len(frames) == 1
        data = [d for d in instances(pts, views.carve(vs, gt["inst"].max() + 1), fill_deep=not single) if d["k"] in ref]
        ks = [d["k"] for d in data]
        net = network.predict(model, *network.instance_batch(pts, ks))  # the network on the view points
        counts = np.bincount(net["instance"], minlength=len(ks)).clip(1)
        labels = [kmeans_labels(d, n) for d, n in zip(data, counts)]
        terms = tuple(t for t in FULL if t != "bbox") if single else FULL
        fit(scene, f"views_{setting}", data, labels, terms, False, [ref[k] for k in ks], masked=single)


if __name__ == "__main__":
    modes = {"--views": main_views, "--whole": main_whole, "--views-whole": main_views_scene, None: main}
    mode = next((m for m in modes if m in sys.argv), None)
    if mode:
        sys.argv.remove(mode)
    for scene in cli():
        modes[mode](scene)
