"""SuperSuperquadrics: groups of k superquadrics that form one smooth shape.

A k-set {SQ_1, ..., SQ_k} is blended with barycentric weights w on the (k-1)-simplex:
    scale, eps, trans   weighted average  sum_i w_i * p_i
    rotation            normalized weighted quaternion average. Before averaging, every node's
                        rotation is replaced by its symmetric equivalent (180 deg flips about
                        local axes leave an SQ unchanged) closest to node 0's, and quaternion
                        signs are aligned, so the blend does not twist unnecessarily.
The shape of a blend is the union over weights of SQ(w). For k=2 this is a sweep along the edge,
for k=3 a thickened triangle, ...

Two ways to turn a k-set into geometry:
    "blend"    one blend over the full (k-1)-simplex of weights
    "simplex"  for k > 4, Delaunay-tetrahedralize the k centers and blend each tetrahedron
               separately (identical to "blend" for k <= 4)
A connected component (k-sets sharing nodes) is the union of its blends.

The union has no closed form, so we evaluate the radial distance field
    d(x) = min_w  |p_w| * (1 - F_w(p_w)^(-e1_w / 2)),   p_w = R_w^T (x - c_w)
on a voxel grid and extract the zero level set with marching cubes. The min over w is found
per voxel: start from the better of (a) the best of a fixed set of weight samples and (b) the
weights whose blended center is closest to the voxel, then refine with adaptive projected
gradient steps on the weights.
"""

import itertools
import math

import numpy as np
import torch
from scipy.spatial import Delaunay, QhullError
from skimage.measure import marching_cubes

from superquadric import Superquadrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# the four rotations (det = +1 sign flips of local axes) under which an SQ is invariant
_FLIPS = torch.tensor([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=torch.float32)


def matrix_to_quat(r: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) rotation matrices -> (..., 4) unit quaternions (w, x, y, z)."""
    m = r.reshape(-1, 3, 3)
    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    cands = torch.stack([
        torch.stack([1 + tr, m[:, 2, 1] - m[:, 1, 2], m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] - m[:, 0, 1]], -1),
        torch.stack([m[:, 2, 1] - m[:, 1, 2], 1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], m[:, 0, 1] + m[:, 1, 0], m[:, 0, 2] + m[:, 2, 0]], -1),
        torch.stack([m[:, 0, 2] - m[:, 2, 0], m[:, 0, 1] + m[:, 1, 0], 1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2], m[:, 1, 2] + m[:, 2, 1]], -1),
        torch.stack([m[:, 1, 0] - m[:, 0, 1], m[:, 0, 2] + m[:, 2, 0], m[:, 1, 2] + m[:, 2, 1], 1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2]], -1),
    ], dim=1)  # (N, 4 candidates, 4); pick the numerically best-conditioned one
    best = cands.norm(dim=-1).argmax(dim=1)
    q = cands[torch.arange(len(m)), best]
    return (q / q.norm(dim=-1, keepdim=True)).reshape(*r.shape[:-2], 4)


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def simplex_lattice(k: int, n_max: int, seed: int = 0) -> torch.Tensor:
    """Weight samples on the (k-1)-simplex: the finest regular lattice with at most n_max points,
    topped up with random (Dirichlet) samples. Always contains the k vertices."""
    r = 1
    while k <= 6 and math.comb(r + 1 + k - 1, k - 1) <= n_max:
        r += 1
    pts = [c for c in itertools.product(range(r + 1), repeat=k) if sum(c) == r]
    w = torch.tensor(pts, dtype=torch.float32) / r
    n_rand = n_max - len(w)
    if n_rand > 0:
        g = torch.Generator().manual_seed(seed)
        e = -torch.log(torch.rand(n_rand, k, generator=g).clamp_min(1e-9))  # Dirichlet(1)
        w = torch.cat([w, e / e.sum(1, keepdim=True)])
    return w, 1.0 / r


class SimplexBlend:
    """Barycentric parameter blend of k superquadrics."""

    def __init__(self, sq: Superquadrics, nodes: list[int]):
        self.nodes = list(nodes)
        idx = torch.tensor(self.nodes)
        self.k = len(idx)
        self.scale = sq.scale[idx].to(DEVICE)
        self.eps = sq.eps[idx].to(DEVICE)
        self.trans = sq.trans[idx].to(DEVICE)
        rots = sq.rot[idx].to(DEVICE)
        r0 = rots[0]
        aligned = []
        for r in rots:
            cands = r[None] * _FLIPS.to(DEVICE)[:, None, :]  # flips columns = local axes
            aligned.append(cands[torch.einsum("ij,kij->k", r0, cands).argmax()])
        q = matrix_to_quat(torch.stack(aligned))
        q = q * torch.where((q * q[0]).sum(-1, keepdim=True) < 0, -1.0, 1.0)
        self.quat = q

    def params(self, w: torch.Tensor):
        """w: (N, k) -> per-sample scale (N,3), eps (N,2), rot (N,3,3), trans (N,3)."""
        q = w @ self.quat
        q = q / q.norm(dim=-1, keepdim=True)
        return w @ self.scale, w @ self.eps, quat_to_matrix(q), w @ self.trans

    def bounds(self):
        """Axis-aligned box containing every blend (an SQ with eps <= 2 lies in the ball of radius |scale|)."""
        r = self.scale.norm(dim=1).max()
        return self.trans.min(0).values - r, self.trans.max(0).values + r


def radial_distance(x, scale, eps, rot, trans):
    """Radial (pseudo-signed) distance of points x (N, 3) to SQs with per-point or shared params.
    Computed in log space so tiny eps (huge exponents) neither overflow nor produce NaNs."""
    p = torch.einsum("...ji,...j->...i", rot, x - trans)  # R^T (x - c)
    e1, e2 = eps[..., 0], eps[..., 1]
    logq = torch.log((p / scale).abs().clamp_min(1e-12))
    lxy = torch.logaddexp(2 / e2 * logq[..., 0], 2 / e2 * logq[..., 1]) * (e2 / e1)
    log_f = torch.logaddexp(lxy, 2 / e1 * logq[..., 2])
    return p.norm(dim=-1) * (1 - torch.exp(-e1 / 2 * log_f))


def project_simplex(v: torch.Tensor) -> torch.Tensor:
    """Euclidean projection of each row of v (N, k) onto the probability simplex."""
    u, _ = v.sort(dim=1, descending=True)
    css = u.cumsum(1) - 1
    ks = torch.arange(1, v.shape[1] + 1, device=v.device)
    rho = (u - css / ks > 0).sum(1, keepdim=True)
    theta = css.gather(1, rho - 1) / rho
    return (v - theta).clamp_min(0)


def center_init(blend: SimplexBlend, x: torch.Tensor, iters: int = 60, reg: float = 1e-2) -> torch.Tensor:
    """Weights whose blended center is closest to x (slightly regularized towards uniform so the
    choice is unique for k > 4), via projected gradient descent. The covering blend of a point is
    usually centered near it, so this is a much better start than any fixed sample for thin shapes."""
    c = blend.trans  # (k, 3)
    lip = 2 * (torch.linalg.eigvalsh(c @ c.T).max() + reg)
    w = torch.full((len(x), blend.k), 1 / blend.k, device=DEVICE)
    for _ in range(iters):
        grad = 2 * ((w @ c - x) @ c.T + reg * w)
        w = project_simplex(w - grad / lip)
    return w


def refine(blend: SimplexBlend, x, d, w, step0: float, n_rounds: int = 60):
    """Per-point projected gradient descent on the weights with an adaptive step: accept a step
    along the (L2-normalized) negative gradient if it lowers the distance and grow the step,
    otherwise shrink it."""
    step = torch.full((len(x),), step0, device=DEVICE)
    for _ in range(n_rounds):
        with torch.enable_grad():
            wg = w.detach().requires_grad_()
            g, = torch.autograd.grad(radial_distance(x, *blend.params(wg)).sum(), wg)
        g = torch.nan_to_num(g)
        trial = project_simplex(w - step[:, None] * g / (g.norm(dim=1, keepdim=True) + 1e-12))
        d_t = radial_distance(x, *blend.params(trial))
        ok = d_t < d
        d = torch.where(ok, d_t, d)
        w = torch.where(ok[:, None], trial, w)
        step = torch.where(ok, step * 1.5, step * 0.5).clamp(1e-4, 1.0)
    return d, w


@torch.no_grad()
def blend_distance(blend: SimplexBlend, x: torch.Tensor, n_coarse: int = 128, n_starts: int = 4,
                   chunk: int = 1 << 20):
    """min over simplex weights of the radial distance for points x (N, 3). Returns (dist, w_best).

    The landscape over w is multimodal, so for voxels that can be near the surface we run the
    gradient refinement from several starts: the closest-center weights plus the n_starts best
    fixed weight samples. Measured against brute force (80k samples) on random points in the
    bounding box, this misses ~0% of inside points for k <= 4 and ~3% for k = 6."""
    if blend.k == 2:
        # samples along the edge must be denser than the thinnest shape is thick
        length = float((blend.trans[0] - blend.trans[1]).norm())
        n_coarse = int(min(512, max(32, math.ceil(2 * length / float(blend.scale.min())))))
    w_coarse, spacing = simplex_lattice(blend.k, n_coarse)
    w_coarse = w_coarse.to(DEVICE)
    s_c, e_c, r_c, c_c = blend.params(w_coarse)
    if blend.k == 2:
        n_starts = 1  # 1D: handled by a dense local search below
    # generous bound on how much refinement can lower the coarse distance: a weight change of one
    # lattice step moves the blended center by at most spacing * (diameter of the node centers)
    diam = float(torch.cdist(blend.trans, blend.trans).max())
    band = 2 * spacing * diam + float(blend.scale.max())

    if blend.k == 2:
        # every blend lies within radius max|scale| of its center on the segment, so points further
        # away are outside: give them a positive lower-bound distance and skip the search
        a, ab = blend.trans[0], blend.trans[1] - blend.trans[0]
        t = (((x - a) @ ab) / (ab @ ab).clamp_min(1e-12)).clamp(0, 1)
        seg_d = (x - a - t[:, None] * ab).norm(dim=1)
        r_max = float(blend.scale.norm(dim=1).max())
        close = (seg_d <= r_max).nonzero().squeeze(1)
        dist = seg_d - r_max
        w_out = torch.stack([1 - t, t], 1)
        if len(close) < len(x):
            if len(close):
                dist[close], w_out[close] = blend_distance(blend, x[close], n_coarse, n_starts, chunk)
            return dist, w_out

    dist = torch.empty(len(x), device=DEVICE)
    w_out = torch.empty(len(x), blend.k, device=DEVICE)
    for i in range(0, len(x), chunk):
        xs = x[i:i + chunk]
        top_d = torch.full((len(xs), n_starts), torch.inf, device=DEVICE)
        top_j = torch.zeros((len(xs), n_starts), dtype=torch.long, device=DEVICE)
        for j in range(len(w_coarse)):  # keep the n_starts best samples per point
            d = radial_distance(xs, s_c[j], e_c[j], r_c[j], c_c[j])
            cand_d = torch.cat([top_d, d[:, None]], 1)
            cand_j = torch.cat([top_j, torch.full_like(top_j[:, :1], j)], 1)
            top_d, idx = cand_d.topk(n_starts, dim=1, largest=False)
            top_j = cand_j.gather(1, idx)
        best_d, best_w = top_d[:, 0].clone(), w_coarse[top_j[:, 0]]
        if blend.k == 2:
            # 1D: a dense local search around the best coarse t is exact enough and ~30x cheaper
            # than the gradient refinement
            for off in torch.linspace(-spacing, spacing, 16, device=DEVICE):
                t = (best_w[:, 1] + off).clamp(0, 1)
                w = torch.stack([1 - t, t], 1)
                d = radial_distance(xs, *blend.params(w))
                better = d < best_d
                best_d = torch.where(better, d, best_d)
                best_w = torch.where(better[:, None], w, best_w)
        elif blend.k > 2:
            near = (top_d[:, 0] < band).nonzero().squeeze(1)
            if len(near):
                xn = xs[near]
                wc = center_init(blend, xn)
                starts = [(radial_distance(xn, *blend.params(wc)), wc)]
                starts += [(top_d[near, q], w_coarse[top_j[near, q]]) for q in range(n_starts)]
                dn, wn = best_d[near], best_w[near]
                for d0, w0 in starts:
                    d, w = refine(blend, xn, d0, w0, spacing)
                    better = d < dn
                    dn = torch.where(better, d, dn)
                    wn = torch.where(better[:, None], w, wn)
                best_d[near], best_w[near] = dn, wn
        dist[i:i + chunk], w_out[i:i + chunk] = best_d, best_w
    return dist, w_out


def decompose(nodes: list[int], centers: np.ndarray) -> list[list[int]]:
    """Split a k-set into simplices with at most 4 nodes (Delaunay on the centers)."""
    if len(nodes) <= 4:
        return [list(nodes)]
    try:
        tets = Delaunay(centers).simplices
    except QhullError:  # (nearly) coplanar centers: triangulate in the best-fit plane
        c = centers - centers.mean(0)
        basis = np.linalg.svd(c, full_matrices=False)[2][:2]
        tets = Delaunay(c @ basis.T).simplices
    return [[nodes[i] for i in t] for t in tets]


def components(n: int, groups: list[list[int]]) -> list[list[int]]:
    """Indices of groups per connected component (groups sharing a node are connected)."""
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for g in groups:
        for v in g[1:]:
            parent[find(v)] = find(g[0])
    comps = {}
    for gi, g in enumerate(groups):
        comps.setdefault(find(g[0]), []).append(gi)
    return list(comps.values())


@torch.no_grad()
def mesh_union(sq: Superquadrics, pieces: list[list[int]], node_colors,
               voxel: float = 0.03, max_dim: int = 256):
    """Mesh the union of the given blends. Vertex colors are the blend weights applied to the
    node colors of whichever piece is closest."""
    blends = [SimplexBlend(sq, p) for p in pieces]
    lo = torch.stack([b.bounds()[0] for b in blends]).min(0).values
    hi = torch.stack([b.bounds()[1] for b in blends]).max(0).values
    voxel = max(voxel, float((hi - lo).max()) / (max_dim - 4))
    lo, hi = lo - 2 * voxel, hi + 2 * voxel
    dims = ((hi - lo) / voxel).ceil().long() + 1
    axes = [lo[i] + voxel * torch.arange(int(dims[i]), device=DEVICE) for i in range(3)]

    # each blend only needs to be evaluated inside its own bounding box; outside, the field is
    # clipped to a positive value, which is all marching cubes needs there
    field = torch.full(dims.tolist(), voxel * 4, device=DEVICE)
    for b in blends:
        b_lo, b_hi = b.bounds()
        i0 = ((b_lo - lo) / voxel).floor().long().clamp_min(0).tolist()
        i1 = ((b_hi - lo) / voxel).ceil().long().add(2).minimum(dims).tolist()
        sub = torch.stack(torch.meshgrid(*[axes[a][i0[a]:i1[a]] for a in range(3)], indexing="ij"), -1)
        d = blend_distance(b, sub.reshape(-1, 3))[0].reshape(sub.shape[:3])
        view = field[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]]
        torch.minimum(view, d, out=view)
    field = field.cpu().numpy()

    verts, faces, _, _ = marching_cubes(field, level=0.0, spacing=(voxel,) * 3)
    verts = torch.from_numpy(verts.copy()).float().to(DEVICE) + lo

    best_d = torch.full((len(verts),), torch.inf, device=DEVICE)
    colors = torch.zeros(len(verts), 3, device=DEVICE)
    node_colors = torch.as_tensor(node_colors, dtype=torch.float32, device=DEVICE)
    for b in blends:
        b_lo, b_hi = b.bounds()
        inside = ((verts >= b_lo - 2 * voxel) & (verts <= b_hi + 2 * voxel)).all(1).nonzero().squeeze(1)
        d, w = blend_distance(b, verts[inside])
        d = d.abs()
        better = d < best_d[inside]
        best_d[inside[better]] = d[better]
        colors[inside[better]] = (w @ node_colors[b.nodes])[better]
    return verts.cpu(), torch.from_numpy(faces.astype(np.int64)), colors.cpu()
