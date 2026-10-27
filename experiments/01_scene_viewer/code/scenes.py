"""Point clouds for the Replica scenes.

Replica only ships meshes, so a point cloud is made by area-uniform sampling of the mesh
surface, followed by a 1 cm voxel-grid downsampling. Per voxel:
    pos, color, normal   mean of the samples (vertex attributes interpolated barycentrically;
                         Replica's stored normals point inwards and are flipped)
    instance, segment    majority vote of the per-face object ids of habitat/mesh_semantic.ply
                         (annotated objects) and habitat/mesh_preseg_semantic.ply (over-segmentation)
Semantic classes follow from instances via info_semantic.json ("id_to_label").
Results are cached as .npz next to the dataset. Replica meshes are z-up, in meters.
"""

import json
from pathlib import Path

import numpy as np
import torch

DATA = Path("/home/sebastian/Documents/Uni/Thesis/data")
REPLICA = DATA / "replica_v1"
CACHE = DATA / "replica_v1_points"
VOXEL = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1",
              "char": "i1", "int8": "i1", "ushort": "<u2", "uint16": "<u2", "short": "<i2",
              "int16": "<i2", "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4"}


def list_scenes() -> list[str]:
    return sorted(p.parent.name for p in REPLICA.glob("*/mesh.ply"))


def read_ply_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Binary little-endian PLY with fixed-size polygon faces -> (vertex records, face indices (F,k), face records)."""
    with open(path, "rb") as f:
        header = []
        while (line := f.readline().decode().strip()) != "end_header":
            header.append(line.split())
        assert ["format", "binary_little_endian", "1.0"] in header, "only binary little-endian PLY"
        elements, cur = [], None
        for tok in header:
            if tok[0] == "element":
                cur = (tok[1], int(tok[2]), [])
                elements.append(cur)
            elif tok[0] == "property":
                cur[2].append(tok[1:])
        (vname, nv, vprops), (fname, nf, fprops) = elements
        assert vname == "vertex" and fname == "face"

        verts = np.fromfile(f, dtype=[(p[1], _PLY_TYPES[p[0]]) for p in vprops], count=nv)
        # faces: "list <count type> <index type> vertex_indices" (+ optional scalar props); assume a fixed k
        cnt_t, idx_t = _PLY_TYPES[fprops[0][1]], _PLY_TYPES[fprops[0][2]]
        k = int(np.frombuffer(f.read(np.dtype(cnt_t).itemsize), cnt_t)[0])
        f.seek(-np.dtype(cnt_t).itemsize, 1)
        fdtype = [("k", cnt_t), ("idx", idx_t, (k,))] + [(p[1], _PLY_TYPES[p[0]]) for p in fprops[1:]]
        faces = np.fromfile(f, dtype=fdtype, count=nf)
        assert (faces["k"] == k).all(), "mixed polygon sizes are not supported"
    return verts, faces["idx"].astype(np.int64), faces


def triangulate(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fan triangulation of (F, k) polygons -> triangles (F * (k - 2), 3) and their source face index."""
    k = faces.shape[1]
    tris = np.concatenate([faces[:, [0, i, i + 1]] for i in range(1, k - 1)])
    return tris, np.tile(np.arange(len(faces)), k - 2)


def _mode(inv: torch.Tensor, labels: torch.Tensor, n_vox: int) -> torch.Tensor:
    """Most frequent label per voxel (ties broken arbitrarily)."""
    pair, cnt = torch.unique(inv * 65536 + labels, return_counts=True)
    vox, lab = pair // 65536, pair % 65536
    best = torch.zeros(n_vox, dtype=cnt.dtype, device=cnt.device).scatter_reduce(0, vox, cnt, "amax")
    win = cnt == best[vox]
    out = torch.zeros(n_vox, dtype=torch.long, device=cnt.device)
    out[vox[win]] = lab[win]
    return out


@torch.no_grad()
def sample_voxelized(xyz, rgb, nrm, tris, tri_labels: dict[str, np.ndarray], voxel: float = VOXEL,
                     per_voxel: float = 8.0, seed: int = 0) -> dict[str, np.ndarray]:
    """Area-uniform surface samples (~per_voxel samples per voxel face area), aggregated per voxel."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    v = torch.from_numpy(xyz).to(DEVICE)
    attrs = torch.cat([torch.from_numpy(rgb).float(), torch.from_numpy(nrm)], 1).to(DEVICE)  # (V, 6)
    t = torch.from_numpy(tris).to(DEVICE)
    area = 0.5 * torch.linalg.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]).norm(dim=1)
    m = int(area.sum().item() * per_voxel / voxel ** 2)
    cdf = torch.cumsum(area.double(), 0)
    cdf /= cdf[-1].clone()

    tri = torch.searchsorted(cdf, torch.rand(m, device=DEVICE, generator=g, dtype=torch.float64)).clamp_max(len(t) - 1)
    r1, r2 = torch.rand(2, m, 1, device=DEVICE, generator=g)
    s = r1.sqrt()
    w = (1 - s, s * (1 - r2), s * r2)  # uniform barycentric coordinates
    ti = t[tri]
    pos = sum(w[j] * v[ti[:, j]] for j in range(3))
    att = sum(w[j] * attrs[ti[:, j]] for j in range(3))
    del ti

    ijk = ((pos - (v.min(0).values - voxel)) / voxel).floor().long()
    key, inv = torch.unique((ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2], return_inverse=True)  # 21 bits per axis
    n = len(key)
    cnt = torch.zeros(n, device=DEVICE).index_add_(0, inv, torch.ones(m, device=DEVICE))[:, None]
    pos = torch.zeros(n, 3, device=DEVICE).index_add_(0, inv, pos) / cnt
    att = torch.zeros(n, 6, device=DEVICE).index_add_(0, inv, att) / cnt
    out = {
        "pos": pos.cpu().numpy(),
        "col": att[:, :3].round().clamp(0, 255).to(torch.uint8).cpu().numpy(),
        "nrm": torch.nn.functional.normalize(att[:, 3:], dim=1).cpu().numpy().astype(np.float16),
    }
    for name, lab in tri_labels.items():
        lab = torch.from_numpy(lab.astype(np.int64)).to(DEVICE)[tri]
        out[name] = _mode(inv, lab, n).cpu().numpy().astype(np.uint16)
    return out


def load_points(scene: str) -> dict[str, np.ndarray]:
    """1 cm point cloud of a scene: pos (N,3) f32, col (N,3) u8, nrm (N,3) f16, instance (N,) u16, segment (N,) u16."""
    path = CACHE / f"{scene}_voxel{VOXEL * 1000:g}mm.npz"
    if path.exists():
        return dict(np.load(path))
    verts, faces, _ = read_ply_mesh(REPLICA / scene / "mesh.ply")
    _, _, sem = read_ply_mesh(REPLICA / scene / "habitat/mesh_semantic.ply")
    _, _, pre = read_ply_mesh(REPLICA / scene / "habitat/mesh_preseg_semantic.ply")
    xyz = np.stack([verts["x"], verts["y"], verts["z"]], 1).astype(np.float32)
    rgb = np.stack([verts["red"], verts["green"], verts["blue"]], 1)
    # Replica's vertex normals point into the surface (floors down, ceilings up, opposite to the face winding)
    nrm = -np.stack([verts["nx"], verts["ny"], verts["nz"]], 1).astype(np.float32)
    tris, src = triangulate(faces)
    pts = sample_voxelized(xyz, rgb, nrm, tris, {"instance": sem["object_id"][src], "segment": pre["object_id"][src]})
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, **pts)
    return pts


def load_semantics(scene: str) -> dict:
    """Class names and the instance -> class lookup (class id 0 = unlabeled)."""
    info = json.loads((REPLICA / scene / "habitat/info_semantic.json").read_text())
    return {
        "classes": {c["id"]: c["name"] for c in info["classes"]},
        "id_to_label": [max(c, 0) for c in info["id_to_label"]],
    }
