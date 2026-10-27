"""Superquadric parametrization, meshing and random crowd generation.

A superquadric is described by
    scale  (a1, a2, a3)   half-extents along the local x, y, z axes
    eps    (e1, e2)       shape exponents; e1 controls the z profile, e2 the xy cross-section
                          (0.1 ~ box-like, 1 ~ ellipsoid, 2 ~ octahedron/diamond)
    rot    (3, 3)         local-to-world rotation
    trans  (3,)           center in world coordinates

Parametric surface for eta in [-pi/2, pi/2], omega in [-pi, pi):
    x = a1 * cos(eta)^e1 * cos(omega)^e2
    y = a2 * cos(eta)^e1 * sin(omega)^e2
    z = a3 * sin(eta)^e1
with signed powers f(x)^e = sign(x) * |x|^e.
"""

from dataclasses import dataclass
import math

import torch


@dataclass
class Superquadrics:
    scale: torch.Tensor  # (M, 3)
    eps: torch.Tensor  # (M, 2)
    rot: torch.Tensor  # (M, 3, 3)
    trans: torch.Tensor  # (M, 3)

    def __len__(self):
        return self.scale.shape[0]


def signed_pow(x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * x.abs().clamp_min(1e-12) ** e


def random_rotation(m: int, generator: torch.Generator) -> torch.Tensor:
    """Uniformly random rotations from random unit quaternions."""
    q = torch.randn(m, 4, generator=generator)
    q = q / q.norm(dim=1, keepdim=True)
    w, x, y, z = q.unbind(1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=1).reshape(m, 3, 3)


def sample_surface_grid(sq: Superquadrics, n_eta: int = 32, n_omega: int = 64):
    """Evaluate every superquadric on a regular (eta, omega) grid.

    Returns world-space vertices (M, n_eta + 1, n_omega, 3). The omega axis wraps around,
    the first/last eta rows are the (degenerate) poles.
    """
    eta = torch.linspace(-math.pi / 2, math.pi / 2, n_eta + 1)
    omega = torch.linspace(-math.pi, math.pi, n_omega + 1)[:-1]
    eta, omega = torch.meshgrid(eta, omega, indexing="ij")  # (E, O)

    e1 = sq.eps[:, 0, None, None]
    e2 = sq.eps[:, 1, None, None]
    ce, se = signed_pow(torch.cos(eta), e1), signed_pow(torch.sin(eta), e1)
    co, so = signed_pow(torch.cos(omega), e2), signed_pow(torch.sin(omega), e2)
    local = torch.stack([ce * co, ce * so, se.expand_as(ce * co)], dim=-1)  # (M, E, O, 3)
    local = local * sq.scale[:, None, None, :]
    return torch.einsum("mij,meoj->meoi", sq.rot, local) + sq.trans[:, None, None, :]


def grid_faces(n_eta: int, n_omega: int) -> torch.Tensor:
    """Triangle indices for a (n_eta + 1) x n_omega vertex grid that wraps in omega."""
    r = torch.arange(n_eta)[:, None]
    c = torch.arange(n_omega)[None, :]
    c_next = (c + 1) % n_omega
    v00, v01 = r * n_omega + c, r * n_omega + c_next
    v10, v11 = (r + 1) * n_omega + c, (r + 1) * n_omega + c_next
    tris = torch.stack([
        torch.stack([v00, v01, v11], -1),
        torch.stack([v00, v11, v10], -1),
    ], dim=2)  # (n_eta, n_omega, 2, 3)
    return tris.reshape(-1, 3)


# Named shape presets (e1, e2) so the crowd covers the interesting corners of shape space.
SHAPE_PRESETS = {
    "box": (0.1, 0.1),
    "ellipsoid": (1.0, 1.0),
    "cylinder": (0.1, 1.0),
    "octahedron": (2.0, 2.0),
    "double_cone": (2.0, 1.0),
    "rounded_box": (0.4, 0.4),
    "diamond_prism": (0.1, 2.0),
    "pillow": (0.5, 1.0),
}


# Aspect-ratio classes (relative half-extents, before random axis order and jitter).
ASPECT_CLASSES = {
    "blob": (1.0, 1.0, 1.0),
    "rod": (1.0, 0.2, 0.2),
    "slab": (1.0, 1.0, 0.15),
    "plank": (1.0, 0.35, 0.08),
    "chunk": (1.0, 0.6, 0.4),
}


def rotation_about_y(angle: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(angle), torch.sin(angle)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([c, z, s, z, o, z, -s, z, c], -1).reshape(-1, 3, 3)


def random_axis_aligned(m: int, generator: torch.Generator) -> torch.Tensor:
    """Random signed permutation matrices with det +1 (the 24 rotations of a cube)."""
    out = torch.zeros(m, 3, 3)
    for i in range(m):
        perm = torch.randperm(3, generator=generator)
        signs = torch.randint(2, (3,), generator=generator).float() * 2 - 1
        out[i, torch.arange(3), perm] = signs
        if torch.linalg.det(out[i]) < 0:
            out[i, 0] *= -1
    return out


def random_crowd(n: int = 40, seed: int = 0, density: float = 0.12) -> tuple[Superquadrics, list[str]]:
    """Semi-random crowd of non-overlapping superquadrics in a 3D volume (y is up).

    - shape:       half presets, half random exponents
    - scale:       log-uniform overall size (0.15 .. 1.8) times an aspect class
                   (blob / rod / slab / plank / chunk) with jitter and random axis order
    - orientation: 1/3 axis-aligned (cube rotations), 1/3 upright with random yaw and a small tilt,
                   1/3 uniformly random
    - position:    rejection sampling in a wide, flat box whose volume follows from the
                   requested packing density of bounding spheres; largest shapes are placed first
    """
    g = torch.Generator().manual_seed(seed)
    rand = lambda *shape: torch.rand(*shape, generator=g)

    # shape exponents
    presets = list(SHAPE_PRESETS.items())
    eps = 0.1 + 1.9 * rand(n, 2)
    names = []
    pick = torch.randint(len(presets), (n,), generator=g)
    use_preset = rand(n) < 0.5
    for i in range(n):
        if use_preset[i]:
            name, e = presets[pick[i]]
            eps[i] = torch.tensor(e)
            names.append(name)
        else:
            names.append("random")

    # scales
    aspects = list(ASPECT_CLASSES.items())
    a_pick = torch.randint(len(aspects), (n,), generator=g)
    size = torch.exp(math.log(0.15) + (math.log(1.8) - math.log(0.15)) * rand(n))
    scale = torch.empty(n, 3)
    for i in range(n):
        a_name, ratios = aspects[a_pick[i]]
        r = torch.tensor(ratios) * (0.75 + 0.5 * rand(3))
        scale[i] = (size[i] * r)[torch.randperm(3, generator=g)]
        names[i] = f"{names[i]} {a_name}"
    scale = scale.clamp_min(0.06)  # stay thicker than the merge voxel size

    # orientations
    kind = torch.randint(3, (n,), generator=g)
    tilt_axis = torch.nn.functional.normalize(torch.randn(n, 3, generator=g) * torch.tensor([1.0, 0.0, 1.0]), dim=1)
    tilt = math.radians(12) * (2 * rand(n) - 1)
    rot = random_rotation(n, g)
    aligned = random_axis_aligned(n, g)
    yaw = rotation_about_y(2 * math.pi * rand(n))
    for i in range(n):
        if kind[i] == 0:
            rot[i] = aligned[i]
        elif kind[i] == 1:
            k = torch.tensor([[0, -tilt_axis[i, 2], tilt_axis[i, 1]],
                              [tilt_axis[i, 2], 0, -tilt_axis[i, 0]],
                              [-tilt_axis[i, 1], tilt_axis[i, 0], 0]])
            tilt_r = torch.eye(3) + math.sin(tilt[i]) * k + (1 - math.cos(tilt[i])) * k @ k  # Rodrigues
            rot[i] = tilt_r @ yaw[i] @ aligned[i]

    # positions: bounding spheres must not overlap (eps <= 2 keeps an SQ inside radius |scale|,
    # which is conservative, so allow a little sphere overlap)
    radius = scale.norm(dim=1)
    volume = (4 / 3 * math.pi * radius ** 3).sum() / density
    box = torch.tensor([2.0, 0.6, 2.0])
    box = box * (volume / box.prod()) ** (1 / 3)
    trans = torch.zeros(n, 3)
    placed = []
    for i in radius.argsort(descending=True).tolist():
        for attempt in range(500):
            grow = 1 + attempt / 250  # slowly enlarge the box if it is too crowded
            p = (rand(3) - 0.5) * box * grow
            if all((p - trans[j]).norm() > 0.85 * (radius[i] + radius[j]) for j in placed):
                break
        trans[i] = p
        placed.append(i)
    return Superquadrics(scale, eps, rot, trans), names
