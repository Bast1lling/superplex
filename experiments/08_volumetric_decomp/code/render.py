"""Minimal software renderer (orthographic point splatting with a z-buffer) for point clouds and SQ sets.

Used to inspect decompositions visually (PNG), no OpenGL needed.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def palette(n: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = rng.permutation(n) / max(n, 1)
    s = 0.55 + 0.35 * rng.random(n)
    v = 0.75 + 0.25 * rng.random(n)
    return matplotlib.colors.hsv_to_rgb(np.stack([h, s, v], 1))


def camera(elev: float, azim: float):
    """Rotation world -> view (x right, y up, z towards the viewer)."""
    e, a = np.radians(elev), np.radians(azim)
    fwd = -np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])  # viewing direction
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right) + 1e-12
    up = np.cross(right, fwd)
    return np.stack([right, up, -fwd])


def splat(points, colors, normals, R, bounds, width=900, radius=1):
    """Project points (N, 3) with the view rotation R; bounds = (xmin, xmax, ymin, ymax) in view coords."""
    v = points @ R.T
    xmin, xmax, ymin, ymax = bounds
    scale = width / (xmax - xmin)
    height = int(np.ceil((ymax - ymin) * scale))
    px = ((v[:, 0] - xmin) * scale).astype(int)
    py = ((ymax - v[:, 1]) * scale).astype(int)
    light = np.array([0.3, 0.5, 0.8])
    light /= np.linalg.norm(light)
    if normals is not None:
        nv = normals @ R.T
        nv *= np.sign(nv[:, 2:3] + 1e-9)  # two-sided
        shade = 0.45 + 0.55 * np.clip(nv @ light, 0, 1)
    else:
        shade = np.ones(len(points))
    col = np.clip(colors * shade[:, None], 0, 1)
    img = np.ones((height, width, 3))
    depth = np.full((height, width), -np.inf)
    order = np.argsort(v[:, 2])  # far to near, later writes win
    for dx in range(-radius + 1, radius):
        for dy in range(-radius + 1, radius):
            pass
    offs = [(dx, dy) for dx in range(-radius + 1, radius) for dy in range(-radius + 1, radius)] or [(0, 0)]
    for dx, dy in offs:
        x, y = px[order] + dx, py[order] + dy
        ok = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x, y, z, c = x[ok], y[ok], v[order][ok, 2], col[order][ok]
        upd = z >= depth[y, x]
        # z sorted ascending: the last write per pixel is the nearest
        img[y[upd], x[upd]] = c[upd]
        depth[y[upd], x[upd]] = np.maximum(depth[y[upd], x[upd]], z[upd])
    return img


def view_bounds(points, R, margin=0.05):
    v = points @ R.T
    lo, hi = v.min(0), v.max(0)
    return (lo[0] - margin, hi[0] + margin, lo[1] - margin, hi[1] + margin)


def save_grid(images, titles, path, cols=2):
    rows = int(np.ceil(len(images) / cols))
    h, w = images[0].shape[:2]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * w / 100, rows * h / 100 + 0.4 * rows), dpi=100)
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, im, t in zip(np.atleast_1d(axes).ravel(), images, titles):
        ax.imshow(im)
        ax.set_title(t, fontsize=10)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- superquadrics

def sq_gauge(q, a, e1, e2):
    """Gauge function G (=1 on the surface), numpy, q (..., 3)."""
    x = np.abs(q / a)
    xy = (x[..., 0] ** (2 / e2) + x[..., 1] ** (2 / e2)) ** (e2 / e1)
    return (xy + x[..., 2] ** (2 / e1)) ** (e1 / 2)


def sq_surface(scale, eps, rot, trans, spacing=0.015):
    """Surface samples + normals of one SQ: points on the unit cube surface, scaled by a, radially projected."""
    a = np.maximum(scale, 1e-3)
    pts = []
    for ax in range(3):
        o = [i for i in range(3) if i != ax]
        n0 = max(2, int(2 * a[o[0]] / spacing))
        n1 = max(2, int(2 * a[o[1]] / spacing))
        u, v = np.meshgrid(np.linspace(-1, 1, n0), np.linspace(-1, 1, n1), indexing="ij")
        for s in (-1, 1):
            c = np.zeros((u.size, 3))
            c[:, ax] = s
            c[:, o[0]] = u.ravel()
            c[:, o[1]] = v.ravel()
            pts.append(c)
    q = np.concatenate(pts) * a
    G = sq_gauge(q, a, eps[0], eps[1])
    q = q / G[:, None]
    # normal = grad G (numerical)
    h = 1e-4 * a.max()
    grad = np.stack([(sq_gauge(q + h * np.eye(3)[i], a, eps[0], eps[1]) -
                      sq_gauge(q - h * np.eye(3)[i], a, eps[0], eps[1])) for i in range(3)], -1)
    nrm = grad / (np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12)
    return q @ rot.T + trans, nrm @ rot.T


def sqs_points(sqs: dict, colors: np.ndarray, spacing=0.015):
    P, N, C = [], [], []
    for j in range(len(sqs["scale"])):
        p, n = sq_surface(sqs["scale"][j], sqs["eps"][j], sqs["rot"][j], sqs["trans"][j], spacing)
        P.append(p), N.append(n), C.append(np.repeat(colors[j][None], len(p), 0))
    if not P:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(P), np.concatenate(N), np.concatenate(C)
