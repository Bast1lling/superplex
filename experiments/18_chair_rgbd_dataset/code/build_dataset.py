"""Build a chair dataset from ShapeNet-Core: textured RGB-D renderings + SuperFlex geometry samples.

Per object, in the frame of the Occupancy Networks ShapeNet data (bbox centered, longest side 1, y up), so the
geometry matches what the SuperFlex checkpoints were trained on:
    pointcloud.npz       100k visible surface points + outward normals (float16, keys as in ONet)
    pointcloud_4096.npz  farthest-point subset of them
    points.npz           100k uniform points in [-0.55, 0.55]^3 + packed occupancies
    cameras.npz          K, c2w (V, 4, 4, OpenCV axes), light per texture
    depth/v00.png        16-bit z-depth in 1/10000 object units, 0 = background
    rgb/t00_v00.png      t00 = original ShapeNet materials, t01.. = random procedural textures per material

Occupancy comes from depth fusion, as in ONet: a point is free if some depth map sees it in front of the surface.
Faces no view sees (mesh interiors) get no surface samples. ShapeNet's solid binvox are unusable (see README).

    uv run python experiments/18_chair_rgbd_dataset/code/build_dataset.py --out /home/sebastian/data/ShapeNet-Chairs-RGBD
Finished objects are skipped, so an interrupted run can be restarted with the same command.
"""
import argparse
import json
import math
import os
import shutil
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import trimesh
from PIL import Image

SYNSET = "03001627"
CORE = Path("/home/sebastian/data/ShapeNet-Core") / SYNSET / SYNSET
ONET = Path("/home/sebastian/Documents/Uni/GF4Shape/data/ShapeNet") / SYNSET  # only for the train/val/test lists
TO_ONET = torch.tensor([[0.0, 0, -1], [0, 1, 0], [1, 0, 0]])  # ShapeNet v2 -> ONet: 270 deg around y

N_SURFACE, N_OCC, N_FPS = 100_000, 100_000, 4096
FUSION_VIEWS, FUSION_RES, FUSION_EPS = 100, 512, 0.005  # eps: shell thickness, in object units
RENDER_FOCAL = 1.5  # focal length / image width, fov 36.9 deg
DEPTH_UNIT = 1e-4


def load_mesh(path, device, max_tex=1024):
    """Vertices (ONet frame), faces, uvs, per-face material id and the materials (color or image)."""
    scene = trimesh.load(path, process=False)
    parts = list(scene.geometry.values()) if isinstance(scene, trimesh.Scene) else [scene]
    verts, faces, uvs, face_mat, materials, offset = [], [], [], [], [], 0
    for m, part in enumerate(parts):
        n = len(part.vertices)
        uv = getattr(part.visual, "uv", None)
        uvs.append(np.zeros((n, 2)) if uv is None or len(uv) != n else np.asarray(uv))
        verts.append(np.asarray(part.vertices))
        faces.append(np.asarray(part.faces) + offset)
        face_mat.append(np.full(len(part.faces), m))
        offset += n

        material = getattr(part.visual, "material", None)
        image = getattr(material, "image", None)
        color = np.asarray(getattr(material, "main_color", (200, 200, 200, 255)), np.float32)[:3] / 255
        if image is not None:
            image = image.convert("RGB")
            image.thumbnail((max_tex, max_tex))
            image = torch.tensor(np.asarray(image).copy(), device=device, dtype=torch.float32).permute(2, 0, 1) / 255
        materials.append({"color": torch.tensor(color, device=device), "image": image})

    v = torch.tensor(np.concatenate(verts), device=device, dtype=torch.float32)
    low, high = v.amin(0), v.amax(0)
    scale, loc = (high - low).amax(), (low + high) / 2
    v = (v - loc) / scale @ TO_ONET.to(device).T
    f = torch.tensor(np.concatenate(faces), device=device, dtype=torch.long)
    uv = torch.tensor(np.concatenate(uvs), device=device, dtype=torch.float32)
    return v, f, uv, torch.tensor(np.concatenate(face_mat), device=device), materials, loc, scale


def look_at(eye):
    """OpenCV camera-to-world looking at the origin, y up (eye: (V, 3))."""
    forward = F.normalize(-eye, dim=-1)
    up = torch.tensor([0.0, 1, 0], device=eye.device).expand_as(eye)
    up = torch.where(forward[:, 1:2].abs() > 0.99, torch.tensor([0.0, 0, 1], device=eye.device), up)
    right = F.normalize(torch.linalg.cross(forward, up), dim=-1)
    down = torch.linalg.cross(forward, right)
    c2w = torch.eye(4, device=eye.device).repeat(len(eye), 1, 1)
    c2w[:, :3, 0], c2w[:, :3, 1], c2w[:, :3, 2], c2w[:, :3, 3] = right, down, forward, eye
    return c2w


def to_camera(points, c2w):
    return (points - c2w[:3, 3]) @ c2w[:3, :3]


def rasterize(verts, faces, c2w, focal, res, max_candidates=4_000_000):
    """Z-buffer one view. Returns depth (res, res; inf = empty) and the visible face (-1 = empty)."""
    cam = to_camera(verts, c2w)
    z = cam[:, 2]
    px = cam[:, :2] / z[:, None] * focal + res / 2 - 0.5  # pixel centers at integers
    tri, tri_invz = px[faces], (1 / z)[faces]
    low = tri.amin(1).ceil().clamp(0, res - 1).long()
    high = tri.amax(1).floor().clamp(0, res - 1).long()
    keep = ((tri.amax(1) >= 0) & (tri.amin(1) <= res - 1)).all(1) & (high >= low).all(1)
    ids = keep.nonzero()[:, 0]
    width = (high - low + 1)[ids]
    area = width.prod(1)

    invz_buffer = torch.zeros(res * res, device=verts.device)
    face_buffer = torch.full((res * res,), -1, device=verts.device, dtype=torch.long)
    chunk = (area.cumsum(0) / max_candidates).long()
    for c in range(int(chunk[-1]) + 1 if len(chunk) else 0):
        sel = chunk == c
        fid, w, count = ids[sel], width[sel], area[sel]
        owner = torch.repeat_interleave(torch.arange(len(fid), device=fid.device), count)
        local = torch.arange(len(owner), device=fid.device) - (count.cumsum(0) - count)[owner]
        x = low[fid[owner], 0] + local % w[owner, 0]
        y = low[fid[owner], 1] + torch.div(local, w[owner, 0], rounding_mode="floor")

        a, b, c_ = tri[fid[owner], 0], tri[fid[owner], 1], tri[fid[owner], 2]
        p = torch.stack([x, y], -1).float()
        edge = lambda s, t: (t[:, 0] - s[:, 0]) * (p[:, 1] - s[:, 1]) - (t[:, 1] - s[:, 1]) * (p[:, 0] - s[:, 0])
        w0, w1, w2 = edge(b, c_), edge(c_, a), edge(a, b)
        total = w0 + w1 + w2
        inside = ((w0 * total >= 0) & (w1 * total >= 0) & (w2 * total >= 0)) & (total != 0)
        invz = (torch.stack([w0, w1, w2], -1) / total[:, None] * tri_invz[fid[owner]]).sum(-1)
        inside &= invz > 0
        pixel, invz, face = (y * res + x)[inside], invz[inside], fid[owner][inside]
        invz_buffer = invz_buffer.scatter_reduce(0, pixel, invz, "amax")
        win = invz == invz_buffer[pixel]
        face_buffer[pixel[win]] = face[win]
    return (1 / invz_buffer).view(res, res), face_buffer.view(res, res)


def fibonacci_sphere(n, device):
    i = torch.arange(n, device=device) + 0.5
    y, phi = 1 - 2 * i / n, math.pi * (3 - 5**0.5) * i
    r = (1 - y * y).sqrt()
    return torch.stack([r * phi.cos(), y, r * phi.sin()], -1)


class Fusion:
    """All-around depth maps; answers which points are free space and which lie on the visible surface."""

    def __init__(self, verts, faces):
        self.res, self.focal = FUSION_RES, FUSION_RES * 1.6
        distance = verts.norm(dim=-1).amax() * 3.5  # keeps the bounding sphere inside the image
        self.c2w = look_at(fibonacci_sphere(FUSION_VIEWS, verts.device) * distance)
        maps = [rasterize(verts, faces, c, self.focal, self.res)[0] for c in self.c2w]
        # nearest depth in a 3x3 window, so sub-pixel legs are not carved away
        self.depth = -F.max_pool2d(-torch.stack(maps)[:, None], 3, 1, 1)[:, 0]

    def query(self, points):
        """free: seen in front of the surface by some view. seen_from: a view seeing it on the surface, or -1."""
        free = torch.zeros(len(points), device=points.device, dtype=torch.bool)
        seen_from = torch.full((len(points),), -1, device=points.device)
        for v, c in enumerate(self.c2w):
            cam = to_camera(points, c)
            z = cam[:, 2]
            uv = (cam[:, :2] / z[:, None] * self.focal + self.res / 2 - 0.5).round().long()
            ok = (uv >= 0).all(1) & (uv < self.res).all(1)
            depth = torch.full_like(z, float("inf"))
            depth[ok] = self.depth[v, uv[ok, 1], uv[ok, 0]]
            free |= z < depth - FUSION_EPS
            seen_from[(z - depth).abs() < FUSION_EPS] = v
        return free, seen_from


def sample_surface(verts, faces, n, generator):
    tri = verts[faces]
    normal = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = normal.norm(dim=-1)
    pick = torch.searchsorted(area.double().cumsum(0), torch.rand(n, device=verts.device, generator=generator).double() * area.double().sum())
    pick = pick.clamp(max=len(faces) - 1)
    r = torch.rand(n, 2, device=verts.device, generator=generator)
    s = r[:, :1].sqrt()
    t = tri[pick]
    return (1 - s) * t[:, 0] + s * (1 - r[:, 1:]) * t[:, 1] + s * r[:, 1:] * t[:, 2], F.normalize(normal[pick], dim=-1)


def farthest_points(points, n, generator):
    picks = torch.zeros(n, device=points.device, dtype=torch.long)
    picks[0] = torch.randint(len(points), (1,), device=points.device, generator=generator)
    distance = (points - points[picks[0]]).square().sum(-1)
    for i in range(1, n):
        picks[i] = distance.argmax()
        distance = distance.minimum((points - points[picks[i]]).square().sum(-1))
    return picks


def geometry(verts, faces, generator):
    fusion = Fusion(verts, faces)
    surface, normals = sample_surface(verts, faces, 3 * N_SURFACE, generator)
    _, seen_from = fusion.query(surface)
    surface, normals, seen_from = surface[seen_from >= 0], normals[seen_from >= 0], seen_from[seen_from >= 0]
    toward_camera = fusion.c2w[seen_from, :3, 3] - surface
    normals = torch.where((normals * toward_camera).sum(-1, keepdim=True) < 0, -normals, normals)
    keep = torch.randint(len(surface), (N_SURFACE,), device=verts.device, generator=generator)  # with replacement
    surface, normals = surface[keep], normals[keep]

    query = (torch.rand(N_OCC, 3, device=verts.device, generator=generator) - 0.5) * 1.1
    free = torch.cat([fusion.query(chunk)[0] for chunk in query.split(20_000)])
    return surface, normals, query, ~free


def random_cameras(verts, n, rng, device):
    """Stratified azimuth, elevation -10..50 deg, object just fills the image."""
    azimuth = (np.arange(n) + rng.uniform(0, 1, n)) / n * 2 * math.pi
    elevation = np.radians(rng.uniform(-10, 50, n))
    half_fov = math.atan(0.5 / RENDER_FOCAL)
    distance = float(verts.norm(dim=-1).amax()) / math.sin(half_fov) * rng.uniform(1.0, 1.2, n)
    eye = np.stack([np.cos(elevation) * np.cos(azimuth), np.sin(elevation), np.cos(elevation) * np.sin(azimuth)], -1)
    return look_at(torch.tensor(eye * distance[:, None], device=device, dtype=torch.float32))


def random_color(rng):
    """Saturated or muted, dark or light: anything but the gray of untextured ShapeNet."""
    hue, saturation, value = rng.uniform(0, 1), rng.uniform(0.1, 0.9), rng.uniform(0.25, 0.95)
    k = (np.array([5, 3, 1]) + hue * 6) % 6
    return value - value * saturation * np.clip(np.minimum(k, 4 - k), 0, 1)


def value_noise(points, rng, octaves=3):
    grid = torch.tensor(rng.uniform(0, 1, (1, 1, 32, 32, 32)), device=points.device, dtype=torch.float32)
    noise = 0
    for octave in range(octaves):
        coords = (points * rng.uniform(0.6, 1.0) * 2**octave + rng.uniform(0, 2)) % 2 - 1
        sample = F.grid_sample(grid, coords.view(1, 1, 1, -1, 3).flip(-1), align_corners=True, padding_mode="border")
        noise = noise + sample.view(-1) / 2**octave
    return noise / (2 - 2 ** (1 - octaves))


def procedural(points, rng):
    """Albedo at 3D points: one random pattern, so it stays consistent between views."""
    a, b = (torch.tensor(random_color(rng), device=points.device, dtype=torch.float32) for _ in range(2))
    kind = rng.choice(["solid", "checker", "stripes", "noise", "wood"])
    direction = torch.tensor(rng.normal(size=3), device=points.device, dtype=torch.float32)
    direction = direction / direction.norm()
    if kind == "solid":
        mix = torch.zeros(len(points), device=points.device)
    elif kind == "checker":
        mix = ((points * rng.uniform(6, 24)).floor().sum(-1) % 2 == 0).float()
    elif kind == "stripes":
        mix = (torch.sin(points @ direction * rng.uniform(15, 80)) > 0).float()
    elif kind == "noise":
        mix = (value_noise(points, rng) - 0.5) * 3 + 0.5
    else:
        mix = 0.5 + 0.5 * torch.sin((points @ direction + 0.15 * value_noise(points, rng)) * rng.uniform(40, 120))
    return a + (b - a) * mix.clamp(0, 1)[:, None]


def shade(face, depth, verts, faces, uvs, face_mat, materials, c2w, focal, res, light, ambient, original):
    """RGB of one view on white. All of it is per pixel, so RGB and depth stay perfectly aligned."""
    device = verts.device
    rgb = torch.ones(res, res, 3, device=device)
    mask = face >= 0
    rows, cols = mask.nonzero(as_tuple=True)
    rays = torch.stack([(cols - res / 2 + 0.5) / focal, (rows - res / 2 + 0.5) / focal, torch.ones_like(rows)], -1).float()
    points = c2w[:3, 3] + (rays * depth[mask][:, None]) @ c2w[:3, :3].T

    fid = face[mask]
    tri = verts[faces[fid]]
    normal = F.normalize(torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), dim=-1)
    view = F.normalize(c2w[:3, 3] - points, dim=-1)
    normal = torch.where((normal * view).sum(-1, keepdim=True) < 0, -normal, normal)  # ShapeNet winding is unreliable

    albedo = torch.zeros(len(fid), 3, device=device)
    pixel_mat = face_mat[fid]
    for m in pixel_mat.unique().tolist():
        sel = pixel_mat == m
        material = materials[m]
        if not original:
            albedo[sel] = material["pattern"](points[sel])
        elif material["image"] is None:
            albedo[sel] = material["color"]
        else:
            t = tri[sel]
            cross = torch.linalg.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
            denominator = cross.square().sum(-1).clamp(min=1e-20)
            p = points[sel]
            w1 = (torch.linalg.cross(p - t[:, 0], t[:, 2] - t[:, 0]) * cross).sum(-1) / denominator
            w2 = (torch.linalg.cross(t[:, 1] - t[:, 0], p - t[:, 0]) * cross).sum(-1) / denominator
            bary = torch.stack([1 - w1 - w2, w1, w2], -1)
            uv = (uvs[faces[fid[sel]]] * bary[:, :, None]).sum(1) % 1
            grid = torch.stack([uv[:, 0] * 2 - 1, 1 - uv[:, 1] * 2], -1).view(1, 1, -1, 2)
            albedo[sel] = F.grid_sample(material["image"][None], grid, align_corners=False)[0, :, 0].T

    diffuse = 0.5 * (normal * view).sum(-1) + 0.5 * (normal @ light).clamp(min=0)
    rgb[mask] = (albedo * (ambient + (1 - ambient) * diffuse)[:, None]).clamp(0, 1)
    return rgb


def save_png(array, path):
    Image.fromarray(array).save(path, compress_level=3)


def process(model_id, out, args, device):
    final = out / SYNSET / model_id
    tmp = final.with_name(model_id + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "rgb").mkdir(parents=True)
    (tmp / "depth").mkdir()
    seed = int(model_id.removeprefix("u")[:8], 16)  # a few ids start with a "u"; the others keep their seed
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device).manual_seed(seed)

    verts, faces, uvs, face_mat, materials, loc, scale = load_mesh(CORE / model_id / "models" / "model_normalized.obj", device)

    surface, normals, query, occupied = geometry(verts, faces, generator)
    keep = farthest_points(surface, N_FPS, generator)
    origin = dict(loc=np.zeros(3), scale=scale.item())  # as in ONet; points are already normalized
    to_half = lambda x: x.cpu().numpy().astype(np.float16)
    np.savez(tmp / "pointcloud.npz", points=to_half(surface), normals=to_half(normals), **origin)
    np.savez(tmp / "pointcloud_4096.npz", points=surface[keep].cpu().numpy(), normals=normals[keep].cpu().numpy())
    np.savez(tmp / "points.npz", points=to_half(query), occupancies=np.packbits(occupied.cpu().numpy()), **origin)

    focal = RENDER_FOCAL * args.res
    c2w = random_cameras(verts, args.views, rng, device)
    renders = [rasterize(verts, faces, c, focal, args.res) for c in c2w]
    for v, (depth, _) in enumerate(renders):
        pixels = (depth.nan_to_num(0, posinf=0) / DEPTH_UNIT).round().clamp(0, 65535).cpu().numpy().astype(np.uint16)
        save_png(pixels, tmp / "depth" / f"v{v:02d}.png")

    lights = []
    for t in range(args.textures):
        for material in materials:
            pattern_seed = rng.integers(2**32)
            material["pattern"] = lambda p, s=pattern_seed: procedural(p, np.random.default_rng(s))
        light = F.normalize(torch.tensor(rng.normal(size=3) + [0, 1, 0], device=device, dtype=torch.float32), dim=0)
        ambient = float(rng.uniform(0.3, 0.5))
        lights.append(light.cpu().numpy())
        for v, (depth, face) in enumerate(renders):
            rgb = shade(face, depth, verts, faces, uvs, face_mat, materials, c2w[v], focal, args.res, light, ambient, t == 0)
            save_png((rgb * 255).round().byte().cpu().numpy(), tmp / "rgb" / f"t{t:02d}_v{v:02d}.png")

    K = np.array([[focal, 0, args.res / 2 - 0.5], [0, focal, args.res / 2 - 0.5], [0, 0, 1]], np.float32)
    np.savez(tmp / "cameras.npz", K=K, c2w=c2w.cpu().numpy(), light=np.stack(lights), depth_unit=DEPTH_UNIT,
             mesh_loc=loc.cpu().numpy(), mesh_scale=scale.item())
    tmp.rename(final)


def worker(ids, out, args, memory_fraction):
    torch.set_num_threads(1)
    device = "cuda:0"
    torch.cuda.set_per_process_memory_fraction(memory_fraction, 0)  # empties the cache before running out of memory
    for model_id in ids:
        try:
            process(model_id, out, args, device)
        except Exception:
            with open(out / "errors.log", "a") as log:
                log.write(f"=== {model_id}\n{traceback.format_exc()}\n")
            shutil.rmtree(out / SYNSET / (model_id + ".tmp"), ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--textures", type=int, default=4, help="t00 is the original material")
    parser.add_argument("--res", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="only the first N objects, for a pilot")
    args = parser.parse_args()

    (args.out / SYNSET).mkdir(parents=True, exist_ok=True)
    ids = []
    for split in ["train", "val", "test"]:  # ONet's split, so the checkpoints' held-out objects stay held out
        lst = (ONET / f"{split}.lst").read_text()
        (args.out / SYNSET / f"{split}.lst").write_text(lst)
        ids += lst.split()
    if args.limit:
        ids = sorted(ids)[: args.limit]
    todo = [i for i in ids if not (args.out / SYNSET / i).is_dir()]
    print(f"{len(ids) - len(todo)} done, {len(todo)} to do", flush=True)
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    ctx = mp.get_context("spawn")
    for workers in [args.workers, 1]:  # the second pass gives what failed (huge meshes) the whole GPU
        todo = [i for i in ids if not (args.out / SYNSET / i).is_dir()]
        procs = [ctx.Process(target=worker, args=(todo[r :: workers], args.out, args, 0.9 / workers), daemon=True)
                 for r in range(workers)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
    print(f"finished, {sum((args.out / SYNSET / i).is_dir() for i in ids)} of {len(ids)} objects", flush=True)


if __name__ == "__main__":
    main()
