"""Speed benchmark: dense SDF of N points to each of M random SQs (M * N evaluations), per method.

Times forward (no grad) and forward + backward (loss = sum of SDFs, gradients w.r.t. all SQ
parameters), CUDA-synchronized, median over repeats after a warm-up. Writes results/benchmark.{json,md}.

    uv run python experiments/02_sq_sdf/code/benchmark.py
"""

import json
from pathlib import Path
import time

import torch

from sdf import METHODS, to_local

DEVICE = "cuda"
NS = [1_000, 10_000, 100_000]
MS = [10, 100, 1000]
CHUNK = 1 << 22  # pairs per chunk (bounds memory of the M x N x ... intermediates)
RESULTS = Path(__file__).parent.parent / "results"


def random_sqs(m, g):
    q = torch.randn(m, 4, generator=g, device=DEVICE)
    w, x, y, z = (q / q.norm(dim=1, keepdim=True)).unbind(1)
    rot = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                       2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                       2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(m, 3, 3)
    scale = 0.1 + 0.9 * torch.rand(m, 3, generator=g, device=DEVICE)
    eps = 0.1 + 1.9 * torch.rand(m, 2, generator=g, device=DEVICE)
    trans = 2 * torch.rand(m, 3, generator=g, device=DEVICE) - 1
    return scale, eps, rot, trans


def run(method, x, params, backward):
    m = len(params[0])
    n_chunk = max(1, CHUNK // m)
    if backward:
        params = [p.detach().requires_grad_() for p in params]
    scale, eps, rot, trans = params
    a, e1, e2 = scale[:, None], eps[:, 0, None], eps[:, 1, None]
    for i in range(0, len(x), n_chunk):
        if backward:
            method(to_local(x[i:i + n_chunk], rot, trans), a, e1, e2).sum().backward()
        else:
            with torch.no_grad():
                method(to_local(x[i:i + n_chunk], rot, trans), a, e1, e2)


def timeit(fn, repeats):
    fn()  # warm-up
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return sorted(times)[len(times) // 2]


def main():
    g = torch.Generator(device=DEVICE).manual_seed(0)
    rows = []
    for m in MS:
        params = random_sqs(m, g)
        for n in NS:
            x = 4 * torch.rand(n, 3, generator=g, device=DEVICE) - 2
            repeats = 5 if m * n <= 10_000_000 else 3
            for name, method in METHODS.items():
                row = {"method": name, "M": m, "N": n, "pairs": m * n}
                for mode in ("fwd", "fwd_bwd"):
                    torch.cuda.reset_peak_memory_stats()
                    row[f"{mode}_ms"] = 1000 * timeit(lambda: run(method, x, params, mode == "fwd_bwd"), repeats)
                    row[f"{mode}_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
                rows.append(row)
                print(f"{name:14s} M={m:5d} N={n:7d}  fwd {row['fwd_ms']:9.2f} ms  fwd+bwd {row['fwd_bwd_ms']:9.2f} ms", flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "benchmark.json").write_text(json.dumps({"device": torch.cuda.get_device_name(), "torch": torch.__version__,
                                                        "chunk_pairs": CHUNK, "rows": rows}, indent=1))
    lines = [f"# SDF speed benchmark\n", f"{torch.cuda.get_device_name()}, torch {torch.__version__}, float32, "
             f"dense M SQs x N points, chunks of {CHUNK:,} pairs. Median time in ms; forward / forward+backward "
             f"(gradients w.r.t. all SQ parameters). Throughput in the last column is for the forward pass at the largest size.\n"]
    for mode, title in (("fwd_ms", "Forward"), ("fwd_bwd_ms", "Forward + backward")):
        lines.append(f"\n## {title} (ms)\n")
        header = "| method | " + " | ".join(f"M={m} N={n:,}" for m in MS for n in NS) + " |"
        lines += [header, "|" + "---|" * (len(MS) * len(NS) + 1)]
        for name in METHODS:
            vals = [r[mode] for r in rows if r["method"] == name]
            lines.append(f"| {name} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
    lines.append("\n## Throughput at M=1000, N=100,000 (forward)\n")
    for r in rows:
        if r["M"] == MS[-1] and r["N"] == NS[-1]:
            lines.append(f"- {r['method']}: {r['pairs'] / r['fwd_ms'] / 1e6:.2f} G evals/s, peak {r['fwd_peak_gb']:.1f} GB "
                         f"(fwd+bwd peak {r['fwd_bwd_peak_gb']:.1f} GB)")
    (RESULTS / "benchmark.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
