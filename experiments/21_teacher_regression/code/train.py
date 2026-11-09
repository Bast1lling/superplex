"""Does SuperFlex's network just learn from other chairs how chairs are decomposed? Same architecture, supervised only.

    python train.py --name student [--epochs 600]

Teacher: the pretrained `superflex` checkpoint. Student: the same architecture from random weights, with one loss:
L1 on the teacher's outputs slot by slot (its slots are positional) for the slots it uses, BCE on which ones it uses.
No geometry, no occupancy. Data: the 1355 test chairs split into mini-train and mini-test, the director's chair in
mini-test. Both are held out from the teacher's own training.

Before training: baselines on mini-test (teacher, and k-means with the teacher's count). After it: the student, the same
way. "pred" is the plain SQs as they are, "opt" after 1000 steps of SuperFlex's objective, all on the fixed occupancy.
Fits go to `<FITS>/<name>_<kind>.npz`, scores to results/<name>.json.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
os.environ.setdefault("TRACKIO_DIR", str(EXP / "results" / "large_files" / "trackio"))
sys.path.insert(0, str(ROOT / "experiments" / "19_sq_count_sweep" / "code"))
import trackio  # noqa: E402
from fit_sqs import DATA, DEV, kmeans_boxes, load_chair  # noqa: E402
from fit_superflex import CHECKPOINT, OmegaConf, PredictionHandler, SuperFlex, extend_dict, fit, predict  # noqa: E402

PROJECT = "teacher_regression"
TARGET = "fbc560d04067a6b58ce66f12ba927a2b"
FITS = DATA.parent / "sq_fits" / "exp21"
TEST_SIZE, BATCH, EVAL_BATCH, LR = 271, 32, 64, 3e-4
SLOTS = 16


def split():
    ids = (DATA / "test.lst").read_text().split()
    rest = [i for i in ids if i != TARGET]
    rest = [rest[i] for i in np.random.default_rng(0).permutation(len(rest))]
    return rest[TEST_SIZE - 1:], [TARGET] + rest[:TEST_SIZE - 1]


def build(state=None):
    model = SuperFlex(OmegaConf.load(CHECKPOINT / "config.yaml").superflex).to(DEV)
    if state is not None:
        model.load_state_dict(state)
    return model


@torch.no_grad()
def teacher_outputs(teacher, surf):
    out = [{key: value for key, value in teacher(chunk).items() if key != "assign_matrix"} for chunk in surf.split(EVAL_BATCH)]
    return {key: torch.cat([o[key] for o in out]) for key in out[0]}


def angles(a):
    return torch.stack([a.sin(), a.cos()], -1)


def regression_loss(out, target):
    """L1 per parameter on the slots the teacher uses, BCE on which those are."""
    used = target["exist"][..., 0] > 0.5
    geometry = sum(F.l1_loss(out[key][used], target[key][used]) for key in ("trans", "scale", "shape", "rotate", "tapering", "bending_k"))
    geometry = geometry + F.l1_loss(angles(out["bending_a"])[used], angles(target["bending_a"])[used])
    exist = F.binary_cross_entropy(out["exist"][..., 0], used.float())
    return geometry + exist, geometry, exist


def kmeans_handler(chairs, counts):
    """K-means boxes with each chair's count, in 16 slots."""
    B = len(chairs)
    p = {"scale": np.ones((B, SLOTS, 3)), "translation": np.zeros((B, SLOTS, 3)), "rotation": np.tile(np.eye(3), (B, SLOTS, 1, 1)),
         "exponents": np.ones((B, SLOTS, 2)), "exist": np.zeros((B, SLOTS)), "tapering": np.zeros((B, SLOTS, 2)), "bending": np.zeros((B, SLOTS, 6))}
    for b, (chair, K) in enumerate(zip(chairs, counts)):
        box, _ = kmeans_boxes(chair["surf"], int(K))
        for key, value in box.items():
            p[key][b, :int(K)] = value
        p["exist"][b, :int(K)] = 1
    return PredictionHandler({"names": np.arange(B), "pc": np.stack([c["surf"] for c in chairs]), "assign_matrix": np.zeros((B, 1, SLOTS)), **p})


def run(make_handler, chairs, steps):
    """Fits of all chairs in batches, make_handler(batch of chairs, surface points)."""
    parts = []
    for start in range(0, len(chairs), EVAL_BATCH):
        batch = chairs[start:start + EVAL_BATCH]
        surf = np.stack([c["surf"] for c in batch])
        parts.append(fit(make_handler(batch, surf), batch, surf, True, steps))
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}


def scores(r, thin):
    return {"iou": float(r["iou"].mean()), "iou_thin": float(r["iou"][thin].mean()), "iou_rest": float(r["iou"][~thin].mean()),
            "target": float(r["iou"][0]), "count": float(r["exist"].sum(1).mean())}


def evaluate_all(name, make_handler, chairs, thin, ids, report):
    for kind, steps in (("pred", 0), ("opt", 1000)):
        r = run(make_handler, chairs, steps)
        report[f"{name}_{kind}"] = scores(r, thin)
        np.savez(FITS / f"{REPORT_NAME}_{name}_{kind}.npz", ids=np.array(ids), **r)
        print(name, kind, report[f"{name}_{kind}"], flush=True)


def main():
    global REPORT_NAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="student")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--skip_baselines", action="store_true")
    args = ap.parse_args()
    REPORT_NAME = args.name
    FITS.mkdir(parents=True, exist_ok=True)
    (EXP / "results").mkdir(exist_ok=True)
    torch.manual_seed(0)

    train_ids, test_ids = split()
    print(len(train_ids), "mini-train,", len(test_ids), "mini-test", flush=True)
    train_chairs = [load_chair(i) for i in train_ids]
    test_chairs = [load_chair(i) for i in test_ids]
    quality = json.loads((DATA.parent / "quality.json").read_text())["objects"]
    all_occ = np.array([quality[i]["occupancy"] for i in train_ids + test_ids])
    thin = np.array([quality[i]["occupancy"] for i in test_ids]) < np.percentile(all_occ, 25)

    teacher = build(torch.load(CHECKPOINT / "ckpt.pt", map_location=DEV, weights_only=False)["model_state_dict"]).eval()
    train_surf = torch.tensor(np.stack([c["surf"] for c in train_chairs]), device=DEV)
    test_surf = torch.tensor(np.stack([c["surf"] for c in test_chairs]), device=DEV)
    train_target, test_target = teacher_outputs(teacher, train_surf), teacher_outputs(teacher, test_surf)
    counts = (test_target["exist"][..., 0] > 0.5).sum(1).cpu().numpy()
    print("teacher count on mini-test", counts.mean(), flush=True)

    report = {"thin_chairs": int(thin.sum()), "chairs": len(test_ids)}
    if not args.skip_baselines:
        evaluate_all("teacher", lambda batch, surf: predict(teacher, surf, True), test_chairs, thin, test_ids, report)
        place = {id(c): int(n) for c, n in zip(test_chairs, counts)}
        evaluate_all("kmeans", lambda batch, surf: kmeans_handler(batch, [place[id(c)] for c in batch]), test_chairs, thin, test_ids, report)

    student = build()
    optimizer = torch.optim.Adam(student.parameters(), lr=LR)
    steps_per_epoch = len(train_chairs) // BATCH
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, epochs=args.epochs, steps_per_epoch=steps_per_epoch)
    trackio.init(project=PROJECT, name=args.name, config=vars(args) | {"lr": LR, "batch": BATCH, "train": len(train_ids), "test": len(test_ids)},
                 auto_log_gpu=False, auto_log_cpu=False)
    step, tic = 0, time.time()
    for epoch in range(args.epochs):
        student.train()
        order = torch.randperm(len(train_chairs), device=DEV)
        totals = np.zeros(3)
        for b in range(steps_per_epoch):
            index = order[b * BATCH:(b + 1) * BATCH]
            out = student(train_surf[index])
            target = {key: value[index] for key, value in train_target.items()}
            loss, geometry, exist = regression_loss(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            totals += [loss.item(), geometry.item(), exist.item()]
            step += 1
        logs = dict(zip(("train/loss", "train/geometry", "train/exist"), (totals / steps_per_epoch).tolist())) | {"lr": scheduler.get_last_lr()[0], "epoch": epoch}
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            student.eval()
            with torch.no_grad():
                held = [student(chunk) for chunk in test_surf.split(EVAL_BATCH)]
            out = {key: torch.cat([h[key] for h in held]) for key in test_target}
            test_loss, test_geometry, test_exist = regression_loss(out, test_target)
            r = run(lambda batch, surf: predict(student, surf, True), test_chairs, 0)
            s = scores(r, thin)
            logs |= {"test/loss": test_loss.item(), "test/geometry": test_geometry.item(), "test/exist": test_exist.item(),
                     "test/iou": s["iou"], "test/iou_thin": s["iou_thin"], "test/iou_rest": s["iou_rest"], "test/iou_target": s["target"], "test/count": s["count"]}
            for name in ("teacher_pred", "kmeans_opt", "teacher_opt"):
                if name in report:
                    logs[f"baseline/{name}"] = report[name]["iou"]
            print(f"epoch {epoch + 1} {time.time() - tic:.0f}s train {logs['train/loss']:.4f} test {logs['test/loss']:.4f} iou {s['iou']:.3f} thin {s['iou_thin']:.3f} target {s['target']:.3f} count {s['count']:.1f}", flush=True)
            torch.save(student.state_dict(), FITS / f"{args.name}.pt")
        trackio.log(logs, step=step)

    student.eval()
    evaluate_all("student", lambda batch, surf: predict(student, surf, True), test_chairs, thin, test_ids, report)
    trackio.log({f"final/{name}": value["iou"] for name, value in report.items() if isinstance(value, dict)}, step=step)
    trackio.finish()
    (EXP / "results" / f"{args.name}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
