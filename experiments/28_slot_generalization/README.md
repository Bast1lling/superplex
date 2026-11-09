# 28 — Does exp 27's slot model generalize to other views and scenes?

**Question.** Exp 27 overfits one Replica frame: Slot Attention on DINOv3 patches + set part decoder, predicted masks,
frustum IoU 0.878 against the instances' visible SuperFlex SQs (`R_vis_heads1e-3`, its README section 5). Does the
same model, trained on more frames, work on (a) other views of the training scene, (b) other scenes? Exp 22 failed
here (pair IoU 0.46 train → 0.19 held-out views → 0.10 new room), so the rungs go up slowly.

    uv run python experiments/28_slot_generalization/code/train.py --name F10 --train room_2 --n_train 10
    bash experiments/28_slot_generalization/worker.sh                       # runs queue.txt (done: queue.done)
    uv run python experiments/28_slot_generalization/code/dashboard.py      -> http://127.0.0.1:7828
    uv run python experiments/28_slot_generalization/viewer/server.py       -> http://localhost:8028

**Setup** (`code/train.py`'s docstring; model, losses, metrics imported unchanged from exp 27).
- Exp 27's final configuration, but K = 64 (exp 27: 128; its `K64_vis_heads1e-3` checks this on one frame), T = 3, 16 parts per slot, anchor = slot mask centroid (kept, as it should
  transfer better than absolute positions), `instance` labels, visible targets, mask + dice + exist + part_cd +
  part_exist, collapse fixes, lr 3e-5 / heads 1e-3 (not retuned for more frames; decide after the 10-frame rung).
  No augmentation.
- Splits per scene by trajectory position: first 75 % training block, last 20 % held-out views, 5 % gap.
- Rungs: room_2 1 frame → 10 frames → all 150 training-block frames; then 7 scenes, held-out scene room_0 (as exp 22).
- Metrics averaged over frames: `train`, `views` (held-out views, 20 evenly spaced), `scenes`; best.pt by views IoU.
- Viewer (`viewer/`, port 8028): per run the eval curves per split (live from trackio), a strip of every frame of
  the run's scenes colored by split (trained on / unused training block / gap / held-out views / held-out scene), and
  the model on any of them: `final` / `best` run the checkpoint on demand; 3 frames per split also keep every eval
  step. Per frame exp 27's panels (images, per-target table, SQs in 3D, masks; its backend imported) plus the frame's
  metrics.

## Results

### Ladder on room_2 (`ladder_room_2`; K = 64, 10k steps, eval every 500, one seed)

| run | training frames | batch | train: mask mIoU / frustum IoU / part IoU | held-out views (20): mask mIoU / ARI / frustum IoU / part IoU | best views IoU (step) |
|---|---|---|---|---|---|
| `F1` | 1 (000500) | 1 | 0.94 / 0.85 / 0.82 | 0.36 / 0.57 / 0.067 / 0.035 | 0.069 (9500) |
| `F10` | 10, evenly spaced over the training block | 4 | 0.91 / 0.73 / 0.62 | 0.48 / 0.64 / 0.068 / 0.038 | 0.070 (8500) |
| `F150` | all 150 of the training block (`--fast`, eval every 1000) | 4 | 0.83 / 0.47 / 0.37 | 0.62 / 0.84 / 0.091 / 0.056 | 0.099 (8000) |
| `F150_rotaug` | the same, `--rot_aug 180` (uniform SO(3)) | 4 | 0.82 / 0.20 / 0.12 | 0.67 / 0.83 / 0.097 / 0.058 | 0.106 (8000) |
| `F150_rel` | 150; relative part decoder (camera frame, no augmentation) | 4 | 0.85 / 0.59 / 0.49 | 0.59 / 0.81 / 0.190 / 0.114 | 0.190 (10000) |
| **`F150_rel_grav`** | 150; relative part decoder, gravity frame, yaw augmentation | 4 | 0.85 / 0.40 / 0.31 | 0.61 / 0.83 / **0.231** / 0.149 | 0.231 (10000, still rising) |
| `F150_rel_30k` | as `F150_rel`, 30k steps | 4 | 0.89 / 0.74 / 0.65 | 0.62 / 0.83 / 0.211 / 0.134 | 0.211 (30000) |
| **`F150_rel_grav_30k`** | as `F150_rel_grav`, 30k steps | 4 | 0.88 / 0.58 / 0.48 | 0.66 / 0.86 / **0.285** / 0.181 | 0.290 (19000) |

- **No geometric transfer yet.** 10 frames instead of 1 lift the held-out masks (mIoU 0.36 → 0.48, ARI 0.57 → 0.64),
  but the held-out frustum IoU stays at 0.07 in both: the SQs of held-out views barely overlap their targets.
- F10 fits its own 10 frames less tightly than F1 its one (0.73 vs. 0.85) and still rises at 10k steps.
- **150 frames help the masks, not the geometry.** Held-out mask mIoU 0.48 → 0.62, ARI 0.64 → 0.84, but frustum IoU
  only 0.07 → 0.09. The training frames are no longer fit (0.47 at 10k steps, still rising slowly).
- **Rotation augmentation removes the memorization but gives nothing in its place.** With random rotations of the
  input positions (SQs rotated back for the loss), the training frames drop to 0.20 and their axes are 29° off (as
  bad as held-out views), while held-out views stay at 0.10. The model cannot read orientation (or reliably position)
  off the Fourier-encoded patch positions; without augmentation it only memorized it per training view. Masks gain a
  little (held-out mask IoU median 0.75 → 0.82).
- **Geometry relative to the slot's patches + gravity frame transfers** (`F150_rel_grav`: `--part_dec rel --frame
  gravity --yaw_aug`; `fast.RelNet`: the part decoder sees patch content + position relative to the slot's anchor,
  no absolute positions; part centre = pointer-weighted mean of the slot's patch centroids + a small offset; inputs
  in a z-up frame, random yaw in training). Held-out frustum IoU 0.09 → 0.23, and with GT masks 0.10 → 0.31; the
  train / held-out gap shrinks from 0.47 / 0.09 to 0.40 / 0.23. Held-out parts with GT masks: centroid 19 → 8 cm,
  axes 27° → 13°. Masks unchanged (held-out mIoU 0.61).
- **Most of it is the relative decoder** (`F150_rel`, camera frame, no augmentation: held-out 0.09 → 0.19; GT masks
  0.10 → 0.25, centroid 8.5 cm, axes 15°). Gravity frame + yaw adds 0.04 (0.19 → 0.23; GT masks 0.25 → 0.31) and
  mostly through orientation (mean axis error 26° → 21° with GT masks), at the cost of the training-frame fit
  (0.59 → 0.40): without them the decoder still memorizes part of the orientation per view.
- **30k steps** (the first 10k identical to the 10k runs: same seed, constant lr): with gravity + yaw the held-out
  views keep improving (0.23 → 0.29 frustum IoU; with GT masks 0.31 → 0.40, centroid 6.7 cm, axes 11° median,
  thinnest extent ratio 1.12); without them the extra steps go into the training views (0.59 → 0.74) while held-out
  stays at 0.19–0.21 (GT masks 0.26). Gravity frame + yaw augmentation is what makes longer training pay off.
- Viewer: on held-out views the masks are fragmented and only loosely follow the GT instances.

### Seven scenes → room_0 (`scenes7`; rel + gravity + yaw, K = 64, 30k steps, batch 4)

Training block of office_0–4, room_1, room_2 (1050 frames; DINO kept in fp16 on the GPU, as cached); held-out views
= 20 spread over the 7 scenes; held-out scene room_0 (20 frames).

| run | train: mask mIoU / frustum IoU / part IoU | views: mask mIoU / frustum IoU / part IoU | room_0: mask mIoU / frustum IoU / part IoU |
|---|---|---|---|
| `S7_rel_grav_30k` | 0.83 / 0.31 / 0.24 | 0.66 / 0.21 / 0.15 | 0.33 / 0.12 / 0.06 |

`diagnose.py` (medians over matched targets; train / views / room_0): mask IoU 0.93 / 0.72 / **0.14**; frustum IoU
0.31 / 0.17 / 0.04; with GT mask + anchor **0.32 / 0.29 / 0.22** (centroid 7 / 10 / 12 cm, axes 15° / 17° / 22°).

- **Underfitting, not memorization:** train 0.31 vs. views 0.21, both still rising slowly at 30k (F150 on room_2
  alone: 0.58 / 0.29). Even on its training frames and with GT masks the decoder only reaches 0.32. With 1050 frames
  each one is seen ~114 times (F150: ~800). The model is small (d = 128, ~1.1 M trainable parameters).
- **The geometry transfers to the new scene, the masks do not:** with GT masks room_0 gets 0.22 (training frames
  0.32), but the predicted masks match room_0's instances with median IoU 0.14, so the frustum IoU drops to 0.04.

### Why the held-out geometry fails (`code/diagnose.py`, final checkpoints, 20 frames per split)

Per matched target (prediction = the union of its slot's active parts, target = the instance's visible SuperFlex
SQs), inside the frustum; medians:

| | F10 train | F10 views | F1 views | F150 train | F150 views | rotaug train | rotaug views | rel_grav train | rel_grav views | rel train | rel views |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mask IoU | 0.97 | 0.50 | 0.31 | 0.92 | 0.75 | 0.91 | 0.82 | 0.93 | 0.75 | 0.93 | 0.69 |
| frustum IoU | 0.79 | 0.03 | 0.02 | 0.52 | 0.06 | 0.16 | 0.06 | 0.40 | 0.22 | 0.64 | 0.13 |
| ... after shifting the prediction onto the target's centroid | 0.82 | 0.12 | 0.12 | 0.61 | 0.15 | 0.25 | 0.13 | 0.49 | 0.28 | 0.69 | 0.23 |
| ... best over all active slots (oracle assignment) | 0.79 | 0.05 | 0.05 | 0.52 | 0.07 | 0.17 | 0.08 | 0.40 | 0.24 | 0.65 | 0.15 |
| ... part decoder given the GT patch mask + GT anchor | 0.63 | 0.06 | 0.10 | 0.45 | 0.10 | 0.17 | 0.08 | 0.40 | 0.31 | 0.62 | 0.25 |
| centroid error (anchor error) | 1 cm (3 cm) | 32 cm (23 cm) | 41 cm (32 cm) | 4 cm (4 cm) | 26 cm (18 cm) | 10 cm (5 cm) | 28 cm (18 cm) | 4 cm (4 cm) | 15 cm (18 cm) | 2 cm (4 cm) | 16 cm (15 cm) |
| centroid error, GT mask + anchor | 3 cm | 23 cm | 16 cm | 5 cm | 19 cm | 10 cm | 23 cm | 4 cm | 8 cm | 2 cm | 9 cm |
| main-axis angle error (GT mask + anchor) | 2° (2°) | 36° (32°) | 45° (40°) | 6° (6°) | 30° (27°) | 29° (31°) | 35° (31°) | 10° (10°) | 18° (13°) | 4° (4°) | 21° (15°) |
| extent ratio pred / target: longest, thinnest | 0.99, 0.98 | 0.97, 1.72 | 0.96, 1.69 | 0.99, 1.04 | 1.01, 1.77 | 0.84, 1.79 | 0.98, 2.71 | 0.95, 1.02 | 0.93, 1.38 | 0.99, 1.00 | 0.95, 1.31 |

- Both parts fail on held-out views: the masks (hence the anchors) are off, and the part decoder does not transfer
  even when given the right patches and anchor (IoU 0.06–0.10, offsets 16–23 cm off, axes 30–40° off).
- Only the overall size transfers. The decoder has memorized each training frame's camera-frame offsets and
  orientations (from the slot features) rather than reading them off the patches' 3D positions; with 1–10 views of
  one room nothing forces it to.
- Not the matching: no other slot fits better (oracle 0.05).

### Training speed (`code/profile_step.py`, RTX 5090)

Per step: F1 (batch 1) 19 ms, F10 (batch 4) 74 ms (F1 10k steps in 289 s, F10 in 761 s, including the evals).
Unlike earlier experiments, the losses / surface sampling are cheap here (part matching + part Chamfer 1.5 ms per
frame). The cost was the part decoder: exp 27's version expands the token memory to one copy per slot, so
nn.MultiheadAttention projected all 3150 tokens' keys / values K times; plus it decoded parts for unmatched slots,
which get no loss. `code/fast.py` (`--fast`, default since 2026-11-08 evening; same parameters, state dicts
interchangeable): keys / values projected once per layer and shared across slots (scaled_dot_product_attention,
per-slot patch mask), parts decoded only for the matched slots during training (all slots at eval). Outputs agree with
exp 27's to 1e-7, losses / matching identical, gradients to 1e-4 relative (`check()`):

| | exp 27 part decoder | `--fast` |
|---|---|---|
| step, batch 1 | 19.0 ms | **9.7 ms** |
| step, batch 4 | 73.8 ms | **37.2 ms** |
| of it: backward (batch 4) | 36.7 ms | 16.5 ms |

What remains is mostly CPU / kernel-launch overhead (GPU busy ~4.6 ms of ~9 ms per frame): next options, if needed,
are batching the frames of a step into one forward pass, or CUDA graphs / torch.compile of the decoder. Eval costs
~90 ms per frame (IoU by importance sampling), ~4 s per eval at 20 + 20 frames, hence eval every 1000 steps now.
The viewers held ~20 GB of cached GPU memory (an OOM warning during profiling); they now free it after each request.
