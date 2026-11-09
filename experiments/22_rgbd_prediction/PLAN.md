# Plan: SQ decomposition from a single RGB-D frame

Status: being designed, nothing implemented yet. Decisions so far (2026-11-05).

## Data
- Train: the 7 NICE-SLAM scenes other than room_0; test: room_0. Every 10th frame (200 per scene).
- Input: one RGB-D frame (1200 x 680, fx = fy = 600), square crops, scale jitter.
- Targets: plain SQs from `results/targets_winding/`, moved into the camera frame of the frame: every SQ that
  intersects the camera frustum, occluded ones included (their occluded parts are only supervised by the targets).
- Per-pixel instance labels as in experiments 11-12.

## Model
- Image: DINOv3 ViT-L/16 (`facebook/dinov3-vitl16-pretrain-lvd1689m`, HF cache), frozen. One token per 16 x 16 patch.
- Geometry: Sonata (`data/models/sonata/sonata.pth`), frozen, on the frame's back-projected depth (not the cached
  whole-scene features), features precomputed per frame.
- Tokenizer: per patch a small shared MLP over its points (compressed Sonata feature + position relative to the patch
  centroid), max/mean pooled, concatenated with the DINO token, MLP -> one token. The patch's 3D centroid
  (camera frame) as positional encoding. No feature pyramid.
- Decoder: SuperFlex's (3 layers), more queries (count from the per-frame target statistics), heads: plain SQ
  parameters (no tapering, no bending), existence, class.

## Loss (initial setup, then ablated one by one)
- Supervised SQs: Hungarian matching, SuperFlex's geometric loss (Chamfer on sampled SQ surfaces + "forced" term with
  the target's translation) + existence BCE.
- Class per SQ: cross entropy on matched pairs.
- Instance masks (option A): every SQ query predicts the visible mask of its whole instance (Mask2Former mask head:
  query embedding . per-pixel embedding, BCE + Dice on matched queries); instances at inference = queries whose masks
  overlap (IoU > 0.5). Part / hierarchical masks later, no labels for them yet.
- Unsupervised: one-sided Chamfer (depth points -> nearest predicted SQ surface, via the SDF) and free space:
  ~4096 valid pixels, ~16 stratified samples per ray in [0.1 m, d - 5 cm], loss = mean soft occupancy of the
  predicted union there (exist . sigmoid(-sdf / 1 cm), max over SQs).

## Later (noted, not in the first setup)
- Volumetric pair loss instead of Chamfer: truncated SDF L1 at points around / inside the target SQ (dense gradient
  without overlap, unlike soft IoU), also with the target's translation ("forced").
- Soft-IoU occupancy term on query points labelled from the per-instance occupancy.
- Count supervision (beyond existence), overlap penalty between SQs of different instances.
- Reprojection: render predicted SQs (depth + instance id) into the input view, and into neighbouring views with
  known poses (self-supervised amodal shape).
- Unlabelled Replica instances (office_4's conference table is `non-plane`): add them as instances?
- Fine-tuning DINOv3 / EoMT-style queries inside the ViT.

## Open questions
- Decoder width (start: SuperFlex's d = 128, 1 head).

## Per-frame targets (`code/frame_stats.py`, every 10th frame, full 1200 x 680 view)
SQs in the frustum per frame: median 30-98, 95th percentile 53-182, max 187 (office_3); 20-35 % of them belong to
instances not visible at all. Instances in the frustum: median 14-36. -> 200 queries (crops see fewer).
