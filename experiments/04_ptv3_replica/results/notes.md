# 04 PTv3 / Sonata on Replica: notes

## Setup (own uv env in this folder, see pyproject.toml)
- PTv3 implementation + Sonata weights from facebookresearch/sonata (third_party/sonata, commit 18c09ff).
  Weights cached in Thesis/data/models/sonata.
- spconv-cu126 2.3.8 runs on the RTX 5090 / torch 2.14 cu130 (checked against dense conv3d).
- torch_scatter: no wheels for torch 2.14 -> shim with torch.segment_reduce (third_party/shims), checked vs reference.
- No flash-attn: patched Sonata's non-flash attention to use torch SDPA (marked "[superplex patch]" in
  third_party/sonata/sonata/model.py; sonata is a git submodule, so after cloning apply it with
  `git -C third_party/sonata apply ../sonata_sdpa.patch`). Identical to the explicit softmax path (max diff 4e-6 at fixed seed).
- Note: PTv3 shuffles its serialization orders on every forward (shuffle_orders=True), also in eval mode, so
  features differ slightly between runs unless the seed is fixed / shuffling disabled.

## Size and cost (probe.json; bf16 autocast, whole scenes, 2 cm input grid)
- Sonata = encoder-only PTv3, 108.5M params, enc channels 48/96/192/384/512, 4 poolings by 2 (stage 4 ~ 32 cm cells).
- Inference: 0.08 s / 1.4 GB (room_0, 445k pts) ... 0.38 s / 5.0 GB (apartment_0, 2.1M pts).
- Training step (full fine-tune, dummy loss): 8.7-15.3 GB for rooms / frl_apartment_0; apartment_0 OOM at 2 cm.
  At 4 cm input every scene fits (apartment_0 10.3 GB, 0.18 s).
- Tokens at the coarsest stage: 1.5k (room_0) ... 7.4k (apartment_0) at 2 cm input.

## Instance structure (analysis.md)
- Token purity w.r.t. GT instances: stage 2 (8 cm) ~0.96, stage 3 (16 cm) ~0.91, stage 4 (32 cm) ~0.79-0.84.
- Instance coverage (instances that are the majority of >= 1 token): stage 3 0.70-0.82, stage 4 only 0.46-0.73.
  -> Initializing queries from the coarsest tokens would miss many small instances; stage 2/3 are safer.
- Zero-shot feature similarity: distance-controlled same-instance AUC is ~0.68-0.74 for Sonata and ~0.68-0.76
  for a randomly initialized PTv3 -> frozen Sonata features carry little *extra* instance information beyond
  what color/normal inputs already give, at least for cosine similarity. A trained head (linear probe or our
  decoder) is the fairer test; pretrained features usually only pay off there.
