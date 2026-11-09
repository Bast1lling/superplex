
# Latency 2026-11-06 14:11, NVIDIA GeForce RTX 5090, batch 8, 100 steps

## dense

plain step (unwrapped, no syncs): 153 ms; DataLoader wait per step (6 workers): 0.1 ms; one frame loaded + collated in a single process: 6 ms

per training step:
| part | calls | inclusive ms | self ms |
|---|---|---|---|
| step.loss | 1.0 | 76.3 | 5.8 |
| step.backward | 1.0 | 49.1 | 49.1 |
| dense.heatmap | 8.0 | 32.9 | 32.9 |
| train.surface | 24.0 | 22.1 | 22.1 |
| step.forward | 1.0 | 17.5 | 0.0 |
| model.encode | 1.0 | 17.5 | 10.7 |
| dense.assign | 8.0 | 11.0 | 3.7 |
| train.pixel_owner | 8.0 | 7.3 | 1.3 |
| train.sdf_batch | 8.0 | 6.0 | 6.0 |
| model.encoder | 1.0 | 4.2 | 4.2 |
| step.to_device | 1.0 | 3.1 | 3.1 |
| train.chamfer | 16.0 | 1.9 | 1.9 |
| model.enc_heads | 1.0 | 1.4 | 1.4 |
| train.instance_masks | 8.0 | 1.3 | 1.3 |
| dense.center_focal | 8.0 | 1.3 | 1.3 |
| model.tokenizer_mlp | 1.0 | 1.2 | 1.2 |
| step.optimizer | 1.0 | 0.8 | 0.8 |

per test batch of 8 (evaluate):
| part | calls | inclusive ms | self ms |
|---|---|---|---|
| eval.total | 0.1 | 210.9 | 10.4 |
| dense.frame_metrics | 8.0 | 171.5 | 4.0 |
| dense.pair_iou_matrix | 8.0 | 125.6 | 1.2 |
| train.pair_iou | 33.3 | 124.4 | 9.2 |
| train.sdf_batch | 106.6 | 91.0 | 91.0 |
| train.surface | 74.6 | 67.8 | 67.8 |
| dense.occupancy_iou | 16.0 | 34.8 | 4.0 |
| model.encode | 1.0 | 16.8 | 10.5 |
| dense.assign | 8.0 | 11.2 | 3.8 |
| train.pixel_owner | 8.0 | 7.3 | 1.2 |
| model.encoder | 1.0 | 4.1 | 4.1 |
| model.tokenizer_mlp | 1.0 | 1.2 | 1.2 |
| model.enc_heads | 1.0 | 1.1 | 1.1 |
| dense.visible_targets | 1.0 | 1.0 | 1.0 |
| model.peaks | 8.0 | 0.4 | 0.4 |

## topk

plain step (unwrapped, no syncs): 231 ms; DataLoader wait per step (6 workers): 0.1 ms; one frame loaded + collated in a single process: 6 ms

per training step:
| part | calls | inclusive ms | self ms |
|---|---|---|---|
| step.loss | 1.0 | 142.1 | 0.1 |
| step.backward | 1.0 | 86.4 | 86.4 |
| train.surface | 64.0 | 71.5 | 71.5 |
| dense.losses | 1.0 | 71.0 | 5.9 |
| train.losses | 1.0 | 58.7 | 5.0 |
| dense.heatmap | 8.0 | 36.1 | 36.1 |
| step.forward | 1.0 | 20.3 | 0.3 |
| model.encode | 1.0 | 18.1 | 10.8 |
| train.match | 8.0 | 17.8 | 2.1 |
| dense.assign | 8.0 | 11.3 | 3.9 |
| train.pixel_owner | 8.0 | 7.4 | 1.3 |
| train.sdf_batch | 8.0 | 6.1 | 6.1 |
| train.chamfer | 32.0 | 4.7 | 4.7 |
| model.encoder | 1.0 | 4.4 | 4.4 |
| step.to_device | 1.0 | 3.7 | 3.7 |
| model.enc_heads | 1.0 | 1.6 | 1.6 |
| train.instance_masks | 16.0 | 1.6 | 1.6 |
| dense.center_focal | 8.0 | 1.5 | 1.5 |
| model.tokenizer_mlp | 1.0 | 1.3 | 1.3 |
| step.optimizer | 1.0 | 1.3 | 1.3 |
| train.linear_sum_assignment | 8.0 | 1.1 | 1.1 |
| dense.visible_targets | 1.0 | 1.0 | 1.0 |
| model.decoder_layer | 1.0 | 1.0 | 1.0 |
| model.dec_heads | 1.0 | 0.7 | 0.7 |
| train.mask_targets | 8.0 | 0.5 | 0.2 |
| model.peaks | 1.0 | 0.1 | 0.1 |

per test batch of 8 (evaluate):
| part | calls | inclusive ms | self ms |
|---|---|---|---|
| eval.total | 0.1 | 707.9 | 338.6 |
| dense.frame_metrics | 8.0 | 169.7 | 4.1 |
| train.frame_metrics | 8.0 | 168.3 | 67.4 |
| train.pair_iou | 41.3 | 143.3 | 10.4 |
| dense.pair_iou_matrix | 8.0 | 123.6 | 1.2 |
| train.sdf_batch | 146.6 | 109.3 | 109.3 |
| train.surface | 122.6 | 105.5 | 105.5 |
| train.group_instances | 8.0 | 38.2 | 38.2 |
| dense.occupancy_iou | 16.0 | 35.4 | 3.8 |
| model.encode | 1.0 | 17.3 | 10.5 |
| train.match | 8.0 | 15.9 | 1.9 |
| dense.assign | 8.0 | 11.4 | 4.0 |
| train.pixel_owner | 8.0 | 7.4 | 1.3 |
| model.encoder | 1.0 | 4.2 | 4.2 |
| model.enc_heads | 1.0 | 1.3 | 1.3 |
| model.tokenizer_mlp | 1.0 | 1.2 | 1.2 |
| model.decoder_layer | 1.0 | 1.0 | 1.0 |
| dense.visible_targets | 1.0 | 1.0 | 1.0 |
| train.chamfer | 8.0 | 0.9 | 0.9 |
| train.linear_sum_assignment | 8.0 | 0.8 | 0.8 |
| model.dec_heads | 1.0 | 0.5 | 0.5 |
| model.peaks | 9.0 | 0.4 | 0.4 |
| train.instance_masks | 8.0 | 0.3 | 0.3 |

## patch

plain step (unwrapped, no syncs): 169 ms; DataLoader wait per step (6 workers): 0.1 ms; one frame loaded + collated in a single process: 6 ms

per training step:
| part | calls | inclusive ms | self ms |
|---|---|---|---|
| step.backward | 1.0 | 83.6 | 83.6 |
| step.loss | 1.0 | 65.6 | 5.7 |
| train.surface | 40.0 | 41.0 | 41.0 |
| train.match | 8.0 | 31.5 | 8.8 |
| step.forward | 1.0 | 20.3 | 0.1 |
| model.decoder_layer | 3.0 | 12.7 | 12.7 |
| train.linear_sum_assignment | 8.0 | 6.7 | 6.7 |
| model.tokenizer | 1.0 | 5.9 | 5.9 |
| step.to_device | 1.0 | 3.4 | 3.4 |
| train.chamfer | 16.0 | 2.8 | 2.8 |
| model.heads | 1.0 | 1.5 | 1.5 |
| step.optimizer | 1.0 | 1.1 | 1.1 |
| train.mask_targets | 8.0 | 0.6 | 0.2 |
| train.instance_masks | 8.0 | 0.4 | 0.4 |

per test batch of 8 (evaluate):
| part | calls | inclusive ms | self ms |
|---|---|---|---|
| eval.total | 0.1 | 310.7 | 193.8 |
| train.frame_metrics | 8.0 | 98.9 | 12.3 |
| train.surface | 48.0 | 39.5 | 39.5 |
| train.match | 8.0 | 29.6 | 8.8 |
| train.pair_iou | 8.0 | 20.9 | 1.3 |
| train.sdf_batch | 36.9 | 16.6 | 16.6 |
| model.decoder_layer | 3.0 | 11.4 | 11.4 |
| train.group_instances | 8.0 | 11.4 | 11.4 |
| train.linear_sum_assignment | 8.0 | 7.5 | 7.5 |
| model.tokenizer | 1.0 | 5.5 | 5.5 |
| model.heads | 1.0 | 1.2 | 1.2 |
| train.chamfer | 8.0 | 1.0 | 1.0 |
| train.instance_masks | 8.0 | 0.3 | 0.3 |

# Analysis (2026-11-06)

Measured with `code/latency.py` (CUDA-synced timers; times above are per step of 8 frames). The full_topk run took
42 s / epoch = 175 steps x 0.24 s, consistent with the plain step of 231 ms.

- Data loading is free: the loop waits 0.1 ms per step for the 6 workers (a frame loads in 6 ms).
- The model is tiny: forward 17-20 ms, of which ~11 ms is the self time of `encode` = gathering the 1088 Sonata
  features for every pixel (8 x 50400 x 1088 floats, 1.75 GB) only to max-pool them per patch. Encoder 4 ms, decoder 1 ms.
- The step is overhead-bound, not compute-bound: everything is many small kernels in Python loops over frames.
  1. `train.surface` (SuperFlex's `parametric_to_points_extended`): 64 calls / step, 1.1 ms each regardless of the
     number of SQs or samples (n = 4 ... 16 all ~0.9 ms forward, 2.2 ms forward + backward for 300 SQs). A plain
     10-line version (signed powers of cos / sin, rotate, translate) gives the same points (max diff 2e-7) in 0.21 ms /
     0.72 ms. In topk ~70 ms forward + an estimated ~50 ms of the 86 ms backward are spent here.
  2. `dense.heatmap`: 33-36 ms / step on the CPU (numpy distance transform + a Gaussian per SQ); together with
     `dense.assign` (11 ms) it is a pure function of the frame and its targets, i.e. the same in every epoch.
  3. Backward 49-86 ms: mostly the surface / Chamfer graphs of the per-frame loss loops (the network is ~2 M params).
  4. Hungarian matching is cheap now (`linear_sum_assignment` 1 ms / step for 300 queries; 7 ms for 3150 patches).
- Evaluation: 0.2 s (dense) to 0.7 s (topk) per 8 frames, i.e. 8-27 s per evaluation of 200 test + 100 train frames,
  dominated by pair IoU (`pair_iou_matrix`: 300 x visible pairs x 1024 SDF points), surfaces and `group_instances`
  (Python loop, 38 ms / batch). With eval every 10 epochs (~420 s of training) that is ~5 %.

Speed-ups, by expected gain:
1. Replace SuperFlex's surface sampler by the plain one and call it once per step on all frames' SQs concatenated
   (prediction, forced and target surfaces) instead of 64 times: ~230 -> ~120 ms (topk).
2. Precompute the assignment (owner, visible) and the heatmap per frame once (in the cache, or in the Dataset worker
   process, which then costs nothing): another ~45 ms.
3. Cache the patch-max-pooled Sonata features (they are frozen): the tokenizer input becomes a (42, 75, 1088) array
   like the DINO tokens; ~10 ms and much less memory, and simpler code.
4. Batch the per-frame loss loops (pad the targets to the max per batch) and/or `torch.compile` the loss; bf16 autocast
   for the network gives little since it is not the bottleneck.
Together roughly 230 -> 60-80 ms per step (3-4x), after which a larger batch (16-32) becomes cheap too; the GPU
memory use is ~6 GB of 32 GB.

# Latency with speed-ups 1-3, 2026-11-06 14:34, NVIDIA GeForce RTX 5090, batch 8, 100 steps

## dense

plain step (unwrapped, no syncs): 48 ms; DataLoader wait per step (6 workers): 0.2 ms; one frame loaded + collated in a single process: 5 ms

per training step:
| part | calls | inclusive ms | self ms |
|---|---|---|---|
| step.backward | 1.0 | 22.2 | 22.2 |
| step.forward | 1.0 | 9.5 | 0.0 |
| model.encode | 1.0 | 9.5 | 2.9 |
| step.loss | 1.0 | 9.0 | 3.1 |
| train.pair_losses | 1.0 | 5.9 | 4.5 |
| model.encoder | 1.0 | 4.2 | 4.2 |
| step.to_device | 1.0 | 3.7 | 3.7 |
| model.enc_heads | 1.0 | 1.3 | 1.3 |
| model.tokenizer_mlp | 1.0 | 1.1 | 1.1 |
| train.surface | 3.0 | 0.8 | 0.8 |
| step.optimizer | 1.0 | 0.6 | 0.6 |
| train.chamfer | 2.0 | 0.5 | 0.5 |
| dense.assign | 8.0 | 0.0 | 0.0 |

per test batch of 8 (evaluate):
| part | calls | inclusive ms | self ms |
|---|---|---|---|
| eval.total | 0.1 | 139.2 | 8.7 |
| dense.frame_metrics | 8.0 | 120.7 | 4.1 |
| train.sdf_batch | 98.6 | 84.8 | 84.8 |
| dense.pair_iou_matrix | 8.0 | 79.3 | 1.2 |
| train.pair_iou | 33.3 | 78.1 | 9.1 |
| dense.occupancy_iou | 16.0 | 34.8 | 3.6 |
| train.surface | 74.6 | 17.5 | 17.5 |
| model.encode | 1.0 | 8.7 | 2.3 |
| model.encoder | 1.0 | 4.2 | 4.2 |
| dense.visible_targets | 1.0 | 1.2 | 1.2 |
| model.enc_heads | 1.0 | 1.1 | 1.1 |
| model.tokenizer_mlp | 1.0 | 1.1 | 1.1 |
| model.peaks | 8.0 | 0.4 | 0.4 |
| dense.assign | 8.0 | 0.0 | 0.0 |

## topk

plain step (unwrapped, no syncs): 64 ms; DataLoader wait per step (6 workers): 0.2 ms; one frame loaded + collated in a single process: 5 ms

per training step:
| part | calls | inclusive ms | self ms |
|---|---|---|---|
| step.backward | 1.0 | 29.0 | 29.0 |
| step.loss | 1.0 | 15.6 | 0.1 |
| step.forward | 1.0 | 11.5 | 0.3 |
| model.encode | 1.0 | 9.6 | 2.8 |
| train.pair_losses | 2.0 | 8.5 | 5.3 |
| dense.losses_batched | 1.0 | 7.8 | 1.7 |
| train.losses_batched | 1.0 | 6.5 | 1.1 |
| model.encoder | 1.0 | 4.2 | 4.2 |
| step.to_device | 1.0 | 3.7 | 3.7 |
| train.surface | 8.0 | 2.5 | 2.5 |
| train.match_cost | 8.0 | 2.5 | 1.5 |
| model.enc_heads | 1.0 | 1.4 | 1.4 |
| train.chamfer | 4.0 | 1.3 | 1.3 |
| model.tokenizer_mlp | 1.0 | 1.2 | 1.2 |
| dense.visible_targets | 1.0 | 1.1 | 1.1 |
| step.optimizer | 1.0 | 1.0 | 1.0 |
| train.linear_sum_assignment | 8.0 | 1.0 | 1.0 |
| model.decoder_layer | 1.0 | 0.8 | 0.8 |
| model.dec_heads | 1.0 | 0.7 | 0.7 |
| model.peaks | 1.0 | 0.1 | 0.1 |
| dense.assign | 8.0 | 0.1 | 0.1 |

per test batch of 8 (evaluate):
| part | calls | inclusive ms | self ms |
|---|---|---|---|
| eval.total | 0.1 | 586.8 | 323.9 |
| train.frame_metrics | 8.0 | 130.0 | 61.6 |
| dense.frame_metrics | 8.0 | 121.2 | 4.2 |
| train.sdf_batch | 138.6 | 103.5 | 103.5 |
| train.pair_iou | 41.3 | 90.5 | 10.2 |
| dense.pair_iou_matrix | 8.0 | 79.9 | 1.2 |
| dense.occupancy_iou | 16.0 | 34.8 | 3.8 |
| train.group_instances | 8.0 | 32.5 | 32.5 |
| train.surface | 122.6 | 29.4 | 29.4 |
| model.encode | 1.0 | 9.1 | 2.5 |
| train.match | 8.0 | 6.9 | 0.4 |
| model.encoder | 1.0 | 4.2 | 4.2 |
| train.match_cost | 8.0 | 2.5 | 1.7 |
| model.enc_heads | 1.0 | 1.2 | 1.2 |
| train.chamfer | 8.0 | 1.2 | 1.2 |
| dense.visible_targets | 1.0 | 1.1 | 1.1 |
| model.tokenizer_mlp | 1.0 | 1.1 | 1.1 |
| model.decoder_layer | 1.0 | 0.9 | 0.9 |
| train.linear_sum_assignment | 8.0 | 0.8 | 0.8 |
| model.dec_heads | 1.0 | 0.5 | 0.5 |
| model.peaks | 9.0 | 0.5 | 0.5 |
| train.instance_masks | 8.0 | 0.4 | 0.4 |
| dense.assign | 8.0 | 0.0 | 0.0 |
