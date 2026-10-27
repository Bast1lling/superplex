# 03: Fitting SQ unions to instance masks (agreed principles)

## Goal
Represent every instance of a scene as a union of superquadrics (SQs), predicted DETR-style from the point cloud.
Data: Replica 1 cm point clouds (xyz, rgb, normal) with GT instance labels; whole scenes, no crops.

## Model
- Point encoder: an existing architecture (not custom) that aggregates information locally into a sparse set of tokens.
  *Choice open.*
- Queries: N SQ queries, initialized from the scene / encoder tokens rather than as free learned embeddings.
  *Exact scheme open, possibly tied to the encoder's downsampling.*
- Decoder: stack of self-attention (among queries) and cross-attention (queries -> point tokens) blocks.
- Head per query: SQ parameters (scale, exponents, rotation, translation) + existence e_j.
  No explicit instance-label head: an SQ's label is implied by the points it covers (see loss).

## Loss (start simple, extend only if needed)
For a random subset of points, sampled roughly uniformly across instances:

1. SDF d_pj of every point p to every SQ j. Estimator is a switch: `radial` (default, best in experiment 05),
   `gauge_lb` or `taubin` (from experiment 02).
2. Soft membership m_pj in [0, 1] via a smooth function of d_pj: 1 on the surface / inside, 0 far away.
   Weight w_pj = e_j * m_pj.
3. Implicit SQ labels by smooth majority vote: l_j = sum_p w_pj onehot(y_p) / sum_p w_pj.
4. Point prediction: q_p = sum_j pi_pj l_j with ownership pi_pj = w_pj / sum_j' w_pj', plus a small "nothing"
   component (constant weight, uniform labels), so uncovered points are penalized and pi is always defined.
5. Label loss: CE(q_p, y_p). Points of other instances act as negatives: an SQ covering several instances
   gets a mixed label and is penalized.
6. Surface regularizer: the SDF of a point to its assigned SQ must be > -delta, i.e. an SQ may swallow the true
   surface by at most delta (hinge on deeper points). delta > 0 lets the simple SQ shape approximate the surface
   while still enclosing the relevant points. Hyperparameter; start with SuperFlex's truncation 0.05
   (relative to their unit-size objects, i.e. 5 cm in Replica meters).

## Deferred
- Inference-time grouping of SQs into instances (labels only exist during training).
- Overlap / sharp-assignment and sparsity regularizers. One could force the point assignment to be sharp
