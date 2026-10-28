# 09 — Decomposition without semantics: SQs for the whole scene as one blob

**Question.** What happens if the scene is decomposed without any labels, i.e. SQs are fitted to one blob of solid
voxels that represents the whole scene? Which parts of the semantic decomposition does pure geometry recover?

**TL;DR.**
- With experiment 10's quality-driven search (a part gets one SQ if one SQ explains it with tolerant IoU ≥ 0.8;
  top-down splitting + merging), the whole room needs about as many SQs as the instance-aware decomposition
  (office_0 109 vs 94, room_0 149 vs 141), and the SQs are mostly **pure**: 78–83 % of the voxels inside an SQ belong
  to one GT instance (56–62 % of the SQs are ≥ 80 % pure).
- **Free-standing objects are recovered by geometry alone**: sofas, armchairs, desks, chairs, poufs, the plant
  come out with nearly the same SQs as with semantics (see the renders).
- **Everything that is flush with something else disappears**: rugs merge into the floor slab, windows / blinds /
  pictures / doors / switches into wall slabs, cushions into the sofa, objects on a table into one cluster with it.
  47 % (office_0) to 62 % (room_0) of the instances get no SQ of their own — mostly these flush, thin objects.
- **The room shell is fragmented**: walls become several rounded boards with many small pieces at the wall–floor
  junction (baseboards, wall-mounted objects); the floor splits where rugs / furniture sit on it.
- So semantics are needed exactly where geometry has no boundary (flush / co-planar objects, contact regions);
  with semantics the same search gives one slab per wall and one SQ per rug / picture / window (experiment 10).
  A label-free pipeline would need at least appearance (color) or a planar-structure prior for the shell.

Per-instance vIoU here is not a quality score of the shapes — it measures how well GT instances are recovered by
the SQs assigned to them (instances without SQ count 0).

## Method (`code/blob.py`)
- Targets: all solid voxels of experiment 08's labeled volume (instance labels only used to build the volume:
  which voxels are solid / free / unknown / exterior). No planar fill (it needs instances). Groups = connected
  components of the solid (26-neighborhood, ≥ 8 voxels): 4–6 per scene (the room shell + everything touching it is
  one component).
- Decomposition: experiment 10's `topdown` + merge (`method=both`, q = 0.8, tolerant IoU, same loss; overlap with
  other parts of the blob costs `w_sib` = 0.1).
- Evaluation against the GT (never seen by the fit): every SQ is assigned to the instance owning most solid voxels
  inside it. `purity` = that share; per-instance K and strict vIoU of the assigned SQs (same scorer as 10);
  `instances_without_sq`.

## Results

K columns = SQs assigned to instances of that class (0: the class's instances got no SQ).

| run | SQs | vIoU | prec | rec | plausible | over | K indoor-plant | K table | K chair | K lamp | K wall | K rug | K floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 09:office0_both_q08_tol (purity 0.83, no SQ 0.47) | 109 | 0.345 | 0.422 | 0.409 | 0.23 | 0.30 | 24.0 | 3.0 | 2.0 | 0.0 | 5.2 | 6.0 | 0.0 |
| 09:office0_both_q09_tol (purity 0.81, no SQ 0.42) | 242 | 0.374 | 0.470 | 0.439 | 0.05 | 0.53 | 33.0 | 6.0 | 4.5 | 0.0 | 14.2 | 14.0 | 2.0 |
| 09:office0_both_q08_wsib1 (purity 0.84, no SQ 0.42) | 165 | 0.370 | 0.463 | 0.439 | 0.12 | 0.45 | 24.0 | 3.5 | 3.0 | 0.0 | 10.4 | 11.0 | 2.0 |
| 09:room0_both_q08_tol (purity 0.78, no SQ 0.62) | 149 | 0.228 | 0.292 | 0.276 | 0.13 | 0.26 | 6.0 | 1.6 | 3.0 | 2.2 | 10.0 | 6.0 | 5.0 |
| 09:room1_both_q08_tol (purity 0.79, no SQ 0.61) | 111 | 0.237 | 0.302 | 0.283 | 0.12 | 0.27 | 4.0 | – | – | 0.8 | 14.2 | 3.0 | 7.0 |

- Stricter quality (q 0.9) or expensive overlap between blob parts (`w_sib` 1) only add fragments (more wall / rug
  pieces); they do not separate flush objects — there is no geometric boundary to find.
- office_0: the floor SQs are assigned to the large rug (it covers most of the floor), so "floor 0, rug 6".


## Files
- `code/blob.py` (imports experiment 10's `partfit.py` / `decomp.py`), `results/runs/<run>/` (sqs.npz also stores
  `component` and `purity` per SQ), `results/logs/`.
- `viewer/server.py` wrapper of 08's viewer (port 8009) serving `results/viewer_runs/`; SQs are colored by their
  assigned GT instance, the voxel layer shows the instance-labeled volume (the fit only used solid / free / ...).

    cd code && uv run python blob.py --name <run> --set scene=room_0 [--set q=0.9 --set w_sib=1.0 ...]
    uv run python ../../08_volumetric_decomp/code/export_viewer.py --exp 09 <run>   # -> results/viewer_runs
    uv run python ../viewer/server.py      # -> http://localhost:8009
