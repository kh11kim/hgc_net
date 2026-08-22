# Issue #59 — stages 1–3 implementation preparation

## What is fixed now

`contract/issue59_upstream_canonical_v3.json` pins the official HGC-Net base commit
at `abca11a6de36b51c738166771ac7c794a2f6cb21`. The validator requires that base to
exist and be an ancestor of `HEAD`, so a reviewed implementation commit may follow
it. It hashes the architecture,
loss/decode, dataset, training, label-matching/sampling, post-processing sources,
and bundled `model/model_027.pth` checkpoint. The validator checks those files and
the canonical unified-v3 metadata only; it never loads sample arrays, invokes CUDA,
or trains. It also requires the resolved remote layout: writable
`origin=kh11kim/hgc_net`, with the official source retained as `upstream`.

The canonical input contract is:

- `/home/irsl/datasets/dlr/compiled/scdm_justin_right_vgn_train_10000_reconstruction_view_aligned_v3`
- seed-0 scene split: train/val/test = 8,995 / 510 / 495
- 10,000 scenes and views; 31,537 eligible objects; 4,271,648 positive and
14,650,552 negative grasps
- `justin_right_hand_simple`, palm frame `palm`, base link
  `right_hand_base_link`, and the declared ordered 12-DoF joint vector

It also pins the metadata indexes `index/scenes.jsonl`, `index/views.jsonl`, and
`index/samples.jsonl` (10,000 lines each). Hashing the bulk depth, segmentation,
grid, and grasp payload trees is deliberately deferred to a later provenance gate;
this stage never opens those arrays.

Run the gate before any stage-4 code:

```bash
cd /home/irsl/ws/dlr/HGC-Net
python3 tools/validate_issue59_contract.py
python3 -m unittest tests/test_issue59_contract.py
```

## Stage status

1. **Upstream fixed** — complete: the git base, source behavior, and supplied checkpoint are hash-pinned.
2. **Development repository prepared** — complete: `origin` is the writable
   `kh11kim/hgc_net` fork and `upstream` is `yimingli1998/hgc_net`; local branch
   `codex/issue59-upstream-adapter` remains at the pinned base.
3. **Canonical data contract fixed** — complete for metadata: unified-v3 manifest, seed-0 split, counts, and ordered Justin-right 12-DoF declaration are pinned.

The writable fork is now `origin`; the official source stays at `upstream`. The
upstream checkout has no license file. The user has explicitly treated that as a
non-blocker for this work; this record makes no legal conclusion or distribution
claim.

## Paper-modified lineage (context only)

`/home/irsl/ws/scdm_final@acc313f4eeff48bb1e80ef68b9bc2719a33e5832`
contains the existing paper-modified HGC implementation (`src/scdm/baselines/hgc.py`)
and HGC configs/checkpoint references. It is a 3D-FPN, voxel-grid implementation,
not the official PointNet++ baseline, so it is evidence for historical provenance
only. Stages 1–3 neither copy from nor alter it.

This paragraph records the historical scope of stages 1–3.  The current decision
is to use that implementation as a source reference and create a new, independent
paper-modified 3D-FPN arm inside this HGC-Net repository.  See
[hgcnet_model_and_data.md](hgcnet_model_and_data.md) for the current status and
implementation plan.

## Stage 4 planning record — superseded by the implemented adapter

The current implementation, validation commands, and documented behavioral deltas
are in [issue59_stage4_adapter.md](issue59_stage4_adapter.md).  The text below is
the original planning record retained for provenance.

Keep the upstream PointNet++ encoder and point-wise pose representation as the
baseline, but change the hand-specific output boundary to Justin-right. This is a
Justin model, not a DLR-hand model followed by a 20-to-12 conversion. Add a separate
canonical-data adapter and Justin-right head, with tests that make every transform
explicit. The planned flow is:

```text
unified-v3 one view -> visible surface point-cloud adapter -> frozen upstream network
network point-wise pose -> Justin-right pose/joint output -> Justin candidate
```

The adapter must consume one canonical view and derive a visible metric point cloud
from its indexed `depth/*.png` and the pointed-to `view_meta/*.json` camera
intrinsics and `T_reconstruction_grid_camera`; unified v3 does not provide a
precomputed point cloud. It must make deprojection, the camera-to-canonical-frame
conversion, and any subsequent crop explicit before
`model.forward(point, normalized_point)`. The fixed input is **25,000 points per
view**. In a 100-view stride sample after deprojection and a 0.5 m
reconstruction-grid crop, visible point counts were 12,920 at the smallest, 31,483
at the middle, and 56,448 at the largest; 13 of 100 views have fewer than 25,000.
When there are at least 25,000 visible points, select 25,000 distinct points
deterministically. When there are fewer, keep every visible point and repeat only
the missing number of points deterministically. This makes every model input the
same size without inventing new geometry.

## Resolved inputs for Stage 4

1. **Hand boundary:** change the gripper/output to Justin-right; do not build a
   20-to-12 joint adapter.
2. **Positive label anchor:** use the canonical positive grasp's `approach_point`
   to find the corresponding visible surface point.
3. **Point count:** use exactly 25,000 points per view. Sample without replacement
   when possible; otherwise retain all real points and deterministically repeat
   existing points only to fill the shortfall.
4. **Post-processing:** after a fixture verifies the predicted approach-axis sign,
   apply the frame-explicit top-side rule: retain candidates whose palm lies on the
   +world-z side of their grasp point. Apply the exact upstream aggressive 3 cm /
   30 degree NMS: suppress the lower-scored candidate when either it is within 3 cm
   or within 30 degrees of a kept candidate. Log candidate counts before and after
   NMS.

## Former open questions (resolved in the Stage 4 record)

1. **Frame and pose convention:** make the exact depth units, invalid-depth rule,
   camera-to-canonical transform, palm/approach direction, and 0.5 m crop concrete
   in a fixture test.
2. **Negative labels:** decide how canonical explicit negative grasps become
   point-wise negative labels; the positive anchor is now fixed, but the negative
   rule is not.
