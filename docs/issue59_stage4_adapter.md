# Issue #59 Stage 4 — native Justin-right adapter

## Implemented path

`depth PNG` → deproject with the indexed `view_meta` intrinsics → apply the
indexed `T_reconstruction_grid_camera` → retain the centred 0.5 m reconstruction
grid → deterministically produce exactly 25,000 points → PointNet++ → three
Justin template heads (`finger2`, `finger3`, `finger4`) → common runtime
candidate fields.

The immutable official files (`model.py`, `loss_utils.py`, `dataset.py`, and
their contract-pinned dependencies) are not edited.  `justin_hgc/` is the
reviewable adapter boundary.  Its PointNet++ set-abstraction / feature-propagation
layer sizes are copied from official HGC-Net with normalized points enabled.

## Labels

- `thumb_visible_mask` is not used as a supervision filter. Positives whose
  palm local +z / approach direction cosine is below `0.9999` are excluded so
  the 10-degree fallback-ray grasps never become training labels.
- Its canonical `approach_point` labels every sampled point within 5 mm.
  Overlaps are resolved by nearest approach point, then source-grasp index.
- Ten percent of all remaining visible points are deterministic negatives for all
  three templates.  Everything else is ignored, preserving the official sparse
  `{-1, 0, 1}` point-supervision style.
- The view-aligned-v5 payload has `negative_palm_pose9d` and `negative_q_contact`, but
  **no negative approach point**.  We therefore do not invent a surface anchor
  for those rows or silently turn them into point labels.

The label loop is per positive grasp and keeps an `N x 3` best-match state; it
does not allocate an `N x number_of_grasps` distance matrix.

## Justin hand boundary and deliberate deltas

`grasp_type_idx` is checked against the KMK YAML insertion order
`finger2, finger3, finger4`. Each head predicts native Justin 12-DoF
`q_contact` in physical joint units (radians). There is no 20-to-12 mapper and
no joint normalization. The server emits neither `q_open` nor `q_squeeze`;
the grasp-sim client derives both from `q_contact` through
`JustinContactWaypointMapper` and its Jacobian/FK seam.

The official HGC depth target is DLR-specific: `(depth_cm - 20)` in an 8 cm
range. It cannot represent canonical Justin palms. The adapter therefore
retains the exact bin-plus-residual representation and all angular bins, but
uses direct approach-point-to-palm depth in centimetres. A whole canonical
view-aligned-v5 audit over all direct-axis positives found a maximum distance
of `0.29121178 m`, so `[0, 30)` contains the audited targets.
The loss reports and rejects any positive label outside this range before its
internal upstream-compatible clamp could hide it.

## Runtime filters

`decode_justin_candidates()` returns `palm_pose`, `grasp_point`,
`template_index`, `quality`, and `q_contact`.

1. It keeps only palms whose `palm_position - grasp_point` has positive dot
   product with explicit reconstruction-grid world-up `(0,0,1)`.
2. It applies upstream's aggressive NMS exactly: a lower-score candidate survives
   only if it is **both** farther than 3 cm **and** farther than 30 degrees from
   every kept candidate.
3. The server may return internal counts for validation, but the evaluator does
   not persist NMS diagnostics as experiment metrics.

The geometry fixture verifies that canonical local +z points from palm to the
approach point, so the surface-anchor representation uses `-R[:,2]` to point
from surface to palm.

## Commands

Run the immutable contract and CPU-only adapter tests in the HGC-Net-local uv
environment:

```bash
cd /home/irsl/ws/dlr/HGC-Net
uv sync --python 3.12
uv run python -m unittest tests/test_justin_hgc_adapter.py -v
uv run python tools/validate_issue59_contract.py
```

Before a GPU smoke, follow the workspace
[GPU policy](../../docs/ENTRYPOINT.md#gpu-작업-원칙) and verify that the selected
GPU 2/3 MIG UUID is free. Build the local
extension for the installed CUDA toolkit and then use the allocated device:

```bash
cd /home/irsl/ws/dlr/HGC-Net/pointnet2/pointnet2
CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=2 \
  uv run --project /home/irsl/ws/dlr/HGC-Net python setup.py build_ext --inplace
cd /home/irsl/ws/dlr/HGC-Net
CUDA_VISIBLE_DEVICES=0 uv run python -m unittest tests/test_pointnet2_cuda.py -v
```

The resulting dedicated environment is Python 3.12 with Torch `2.9.1+cu128`.
The build uses `/usr/local/cuda-12.8` (nvcc `12.8.93`) and explicitly emits
Blackwell `sm_120` code.  It does not touch `scdm/.venv`.

## Production-training contract

`tools/train_justin_hgc.py` preserves the upstream architecture, optimizer,
schedule, and seed: 80 total epochs, Adam with `(0.9, 0.999)` betas and
`1e-8` epsilon, learning rate `1e-4`, and seed 0. The production loader uses
12 workers to keep the actual 32-scene BatchNorm batch supplied from canonical
depth data. The upstream function defined a decay helper but never called it, so the Justin arm
uses a constant learning rate as the actual upstream behavior.

The Issue #59 production batch size is 32, intentionally replacing the
upstream batch size of 1 so BatchNorm receives a multi-scene batch.  A bounded
GPU3-1 capacity probe on actual canonical 25,000-point samples passed a full
forward, loss, backward, and Adam step at batch 32 with 14.52 GiB peak reserved
memory on a 23.63 GiB MIG slice; batch 40 was a capacity-only ceiling at 18.14
GiB and is not the production setting.

The canonical index's scene split is used without resampling or leakage:
9,505 train views and 495 validation views; evaluation uses a separate dataset.
A new run always
creates a UTC-timestamped directory under `runs/issue59/`; an existing directory
is an error.  Every epoch writes `last.pt` and an epoch checkpoint, validation
selects `best.pt`, and each checkpoint includes model, Adam state, counters,
metrics, and RNG states.  `provenance.json` pins command, commit, worktree
status, dataset root/counts, Torch/CUDA/device details, and the copied config.

After selecting a free GPU 2/3 MIG UUID under that policy, the smoke and resume
commands are:

```bash
cd /home/irsl/ws/dlr/HGC-Net
CUDA_VISIBLE_DEVICES=<allocated-device> uv run python tools/train_justin_hgc.py \
  --device cuda --run-name smoke --epochs 1 --limit-train-batches 2 --limit-val-batches 1
CUDA_VISIBLE_DEVICES=<allocated-device> uv run python tools/train_justin_hgc.py \
  --device cuda --resume runs/issue59/<run-id>/last.pt --epochs 2 \
  --limit-train-batches 1 --limit-val-batches 1
```

Production omits both batch limits and uses the same command with
`--run-name production`.  No existing dataset or run directory is overwritten.

## Minimal PointNet++ compatibility port

The upstream extension had been tied to Python 3.6/3.7 wheels, `THCState`, and
legacy `torch.cuda.*Tensor` allocation.  Only `pointnet2/pointnet2/` is changed
for this compatibility build: C++ wrappers use the current PyTorch CUDA stream
and `data_ptr`, Python allocations inherit the input device/dtype, package-local
imports are explicit, and the extension compiles as C++17.  The CUDA algorithms,
operator API, and the official HGC-Net model files remain unchanged.

`tests/test_pointnet2_cuda.py` first asserts the public gather operator's exact
forward values and input gradient, then runs `JustinPointNet2` end-to-end on
CUDA.  On the Blackwell `sm_120` fixture, both passed before this document was
updated.
