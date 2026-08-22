# HGC-Net 모델 및 데이터 개요

이 문서는 HGC-Net 관련 작업을 이어받는 agent가 모델 계보, 학습 데이터, 핵심 결정과 현재 구현 범위를 빠르게 파악하도록 돕는 큰그림 문서입니다. 세부 구현은 연결된 코드와 contract를 기준으로 확인합니다.

## 두 HGC-Net 계보

### Upstream-faithful PointNet++

현재 `/home/irsl/ws/dlr/HGC-Net`에서 관리하는 학습 대상입니다. 공식 HGC-Net의 PointNet++ 기반 point-wise prediction 구조를 유지하면서 출력 경계를 Justin-right hand에 맞췄습니다.

- 입력은 한 장의 depth observation에서 만든 25,000-point cloud입니다.
- 각 point와 `finger2`, `finger3`, `finger4` template에 대해 graspability, palm pose, `q_contact`, `q_squeeze`를 예측합니다.
- `q_open`은 모델이 예측하지 않고 KMK grasp template에서 가져옵니다.
- 공식 upstream은 `yimingli1998/hgc_net`의 commit `abca11a6de36b51c738166771ac7c794a2f6cb21`에 고정합니다.
- upstream 저장소에는 `LICENSE` 또는 `COPYING`이 없으므로 재배포와 재사용의 라이선스 문제는 해결되지 않았습니다.

모델 구조는 [`justin_hgc/model.py`](../justin_hgc/model.py), 후보 복원과 후처리는 [`justin_hgc/runtime.py`](../justin_hgc/runtime.py)와 [`justin_hgc/postprocess.py`](../justin_hgc/postprocess.py)를 확인합니다.

### Paper-modified 3D-FPN

`/home/irsl/ws/scdm_final`에서 관리하는 논문 실험용 별도 계보입니다. 이 모델은 GT full-occupancy voxel을 입력으로 사용하는 3D-FPN 기반 구현입니다.

PointNet++ arm과 paper-modified 3D-FPN arm은 구조와 입력 표현이 다릅니다. 따라서 코드, checkpoint와 실험 결과를 서로 재사용하지 않으며, 비교 결과에는 입력 정보량의 차이를 명시합니다.

## Upstream-faithful 추론의 큰그림

한 장의 depth image를 view-aligned point cloud로 변환하고, PointNet++가 point별 grasp 후보를 예측합니다. 예측 결과를 Justin palm pose와 12-DoF hand configuration으로 복원한 뒤, top-side filter와 NMS를 적용하여 최종 후보를 만듭니다.

현재 model과 decoder 구성 요소는 구현되어 있지만, checkpoint loading부터 실제 runtime transport까지 연결하는 완전한 end-to-end inference entrypoint는 아직 없습니다.

## 학습 데이터와 supervision

Canonical source는 다음 unified-v4 데이터셋입니다.

```text
/home/irsl/datasets/dlr/compiled/scdm_justin_right_vgn_train_10000_reconstruction_view_aligned_v4
```

- split은 train 8,995 / validation 510 / test 495입니다.
- 입력은 GT voxel이나 reconstructed volume이 아니라 depth-derived point observation입니다.
- 각 sample은 하나의 scene/view이며, deterministic sampling으로 정확히 25,000 points를 만듭니다.
- Justin-right label은 `finger2`, `finger3`, `finger4`와 12-DoF `q_contact`, `q_squeeze`를 사용합니다.

Positive grasp 데이터는 unified-v4 원본에 들어 있습니다. Loader는 원본의 모든 canonical `approach_point`를 대상으로 positive point label을 만듭니다. `thumb_visible_mask`는 사용하지 않으며, sampled point가 `approach_point`의 5mm 이내에 있을 때만 해당 grasp가 point-wise positive label로 투영됩니다.

Canonical negative grasp에는 point-wise supervision에 필요한 surface anchor가 없습니다. 따라서 임의의 anchor를 만들지 않고, 모든 positive 영역을 제외한 sampled point 중 deterministic 10%를 negative로 사용합니다. `derived/hgc_derived`에는 이 negative point sidecar만 저장하며 positive grasp 데이터는 복사하지 않습니다.

```text
/home/irsl/datasets/dlr/derived/hgc_derived
```

Loader는 sidecar의 source provenance, sample identity, deterministic sampling index, negative 좌표와 positive-negative 비중복을 검증합니다. 자세한 변환 규칙은 [`justin_hgc/data.py`](../justin_hgc/data.py), [`justin_hgc/labels.py`](../justin_hgc/labels.py)와 [`contract/issue59_upstream_canonical_v4.json`](../contract/issue59_upstream_canonical_v4.json)을 확인합니다.

## 학습 계약

- unified-v4에서 처음부터 학습하며 v3 checkpoint를 resume하지 않습니다.
- v3 derived data를 재사용하지 않습니다.
- 기본 설정은 80 epochs, batch size 32, Adam, learning rate `1e-4`, seed 0입니다.
- `best.pt`는 validation total loss로 선택합니다.
- W&B에는 `train/total_loss`, `val/total_loss`, `val/graspability_f1`, `learning_rate`만 기록합니다. 세부 loss는 로컬 `metrics.jsonl`에 보존합니다.
- PointNet++ CUDA test와 실제 v4 train/validation batch smoke는 통과했습니다.
- Production-trained v4 checkpoint는 아직 없습니다.

학습 설정과 checkpoint 형식은 [`config/issue59_justin_adapter.yaml`](../config/issue59_justin_adapter.yaml)과 [`tools/train_justin_hgc.py`](../tools/train_justin_hgc.py)를 기준으로 합니다.

## 다음 작업에서 지켜야 할 결정

- Upstream-faithful PointNet++과 paper-modified 3D-FPN을 별도 arm으로 유지합니다.
- 두 arm이 같은 unified-v4 split을 사용하더라도 입력 표현이 다르다는 사실을 보존합니다.
- `thumb_visible_mask`를 point-wise supervision에 추가하지 않습니다.
- Canonical negative grasp에 없는 surface anchor를 임의로 만들지 않습니다.
- Unified-v4 원본은 immutable하게 유지하고 model-specific data는 derived root에 둡니다.
- 학습 완료, 평가 완료와 runtime integration 완료를 서로 다른 상태로 보고합니다.
