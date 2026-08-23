# HGC-Net 모델 및 데이터 개요

이 문서는 HGC-Net 관련 작업을 이어받는 agent가 모델 계보, 학습 데이터, 핵심 결정과 현재 구현 범위를 빠르게 파악하도록 돕는 큰그림 문서입니다. 세부 구현은 연결된 코드와 contract를 기준으로 확인합니다.

## 두 HGC-Net 계보

### Upstream-faithful PointNet++

현재 `/home/irsl/ws/dlr/HGC-Net`에서 관리하는 학습 대상입니다. 공식 HGC-Net의 PointNet++ 기반 point-wise prediction 구조를 유지하면서 출력 경계를 Justin-right hand에 맞췄습니다.

- 입력은 한 장의 depth observation에서 만든 25,000-point cloud입니다.
- 각 point와 `finger2`, `finger3`, `finger4` template에 대해 graspability, palm pose와 `q_contact`를 예측합니다.
- `q_open`과 `q_squeeze`는 모델 출력이 아닙니다. 평가 client의 `JustinContactWaypointMapper`가 예측된 `q_contact`에서 Jacobian/FK를 사용하여 두 실행 waypoint를 만듭니다.
- 공식 upstream은 `yimingli1998/hgc_net`의 commit `abca11a6de36b51c738166771ac7c794a2f6cb21`에 고정합니다.
- upstream 저장소에는 `LICENSE` 또는 `COPYING`이 없으므로 재배포와 재사용의 라이선스 문제는 해결되지 않았습니다.

모델 구조는 [`justin_hgc/model.py`](../justin_hgc/model.py), 후보 복원과 후처리는 [`justin_hgc/runtime.py`](../justin_hgc/runtime.py)와 [`justin_hgc/postprocess.py`](../justin_hgc/postprocess.py)를 확인합니다.

### Paper-modified 3D-FPN

논문 실험용 paper-modified arm은 `/home/irsl/ws/dlr/HGC-Net` 안에 새로 구현하여 관리합니다. 입력은 view-aligned-v5의 GT full-occupancy voxel이며, 모델은 3D-FPN 기반 구조를 사용합니다.

`/home/irsl/ws/scdm_final`의 `scdm.baselines.hgc.HGC`와 Issue #59 dataset/trainer는 3D-FPN encoder와 voxel feature 처리의 참고 자료입니다. 현재 확인한 source reference는 commit `a344dcf076bc0f080b96d3909676cf734d4fac32`입니다. 새 구현은 해당 패키지를 import하거나 checkpoint를 재사용하지 않습니다. Paper-modified arm은 dense quality, 864-bin orientation, residual과 `q_contact`를 위한 독립 head를 사용합니다.

현재 HGC-Net에는 paper-modified dataset, 3D-FPN encoder, model wrapper, config, contract와 test가 구현되어 있습니다. 기존 `tools/train_justin_hgc.py`가 두 arm을 함께 지원합니다. CPU 검증과 production-channel CUDA smoke를 모두 통과하여 학습 준비가 완료됐지만 production training은 아직 시작하지 않았습니다.

PointNet++ arm과 paper-modified 3D-FPN arm은 입력 표현, encoder, head와 loss가 다릅니다. 두 arm은 trainer, checkpoint 형식과 `palm pose + quality + q_contact` runtime 출력 계약만 공유합니다. Checkpoint와 실험 결과는 서로 재사용하지 않으며 비교 결과에는 입력 정보량의 차이를 명시합니다.

## Upstream-faithful 추론의 큰그림

한 장의 depth image를 view-aligned point cloud로 변환하고, PointNet++가 point별 grasp 후보를 예측합니다. 예측 결과를 Justin palm pose와 12-DoF hand configuration으로 복원한 뒤, top-side filter와 NMS를 적용하여 최종 후보를 만듭니다.

Checkpoint를 strict하게 불러오는 server와 NPZ/ZMQ transport가 구현되어 있습니다. Paper-modified server는 object occupancy와 그 1-voxel 6-neighbor surface에서만 후보를 고릅니다. Server는 top-side filter와 NMS 이후 부족한 후보를 quality 순서를 보존하여 보충하고, 항상 quality 내림차순의 후보 100개를 반환합니다. 실행용 `q_open`과 `q_squeeze`는 server가 아니라 평가 client에서 생성합니다.

## Upstream-faithful 학습 데이터와 supervision

Canonical source는 다음 view-aligned-v5 데이터셋입니다.

```text
/home/irsl/datasets/dlr/compiled/scdm_justin_right_vgn_train_10000_reconstruction_view_aligned_v5
```

- split은 train 9,505 / validation 495 / test 0입니다. 평가는 별도의 evaluation dataset에서 수행합니다.
- 입력은 GT voxel이나 reconstructed volume이 아니라 depth-derived point observation입니다.
- 각 sample은 하나의 scene/view이며, deterministic sampling으로 정확히 25,000 points를 만듭니다.
- Justin-right 학습 label은 `finger2`, `finger3`, `finger4`와 12-DoF `q_contact`를 사용합니다.

Positive grasp 데이터는 view-aligned-v5 원본에 들어 있습니다. Loader는 palm local +z와 approach 방향의 cosine이 `0.9999` 이상인 grasp만 사용하여 10도 fallback-ray grasp를 제외합니다. `thumb_visible_mask`는 사용하지 않으며, sampled point가 `approach_point`의 5mm 이내에 있을 때만 해당 grasp가 point-wise positive label로 투영됩니다.

Canonical negative grasp에는 point-wise supervision에 필요한 surface anchor가 없습니다. 따라서 임의의 anchor를 만들지 않고, 모든 positive 영역을 제외한 sampled point 중 deterministic 10%를 negative로 사용합니다. 데이터셋 내부의 `derived/hgc`에는 이 negative point sidecar만 저장하며 positive grasp 데이터는 복사하지 않습니다.

```text
/home/irsl/datasets/dlr/compiled/scdm_justin_right_vgn_train_10000_reconstruction_view_aligned_v5/derived/hgc
```

Loader는 sidecar의 source provenance, sample identity, deterministic sampling index, negative 좌표와 positive-negative 비중복을 검증합니다. 현재 sidecar 10,000개는 모두 유효하며 총 negative point 수는 24,008,333개입니다. 자세한 변환 규칙은 [`justin_hgc/data.py`](../justin_hgc/data.py), [`justin_hgc/labels.py`](../justin_hgc/labels.py)와 [`contract/issue59_upstream_canonical_v5.json`](../contract/issue59_upstream_canonical_v5.json)을 확인합니다.

## Upstream-faithful 학습 계약

- view-aligned-v5에서 처음부터 학습하며 이전 데이터셋의 checkpoint를 resume하지 않습니다.
- 이전 데이터셋의 derived data를 재사용하지 않습니다.
- 기본 설정은 80 epochs, batch size 32, Adam, learning rate `1e-4`, seed 0입니다.
- `best.pt`는 validation total loss로 선택합니다.
- W&B에는 `train/total_loss`, `val/total_loss`, `val/graspability_f1`, `learning_rate`만 기록합니다. 세부 loss는 로컬 `metrics.jsonl`에 보존합니다.
- PointNet++ CUDA test와 실제 데이터 train/validation batch smoke는 통과했습니다.
- Production-trained v5 checkpoint는 아직 없습니다.

학습 설정과 checkpoint 형식은 [`config/issue59_justin_adapter.yaml`](../config/issue59_justin_adapter.yaml)과 [`tools/train_justin_hgc.py`](../tools/train_justin_hgc.py)를 기준으로 합니다.

## Paper-modified 구현과 검증 상태

Paper-modified arm은 view-aligned-v5의 GT full-occupancy voxel·ground grid와 3D-FPN을 사용합니다. Dense quality volume에서 후보 위치를 고르고, 선택된 위치에서 orientation, residual과 `q_contact`를 예측합니다. Point-wise upstream arm과 supervision 및 head가 독립적이며, trainer와 runtime 출력 형식만 공유합니다.

구현된 코드 구조는 다음과 같습니다.

```text
paper_modified_hgc/
  data.py          view-aligned-v5 voxel과 supervision 위치
  encoder.py       3D-FPN과 voxel feature 선택
  model.py         encoder와 독립 dense-quality/pose/contact head의 결합
config/
  issue59_justin_adapter.yaml
  issue59_paper_modified.yaml
contract/
  issue59_paper_modified_canonical_v5.json
tools/
  train_justin_hgc.py       두 arm이 공유하는 trainer
tests/
  test_paper_modified_hgc.py
```

Paper-modified encoder는 supervised 위치별 local 128D feature와 64³ dense feature grid를 제공합니다. 독립 head가 dense quality와 선택 위치의 pose·contact를 예측합니다. 두 dataset adapter는 서로 다른 원본 표현을 읽지만 공통 trainer가 처리할 수 있는 arm별 batch를 반환합니다.

Trainer는 기존 [`tools/train_justin_hgc.py`](../tools/train_justin_hgc.py) 하나만 사용합니다. Config의 `arm` 값에 따라 dataset과 encoder를 factory에서 선택하고, epoch loop, optimizer 생성, metric, W&B, checkpoint와 strict resume는 같은 코드를 실행합니다. Trainer는 point cloud나 voxel tensor의 내부 구조를 직접 참조하지 않고 `model.forward_batch(batch)`와 공통 loss interface만 호출합니다.

Paper-modified dataset은 한 view에서 inside-grid positive grasp 100개를 deterministic하게 선택합니다. 또한 GT object occupancy에서 negative voxel 100개를 deterministic하게 선택하며, 해당 view의 모든 canonical positive approach voxel을 negative 후보에서 제외합니다. Negative row는 세 template 모두 graspability class 0으로 감독하고, pose와 joint loss에서는 제외합니다.

검증 상태는 다음과 같습니다.

1. `/home/irsl/ws/scdm_final@a344dcf076bc0f080b96d3909676cf734d4fac32`의 3D-FPN 구조와 voxel feature 선택 규칙을 참고 대상으로 고정했습니다. 기존의 별도 head와 loss는 이식하지 않았습니다.
2. view-aligned-v5의 split `9,505/495/0`, 64³ DHW=ZYX voxel과 Justin-right 12-DoF 계약을 [`contract/issue59_paper_modified_canonical_v5.json`](../contract/issue59_paper_modified_canonical_v5.json)에 기록했습니다.
3. 전체 CPU suite 31개가 통과했습니다. Dataset 의미, deterministic positive·negative 선택, 전체 positive voxel과 negative voxel의 비중복, reference GroupNorm, config 의미 검증, 공통 trainer dispatch, finite loss/backward와 strict resume를 포함합니다.
4. 실제 sample에서 `(2,64,64,64)` 입력, 선택 positive와 negative의 비중복, tiny-FPN finite backward를 확인했습니다.
5. 실행 중이던 upstream-faithful production 학습은 공통 trainer 수정 후에도 계속 정상 진행되었습니다.
6. GPU3-2의 별도 24 GiB MIG에서 production channel의 CUDA train backward·AdamW step과 validation forward를 통과했습니다. Peak allocated memory는 약 1.05 GiB, peak reserved memory는 약 1.46 GiB였습니다.
7. 같은 GPU에서 공통 trainer CLI의 train/validation 각 1-batch smoke를 통과했습니다. `metrics.jsonl`, provenance와 `epoch-000.pt`·`last.pt`·`best.pt` 생성을 확인했고, `/tmp` smoke 산출물은 삭제했습니다.
8. Production training을 시작하기 직전에 GPU 사용대장과 실제 점유 상태를 다시 확인합니다.

Production training 설정은 80 epochs, batch size 1, positive 100개와 negative 100개, seed 0, AdamW, learning rate `1e-4`, weight decay `1e-6`입니다. `best.pt`는 validation total loss로 선택합니다. 모든 구현 gate는 통과했으며 production training 시작만 남았습니다.

## 다음 작업에서 지켜야 할 결정

- Upstream-faithful PointNet++과 paper-modified 3D-FPN을 별도 arm으로 유지합니다.
- 두 arm을 모두 `/home/irsl/ws/dlr/HGC-Net`에서 관리합니다. 입력 adapter, encoder, head와 loss는 분리하고, trainer, metric 이름, checkpoint 형식과 runtime 출력 계약을 공유합니다.
- `/home/irsl/ws/scdm_final`의 paper-modified 구현은 source reference로만 사용하며 새 구현의 runtime dependency로 만들지 않습니다.
- 두 arm이 같은 view-aligned-v5 split을 사용하더라도 입력 표현이 다르다는 사실을 보존합니다.
- `thumb_visible_mask`를 point-wise supervision에 추가하지 않습니다.
- Canonical negative grasp에 없는 surface anchor를 임의로 만들지 않습니다.
- View-aligned-v5 원본은 immutable하게 유지하고 model-specific data는 해당 데이터셋의 `derived/hgc`에 둡니다.
- 학습 완료, 평가 완료와 runtime integration 완료를 서로 다른 상태로 보고합니다.
