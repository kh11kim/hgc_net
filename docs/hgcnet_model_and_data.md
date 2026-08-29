# HGC-Net 모델 및 데이터 개요

이 문서는 HGC-Net의 두 실험 arm, 현재 학습 데이터, derived data와 checkpoint 계보를 설명합니다. 세부 구현은 연결된 코드와 contract를 기준으로 확인합니다.

## 현재 상태

- Paper floor-v7에서 두 arm을 fresh training하기 위한 데이터 검증을 완료했습니다.
- Upstream-faithful arm의 negative-point sidecar 9,909개를 생성하고 전체 구조를 검증했습니다.
- 두 arm의 floor-v7 train·validation loader smoke가 통과했습니다.
- 두 학습 config, 코드의 기본 root와 canonical contract를 floor-v7으로 전환했습니다.
- Metadata contract validator와 전체 CPU test 75개가 통과했습니다.
- 물리 GPU 2의 MIG device 2에서 upstream batch 32와 paper-modified batch 16의 train·validation 1-batch smoke가 통과했습니다. 두 smoke checkpoint는 epoch 0, global step 1이며 strict load가 성공했습니다.
- 기존 unified-v5 checkpoint와 평가 결과는 historical candidate로 보존합니다. Floor-v7 학습은 해당 checkpoint를 resume하지 않는 새 run으로 시작합니다.
- Floor-v7 production run은 아직 시작하지 않았습니다.

## 두 HGC-Net 계보

### Upstream-faithful PointNet++

Upstream-faithful arm은 공식 HGC-Net의 PointNet++ 기반 point-wise prediction 구조를 유지하면서 출력 경계를 Justin-right hand에 맞춥니다.

- 입력은 한 장의 depth observation에서 만든 25,000-point cloud입니다.
- 각 point와 `finger2`, `finger3`, `finger4` template에 대해 graspability, palm pose와 `q_contact`를 예측합니다.
- `q_open`과 `q_squeeze`는 모델 출력이 아닙니다. 평가 client의 `JustinContactWaypointMapper`가 예측한 `q_contact`에서 Jacobian/FK를 사용하여 실행 waypoint를 만듭니다.
- 공식 upstream은 `yimingli1998/hgc_net`의 commit `abca11a6de36b51c738166771ac7c794a2f6cb21`에 고정합니다.
- upstream 저장소에는 `LICENSE` 또는 `COPYING`이 없으므로 재배포와 재사용의 라이선스 문제는 해결되지 않았습니다.

모델 구조는 [`justin_hgc/model.py`](../justin_hgc/model.py), 후보 복원과 후처리는 [`justin_hgc/runtime.py`](../justin_hgc/runtime.py)와 [`justin_hgc/postprocess.py`](../justin_hgc/postprocess.py)를 확인합니다.

### Paper-modified 3D-FPN

Paper-modified arm은 `/home/irsl/ws/dlr/HGC-Net` 안에 별도 구현으로 관리합니다. `/home/irsl/ws/scdm_final@a344dcf076bc0f080b96d3909676cf734d4fac32`의 3D-FPN 구조와 voxel feature 처리 방식만 참고했으며, 해당 패키지의 class, checkpoint나 runtime을 재사용하지 않습니다.

- 입력은 GT object full occupancy와 ground로 구성된 `(2,64,64,64)` voxel grid입니다.
- 독립 head가 dense quality, 864-bin orientation, residual과 `q_contact`를 예측합니다.
- 한 view에서 positive grasp 100개와 negative voxel 100개를 deterministic하게 선택합니다.
- 해당 view의 모든 canonical positive approach voxel은 negative 후보에서 제외합니다.

Paper-modified 구현은 [`paper_modified_hgc/`](../paper_modified_hgc)에 있습니다. 두 arm은 trainer와 checkpoint 형식만 공유하며, 입력 adapter, encoder, head, loss와 checkpoint 결과는 서로 재사용하지 않습니다.

## Paper floor-v7 canonical data

두 arm의 다음 fresh training은 다음 compiled root를 사용합니다.

```text
/home/irsl/datasets/dlr/compiled/scdm_paper_floor_10000_v7
```

- format은 `scdm_unified` version 2입니다.
- grid는 64³, `DHW=ZYX`, edge length 0.5 m입니다.
- Justin-right label은 `finger2`, `finger3`, `finger4`와 12-DoF `q_contact`를 사용합니다.
- support surface와 finger collision geom 사이에 2 cm collision margin을 적용한 scene collision annotation을 사용합니다.
- eligible grasp target이 없는 scene 91개를 제외하여 scene·view·sample이 각각 9,909개입니다.
- split seed는 0이며 train 9,422개, validation 487개, test 0개입니다.
- `dataset.yaml` SHA-256은 `3fbdf400816b65bc5a52157b334b3b25d6fbb0028d6e8b85caed8c475e9d2f42`입니다.
- `index/samples.jsonl` SHA-256은 `14f8d62a6b5ca0c2e675ca84172d1fa4132302c9d12c7551bf9ceb884ee31114`입니다.
- `index/splits.json` SHA-256은 `0e8b70c0369113974f29dc48fe9f778c68861bed3ee14f4452b9910a3eafb3ba`입니다.

Source root, object dataset, OI, OG와 collision annotation의 기준은 workspace의 [`docs/OBJECT_GRASP_DATASETS.md`](../../docs/OBJECT_GRASP_DATASETS.md)에서 관리합니다.

## Upstream negative-point derived data

Canonical negative grasp에는 point-wise supervision에 필요한 surface anchor가 없습니다. Upstream loader는 임의의 anchor를 만들지 않고, 모든 positive approach point의 5 mm 영역을 제외한 sampled point 중 deterministic 10%를 negative로 사용합니다. `thumb_visible_mask`는 이 선택에 사용하지 않습니다.

Derived root는 다음과 같습니다.

```text
/home/irsl/datasets/dlr/compiled/scdm_paper_floor_10000_v7/derived/hgc
```

- Sidecar는 source `sample_id`와 1:1로 대응하는 9,909개 NPZ입니다.
- 전체 negative point 수는 23,920,573개이며 derived root의 크기는 약 949 MB입니다.
- 누락·추가·불량 NPZ와 잔여 임시파일은 없습니다.
- Manifest의 source dataset·samples·splits hash가 canonical source와 일치합니다.
- `manifest.json` SHA-256은 `fd6b3c936778892ba9565217653af7a73bb9aa43a326bbb15af62c2d37a4f32d`입니다.
- Loader는 manifest의 `completed_sample_count`를 canonical `index/samples.jsonl`의 실제 sample 수와 대조합니다.

Derived data에는 negative-point supervision만 저장합니다. Positive grasp data를 복사하거나 canonical source를 수정하지 않습니다. 생성 규칙은 [`tools/generate_hgc_negative_points.py`](../tools/generate_hgc_negative_points.py), 로딩 규칙은 [`justin_hgc/data.py`](../justin_hgc/data.py)를 확인합니다.

## 학습 계약과 준비 상태

두 arm은 [`tools/train_justin_hgc.py`](../tools/train_justin_hgc.py)를 공유합니다. Config의 `arm` 값에 따라 dataset과 model factory를 선택하며, epoch loop, checkpoint 형식, metric과 W&B 기록 형식은 공통으로 사용합니다.

### Upstream-faithful

- 80 epochs, batch size 32, workers 12를 사용합니다.
- Optimizer는 Adam, learning rate는 `1e-4`, seed는 0입니다.
- `best.pt`는 validation total loss로 선택합니다.
- Floor-v7 train 9,422개와 validation 487개에서 derived sidecar를 사용한 loader smoke가 통과했습니다.

### Paper-modified

- 80 epochs, batch size 16, workers 8을 사용합니다.
- Optimizer는 AdamW, learning rate는 `1e-4`, weight decay는 `1e-6`, seed는 0입니다.
- 한 sample은 positive 100개와 negative voxel 100개를 사용합니다.
- `best.pt`는 validation total loss로 선택합니다.
- Floor-v7 train 9,422개와 validation 487개에서 `(2,64,64,64)` 입력과 target 생성 smoke가 통과했습니다.

Floor-v7 전환과 production-batch smoke에서 다음 항목을 확인했습니다.

1. `config/issue59_justin_adapter.yaml`과 `config/issue59_paper_modified.yaml`의 root를 floor-v7으로 변경했습니다.
2. `justin_hgc/data.py`, `paper_modified_hgc/data.py`와 derived generator의 기본 root가 같은 canonical root를 가리킵니다.
3. [`issue59_upstream_canonical_v7.json`](../contract/issue59_upstream_canonical_v7.json)과 [`issue59_paper_modified_canonical_v7.json`](../contract/issue59_paper_modified_canonical_v7.json)에 floor-v7 manifest, split, hash와 arm별 학습 계약을 고정했습니다.
4. Metadata contract validator와 전체 CPU test 75개가 통과했습니다.
5. 물리 GPU 2의 MIG device 2(`MIG-6dc2ec27-ea8a-5fc6-a3b1-50dfc4d52550`)에서 두 arm의 production batch로 train·validation 각 1 batch를 실행했습니다. Loss와 metric은 finite였고 생성된 checkpoint가 strict load됐습니다.

Production을 시작할 때에는 두 arm을 서로 다른 fresh run으로 실행하고, 각 run의 config, code commit, dataset root, derived root, W&B URL과 checkpoint를 별도 provenance에 기록합니다.

## Checkpoint와 평가 계보

Model Registry의 `hgc-upstream-faithful-v5`와 `hgc-paper-modified-v5`는 unified-v5로 학습한 historical candidate입니다. 해당 entry의 dataset root, checkpoint, metric과 기존 평가 결과를 floor-v7 provenance로 변경하지 않습니다.

Floor-v7 run이 시작되면 두 arm을 v5와 구분되는 새 Registry entry로 등록합니다. `selected_checkpoint`는 학습이 끝나고 best checkpoint의 epoch, global step, validation metric과 SHA-256을 검증하기 전까지 `null`로 유지합니다. 학습 완료, runtime integration 완료와 Track 2 평가 완료는 서로 다른 상태로 보고합니다.

공통 평가 계약은 workspace의 [`docs/EVALUATION.md`](../../docs/EVALUATION.md)를 따릅니다. Upstream-faithful arm은 depth point 입력을 사용하고, paper-modified arm은 Reconstruction이 반환한 grid를 사용하므로 두 결과의 입력 정보량과 protocol을 분리해서 기록합니다.

## 유지할 결정

- Upstream-faithful PointNet++과 paper-modified 3D-FPN을 별도 arm으로 유지합니다.
- 두 arm은 같은 floor-v7 split을 사용하더라도 입력 표현, model과 loss를 공유하지 않습니다.
- `/home/irsl/ws/scdm_final`의 구현은 source reference로만 사용하고 runtime dependency로 만들지 않습니다.
- `thumb_visible_mask`를 upstream point-wise supervision에 추가하지 않습니다.
- Canonical negative grasp에 없는 surface anchor를 임의로 만들지 않습니다.
- Model-specific derived data는 canonical root의 `derived/hgc`에 두고 canonical source를 수정하지 않습니다.
- Dataset을 바꾼 학습은 이전 checkpoint를 resume하지 않고 fresh run으로 시작합니다.
- 기존 v5 checkpoint와 결과를 floor-v7 이름으로 바꾸지 않습니다.
