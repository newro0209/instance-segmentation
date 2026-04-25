# Mask2Former Segmentation Research Project

Hugging Face `transformers` 기반 Mask2Former 모델로 panoptic 및 instance segmentation을 연구 및 개발하는 프로젝트입니다.

## 🚀 빠른 시작

### 1. 환경 설정

```powershell
# 의존성 설치 및 가상환경 동기화
uv sync
```

### 2. 주요 실행 명령어

`uv run`을 통해 패키징된 명령어를 즉시 실행할 수 있습니다. `inference.task`는 `panoptic` 또는 `instance`를 사용할 수 있습니다.

- **환경 확인**: GPU 인식 및 CUDA 연산 가능 여부 점검

  ```powershell
  uv run verify-gpu
  ```

- **장치 확인**: 연결된 카메라 목록 및 상세 속성 확인

  ```powershell
  uv run list-cameras
  ```

- **실시간 데모**: 카메라 스트림 분석 (1번 카메라 추천)

  ```powershell
  uv run infer-stream --config configs/mask2former_config.yaml --camera_id 1
  ```

- **모델 학습**: 설정 파일을 통한 학습 시작

  ```powershell
  uv run train-mask2former --config configs/mask2former_config.yaml
  ```

  학습 모드는 YAML의 `training.mode` 또는 CLI `--mode`로 선택합니다.

  ```powershell
  uv run train-mask2former --config configs/mask2former_config.yaml --mode full_training
  uv run train-mask2former --config configs/mask2former_config.yaml --mode full_adjustment
  uv run train-mask2former --config configs/mask2former_config.yaml --mode fine_tuning
  ```

  - `full_training`: 체크포인트 설정만 가져오고 가중치는 무작위 초기화해 전체 학습
  - `full_adjustment`: 사전학습 가중치를 불러와 모든 파라미터 조정
  - `fine_tuning`: 사전학습 가중치를 불러오고 backbone은 고정한 채 decoder/head 중심 미세 조정

  학습 데이터는 `dataset.format`으로 `segmentation_folder`, `segmentation_manifest`, `coco_instance`를 사용할 수 있습니다. 실제 학습 전 구성만 확인하려면 `--dry_run`을 붙입니다.

- **이미지 예측**: 단일 이미지 segmentation 결과 확인

  ```powershell
  uv run infer-image --config configs/mask2former_config.yaml --image data/sample.jpg
  ```

- **Mask2Former Large 스트림 실행**

  ```powershell
  uv run infer-stream --config configs/mask2former_swin_large_config.yaml --camera_id 1
  ```

- **Mask2Former Large 이미지 예측**

  ```powershell
  uv run infer-image --config configs/mask2former_swin_large_config.yaml --image data/sample.jpg
  ```

- **Mask2Former Instance 이미지 예측**

  ```powershell
  uv run infer-image --config configs/mask2former_instance_config.yaml --image data/sample.jpg
  ```

- **Mask2Former Large 학습 설정 확인**

  ```powershell
  uv run train-mask2former --config configs/mask2former_swin_large_config.yaml
  ```

Mask2Former 설정 파일의 `inference.task` 값을 `panoptic` 또는 `instance`로 바꾸면 동일한 엔트리포인트로 후처리 방식을 전환할 수 있습니다. Instance 전용 체크포인트 예시는 `configs/mask2former_instance_config.yaml`에 있습니다.

## 🔍 디버깅 가이드 (GPU/CUDA)

### 증상

- `uv run verify-gpu` 결과에서 `PyTorch Version : ...+cpu`, `CUDA Available : False`가 출력됨
- `uv pip install`로 `+cu130`를 설치해도 다음 `uv run` 시 다시 `+cpu`로 되돌아감

### 원인

- `uv`는 실행 시 lock 파일 기준으로 환경을 다시 맞춥니다.
- 즉, lock에 `torch==...+cpu`가 기록되어 있으면 수동 설치(`uv pip install`)로 `+cu130`을 넣어도 재동기화 과정에서 CPU 빌드로 복원됩니다.

### 해결 절차 (확인된 순서)

```powershell
# 1) 현재 상태 확인
uv run verify-gpu

# 2) CUDA 인덱스를 사용해 lock 파일을 갱신 (핵심)
uv lock --upgrade --index pytorch-cu130

# 3) lock 기준으로 환경 동기화
uv sync --refresh

# 4) CUDA 인식 재검증
uv run verify-gpu
```

### 기대 결과

- `PyTorch Version : 2.11.0+cu130`
- `CUDA Available : True`
- `GPU Operation   : SUCCESS`

### 운영 팁

- 의존성 변경 후에는 `uv pip install` 단독 사용보다 `uv lock` + `uv sync`를 우선하세요.
- CUDA 빌드를 계속 고정하려면 lock 파일을 커밋해 팀 환경을 동일하게 유지하세요.

## 📂 프로젝트 구조

- `src/instance_segmentation/runtime/`: Mask2Former 로딩, 전처리, 추론, 후처리 런타임
- `src/instance_segmentation/inference/`: 단일 이미지 및 카메라 스트림 추론 엔트리포인트
- `src/instance_segmentation/training/`: 재현 가능한 학습 설정 및 Trainer 오케스트레이션
- `src/instance_segmentation/camera/`: 스트림 실험을 위한 카메라 선택 및 장치 표시 순수 로직
- `src/instance_segmentation/diagnostics/`: GPU/CUDA 등 실험 환경 진단
- `src/instance_segmentation/infrastructure/`: OpenCV/PowerShell 등 외부 I/O 어댑터
- `configs/`: Mask2Former YAML 설정 파일
- `data/` / `results/`: 데이터셋 및 실험 결과 저장소
