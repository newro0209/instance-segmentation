# Instance Segmentation Research Project

Hugging Face `transformers`를 기반으로 최신 Instance Segmentation 모델을 연구 및 개발하는 프로젝트입니다.

## 🚀 빠른 시작

### 1. 환경 설정

```powershell
# 의존성 설치 및 가상환경 동기화
uv sync
```

### 2. 주요 실행 명령어

`uv run`을 통해 패키징된 명령어를 즉시 실행할 수 있습니다.

- **환경 확인**: GPU 인식 및 CUDA 연산 가능 여부 점검

  ```powershell
  uv run verify
  ```

- **장치 확인**: 연결된 카메라 목록 및 상세 속성 확인

  ```powershell
  uv run devices
  ```

- **실시간 데모**: 카메라 스트림 분석 (1번 카메라 추천)

  ```powershell
  uv run stream --config configs/mask2former_config.yaml --camera_id 1
  ```

- **모델 학습**: 설정 파일을 통한 학습 시작

  ```powershell
  uv run train --config configs/mask2former_config.yaml
  ```

- **이미지 예측**: 단일 이미지 결과 확인

  ```powershell
  uv run predict --config configs/mask2former_config.yaml --image data/sample.jpg
  ```

설정 파일만 바꾸면 동일한 엔트리포인트로 다른 모델을 실행할 수 있습니다.

- **OneFormer 스트림 실행**

  ```powershell
  uv run stream --config configs/oneformer_config.yaml --camera_id 1
  ```

- **OneFormer 이미지 예측**

  ```powershell
  uv run predict --config configs/oneformer_config.yaml --image data/sample.jpg
  ```

- **OneFormer 학습 설정 확인**

  ```powershell
  uv run train --config configs/oneformer_config.yaml
  ```

## 🔍 디버깅 가이드 (GPU/CUDA)

### 증상

- `uv run verify` 결과에서 `PyTorch Version : ...+cpu`, `CUDA Available : False`가 출력됨
- `uv pip install`로 `+cu130`를 설치해도 다음 `uv run` 시 다시 `+cpu`로 되돌아감

### 원인

- `uv`는 실행 시 lock 파일 기준으로 환경을 다시 맞춥니다.
- 즉, lock에 `torch==...+cpu`가 기록되어 있으면 수동 설치(`uv pip install`)로 `+cu130`을 넣어도 재동기화 과정에서 CPU 빌드로 복원됩니다.

### 해결 절차 (확인된 순서)

```powershell
# 1) 현재 상태 확인
uv run verify

# 2) CUDA 인덱스를 사용해 lock 파일을 갱신 (핵심)
uv lock --upgrade --index pytorch-cu130

# 3) lock 기준으로 환경 동기화
uv sync --refresh

# 4) CUDA 인식 재검증
uv run verify
```

### 기대 결과

- `PyTorch Version : 2.11.0+cu130`
- `CUDA Available : True`
- `GPU Operation   : SUCCESS`

### 운영 팁

- 의존성 변경 후에는 `uv pip install` 단독 사용보다 `uv lock` + `uv sync`를 우선하세요.
- CUDA 빌드를 계속 고정하려면 lock 파일을 커밋해 팀 환경을 동일하게 유지하세요.

## 📂 프로젝트 구조

- `src/instance_segmentation/`: 핵심 소스 코드 (Trainer, Predictor, Analyzer 등)
- `configs/`: 모델별 YAML 설정 파일
- `data/` / `results/`: 데이터셋 및 실험 결과 저장소
