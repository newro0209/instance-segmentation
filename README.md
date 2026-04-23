# Instance Segmentation Project

Hugging Face `transformers`를 기반으로 한 최신 Instance Segmentation 모델(예: Mask2Former, OneFormer, SegFormer 등) 연구 및 개발 프로젝트입니다.

## 🛠️ 개발 환경 설정

이 프로젝트는 `uv`를 사용하여 패키지를 관리합니다.

1. **uv 설치** (아직 설치하지 않은 경우):
   ```bash
   pip install uv
   ```

2. **가상 환경 구성 및 패키지 설치**:
   ```bash
   uv sync
   ```

## 📂 프로젝트 구조

- `data/`: 데이터셋 (Raw, Processed)
- `configs/`: 학습 및 모델 설정 파일 (YAML)
- `src/`: 소스 코드
  - `models/`: 모델 정의 및 래퍼
  - `datasets/`: 데이터 로딩 및 전처리 로직
  - `utils/`: 유틸리티 (로깅, 메트릭, 시각화 등)
- `notebooks/`: 실험 및 분석용 Jupyter Notebook
- `tests/`: 단위 테스트
- `results/`: 학습 결과 (Checkpoint, Log, 시각화)

## 🚀 주요 실행 방법

### 학습 (Training)
```bash
uv run train --config configs/mask2former_config.yaml
```

### 이미지 예측 (Prediction)
```bash
uv run predict --config configs/mask2former_config.yaml --image data/sample.jpg
```

### 카메라 스트림 분석 (Stream Analysis)
```bash
uv run stream --config configs/oneformer_config.yaml --camera_id 0
```

## 📝 라이브러리 구성
- **Hugging Face**: `transformers`, `datasets`, `evaluate`, `accelerate`
- **Deep Learning**: `torch`, `torchvision`
- **Augmentation**: `albumentations`
- **Visualization**: `matplotlib`, `wandb`, `tensorboard`
