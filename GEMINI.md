# GEMINI.md - Instance Segmentation Project

## Project Overview
This project is dedicated to the research and development of modern Instance Segmentation models (e.g., Mask2Former, OneFormer, SegFormer) leveraging the Hugging Face `transformers` ecosystem. It is designed with a structured approach to facilitate experimentation, scalability, and collaboration.

### Main Technologies
- **Language:** Python 3.13+
- **Deep Learning:** PyTorch, Torchvision
- **Frameworks:** Hugging Face (`transformers`, `datasets`, `evaluate`, `accelerate`)
- **Package Management:** `uv`
- **Augmentation:** `albumentations`
- **Utilities:** `opencv-python`, `matplotlib`, `pycocotools`, `wandb`, `tensorboard`

### Architecture
- `configs/`: Centralized YAML configuration files for models and training hyper-parameters.
- `src/`: Core implementation.
  - `models/`: Model wrappers and specific architectures.
  - `datasets/`: Data loading, preprocessing, and augmentation pipelines.
  - `utils/`: Reusable utilities for logging, metrics, and visualization.
- `notebooks/`: Interactive exploration and prototyping.
- `results/`: Output directory for checkpoints, logs, and evaluation results.

## Building and Running

### Environment Setup
The project uses `uv` for lightning-fast dependency management and virtual environment handling.
```powershell
# Sync dependencies and create .venv
uv sync
```

### Training
Training is configuration-driven. Use the `train` command.
```powershell
uv run train --config configs/mask2former_config.yaml
```

### Prediction
Predict instances from a single image.
```powershell
uv run predict --config configs/mask2former_config.yaml --image data/sample.jpg
```

### Stream Analysis
Analyze live camera stream.
```powershell
uv run stream --config configs/oneformer_config.yaml --camera_id 0
```

### Testing
```powershell
# TODO: Add unit tests in tests/ directory
uv run pytest
```

## Development Conventions

### Configuration Management
- Always prefer modifying `configs/*.yaml` over hardcoding parameters in scripts.
- Each major experiment should have its own configuration file for reproducibility.

### Naming Conventions
- **Domain-Centric:** 변수 및 함수 이름은 `data`, `temp`, `manager`와 같은 추상적인 단어 대신 `image_batch`, `instance_mask`, `segmentation_trainer` 등 도메인 개념을 명확히 반영합니다.
- **Boolean Variables:** `is_`, `has_`, `can_`, `should_`와 같은 접두사를 **사용하지 않습니다.** 대신 도메인 용어 자체로 상태를 명확히 표현합니다 (예: `training_active`, `mask_present`, `validation_required` 등).
- **Style:** Python 표준(PEP 8)을 따라 함수와 변수는 `snake_case`, 클래스는 `PascalCase`를 사용합니다.
- **Configurations:** YAML 설정 파일의 키값은 일관되게 `snake_case`를 유지합니다.

### Coding Style
- Follow standard Python (PEP 8) conventions.
- Use Hugging Face `Trainer` API where possible to maintain consistency with the ecosystem.
- Modularize data preprocessing within `src/datasets`.

### Testing & Validation
- New features or model integrations should include basic unit tests in the `tests/` directory.
- Use `tensorboard` or `wandb` for tracking experiment progress.
