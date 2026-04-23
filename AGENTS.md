# AI Coding Agent Guidelines for Instance Segmentation

This document helps AI coding agents understand the project structure, conventions, and how to work effectively with this codebase.

## Project Overview

Instance Segmentation research project leveraging Hugging Face `transformers` ecosystem. The project follows a configuration-driven, domain-centric architecture designed for experimentation and scalability.

**Key Technologies**: PyTorch, Hugging Face transformers, OpenCV, YAML config management, `uv` package manager.

## Build & Run Commands

### Environment Setup
```powershell
# Install dependencies and create virtual environment
uv sync
```

### Core Commands (via `pyproject.toml` scripts)
- **Verify GPU/CUDA**: `uv run verify` - Check GPU recognition and CUDA capability
- **List Cameras**: `uv run devices` - Show connected cameras and properties
- **Stream Analysis**: `uv run stream --config configs/mask2former_config.yaml --camera_id 1`
- **Model Training**: `uv run train --config configs/mask2former_config.yaml`
- **Single Image Prediction**: `uv run predict --config configs/mask2former_config.yaml --image data/sample.jpg`

### Testing
```powershell
# TODO: tests/ directory needs unit test implementation
uv run pytest
```

## Project Architecture

```
src/instance_segmentation/
├── segmentation_trainer.py      # Training orchestration using HF Trainer API
├── image_predictor.py           # Single image inference
├── stream_analyzer.py           # Real-time camera stream processing
├── gpu_verifier.py              # GPU/CUDA diagnostic utility
├── device_explorer.py           # Camera and device enumeration
├── domain/                      # Pure domain logic (functional core)
│   ├── camera_selection.py      # Camera mode selection and crop logic (pure)
│   └── device_catalog.py        # Device summary row/table generation (pure)
├── infrastructure/              # Side-effect adapters (I/O boundary)
│   └── camera_probe.py          # OpenCV/PowerShell camera probing
├── models/                      # Model wrappers and specific architectures
├── datasets/                    # Data loading, preprocessing, augmentation
└── utils/
    ├── config_loader.py         # YAML configuration path resolution
    ├── visualization.py         # Shared rendering pipeline (CORE)
    └── __init__.py

configs/                          # YAML-driven experiment configuration
├── mask2former_config.yaml
├── maskrcnn_config.yaml
└── oneformer_config.yaml

results/                          # Checkpoints, logs, evaluation outputs
data/                            # Datasets and sample images
tests/                           # Unit tests (under development)
notebooks/                       # Interactive exploration
```

## Critical Conventions

### 1. Configuration-Driven Design

**All experiment parameters live in `configs/*.yaml` files. Never hardcode hyperparameters.**

Example structure (see `configs/mask2former_config.yaml`):
```yaml
model:
    name: "facebook/mask2former-swin-tiny-coco-instance"
  type: "mask2former"

training:
  learning_rate: 0.00005
  num_epochs: 50
  weight_decay: 0.01

dataset:
  batch_size: 4
  num_workers: 4

augmentation:
  resize: [512, 512]
  horizontal_flip: 0.5
```

Entry points load config via `resolve_config_path(args.config)` (see [config_loader.py](src/instance_segmentation/utils/config_loader.py)).

### 2. Naming Conventions (Domain-Centric)

**Never use generic abstractions.** Use domain terminology that reveals intent.

#### Functions & Variables (snake_case)
✅ **Good**: `normalize_instance_results`, `image_processor`, `segmentation_array`
❌ **Bad**: `process_data`, `temp`, `utility_function`

#### Classes (PascalCase)
✅ **Good**: `SegmentationTrainer`, `ImagePredictor`, `StreamAnalyzer`
❌ **Bad**: `Manager`, `Processor`, `Helper`

#### Boolean-like States (no `is_`, `has_`, `can_`, `should_` prefixes)
✅ **Good**: `training_active`, `model_initialized`, `cuda_available`
❌ **Bad**: `is_training`, `has_model`, `can_use_cuda`

#### Configuration Keys (snake_case in YAML)
```yaml
learning_rate: 0.00005
warmup_steps: 500
num_epochs: 50
output_dir: "./results"
```

### 3. Visualization Pipeline (Centralized)

**Use the shared visualization utilities in [utils/visualization.py](src/instance_segmentation/utils/visualization.py).**

Both `image_predictor.py` and `stream_analyzer.py` follow the same pattern:
```python
from instance_segmentation.utils.visualization import (
    normalize_instance_results,      # Convert raw model output to standard format
    draw_instance_overlay,           # Render masks on image
)

# 1. Get raw model output
raw_output = model(**inputs)

# 2. Normalize to standard format (list of dicts with score, label, mask)
instance_list = normalize_instance_results(raw_output)

# 3. Render on image
visualization = draw_instance_overlay(image, instance_list)
```

This ensures deterministic label colors and consistent rendering across all entry points.

### 4. Hugging Face Trainer API

Training orchestration prefers the HF `Trainer` class for consistency with the ecosystem. See [segmentation_trainer.py](src/instance_segmentation/segmentation_trainer.py) for example.

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir=config['training']['output_dir'],
    num_train_epochs=config['training']['num_epochs'],
    # ... (read from config)
)
model = AutoModelForInstanceSegmentation.from_pretrained(model_name)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

### 5. Comments & Documentation (Korean)

- **What**: Code should self-document via clear naming
- **Why**: Comments explain reasoning, design decisions, and constraints
- **Language**: Use Korean for all comments and documentation

✅ **Good**:
```python
# 외부 인증 시스템은 짧은 시간 내 중복 호출을 제한하므로
# 사용자별 직렬 처리로 요청합니다.
for each user in users:
    ...
```

❌ **Bad**:
```python
# Loop through users
for each user in users:
    ...
```

### 6. CUDA/GPU Handling

GPU detection is critical. Reference [gpu_verifier.py](src/instance-segmentation/gpu_verifier.py) for proper diagnostics.

**Common issue**: After dependency changes, use proper uv synchronization:
```powershell
uv lock --upgrade --index pytorch-cu130
uv sync --refresh
uv run verify
```

Expected success output:
- `PyTorch Version: 2.11.0+cu130`
- `CUDA Available: True`
- `GPU Operation: SUCCESS`

## Code Design Principles

### Functional & Reactive First (Default)

- Prefer functional programming by default: pure functions, immutability, declarative transformations.
- Prefer reactive programming for event-driven and async flows.
- Keep stream pipelines short, readable, and testable; avoid nested subscriptions.
- Push side effects (I/O, GPU calls, filesystem, network) to boundary layers.

### Python 3.13-First Style

- Use modern Python 3.13 syntax and standard-library features first.
- Prefer concise expression-oriented patterns when readability is preserved.
- Optimize for both runtime efficiency and cognitive simplicity; avoid premature micro-optimizations.
- If a shorter implementation reduces clarity, keep the clearer version.

### Type Hinting Rule (Built-in Generics)

- Do not use legacy typing collection aliases like `List`, `Dict`, `Tuple`, `Set`, `Optional`.
- Use built-in generic types: `list[T]`, `dict[K, V]`, `tuple[T, ...]`, `set[T]`, `T | None`.
- Import from `typing` only when there is no built-in alternative (for example `Protocol`, `TypedDict`, `TypeAlias`, `Literal`).

### Cohesion & Coupling Rule

- Maintain high cohesion in each module/function.
- Keep coupling intentionally controlled and minimal across module boundaries.
- Prefer explicit contracts at boundaries so modules can evolve independently.

### Single Responsibility

Each module has one clear purpose:
- `segmentation_trainer.py`: Training orchestration only
- `image_predictor.py`: Single image inference only
- `stream_analyzer.py`: Real-time stream processing only
- `visualization.py`: Rendering logic only

When adding features, ask: "Does this belong in this module's responsibility?"

### Pure Functions Over Mutations

Prefer data transformations that return new values:

```python
# ✅ Good: Returns new processed list
def normalize_instance_results(raw_results):
    processed_results = []
    # ... build and return
    return processed_results

# ❌ Bad: Mutates input or global state
def process_results(results):
    results['modified'] = True  # mutation
    global_cache.update(results)  # side effect
```

### Minimal Dependencies

- Avoid premature abstraction
- Add new dependencies only when they directly solve a problem
- Centralize imports at module level (not hidden inside functions)

### Testability First

Structure code so unit tests can inject mocks:
- Load configuration via explicit parameters
- Use dependency injection for external services
- Avoid hidden global state

Example:
```python
# ✅ Good: Testable
def process_image(image, model, config):
    # ... deterministic, injectable
    return results

# ❌ Bad: Hard to test
def process_image(image_path):
    model = global_model_cache.get()  # hidden dependency
    config = load_config()  # hidden file I/O
    # ...
```

## Development Workflow

### When Adding Features

1. **Define in one sentence**: What is the user-facing behavior?
2. **Identify boundaries**: Input, output, side effects
3. **Design data flow**: Write pure domain logic first
4. **Choose home**: Which module owns this responsibility?
5. **Name precisely**: Names should explain intent
6. **Test immediately**: Write test alongside code
7. **Simplify**: Can this be shorter without losing clarity?

### When Refactoring

- Extract functions only after the pattern repeats 3+ times
- Ensure extracted function has a clear, domain-centric name
- Do not create generic `utils/`, `common/`, or `helpers/` modules
- If unsure where code belongs, it's not ready to extract

### Code Review Questions

- [ ] Can I understand the intent from names alone?
- [ ] Does each file/function have exactly one responsibility?
- [ ] Are there any premature abstractions?
- [ ] Is configuration in YAML, not hardcoded?
- [ ] Can this be tested without mocking filesystem/network?
- [ ] Are there unnecessary `is_`, `has_`, `can_` prefixes?
- [ ] Are side effects at the boundaries (I/O, GPU operations)?

## Key Files & Patterns

| File | Purpose | Key Pattern |
|------|---------|-------------|
| [main.py](main.py) | Placeholder entry point | Keep minimal |
| [pyproject.toml](pyproject.toml) | Package config & entry points | Scripts map to module:function |
| [configs/*.yaml](configs/) | Experiment parameters | Single source of truth for hyperparams |
| [src/.../segmentation_trainer.py](src/instance_segmentation/segmentation_trainer.py) | Training | Load config → build model → use HF Trainer |
| [src/.../image_predictor.py](src/instance_segmentation/image_predictor.py) | Inference (static) | normalize_instance_results → draw_instance_overlay |
| [src/.../stream_analyzer.py](src/instance_segmentation/stream_analyzer.py) | Inference (streaming) | Same visualization pipeline as image_predictor |
| [src/.../domain/camera_selection.py](src/instance_segmentation/domain/camera_selection.py) | Functional camera logic | Size inference, mode ranking, center crop (pure) |
| [src/.../domain/device_catalog.py](src/instance_segmentation/domain/device_catalog.py) | Functional device summary | Device typing + deterministic table rendering (pure) |
| [src/.../infrastructure/camera_probe.py](src/instance_segmentation/infrastructure/camera_probe.py) | I/O boundary | OpenCV/PowerShell probing isolated from domain logic |
| [src/.../utils/visualization.py](src/instance_segmentation/utils/visualization.py) | Shared rendering | Deterministic colors, normalize + draw pattern |
| [src/.../utils/config_loader.py](src/instance_segmentation/utils/config_loader.py) | Config resolution | Handles relative & absolute paths |

## Prohibited Patterns

- ❌ Boolean prefixes: `is_active`, `has_error`, `can_retry` (use `active`, `error_present`, `retry_allowed`)
- ❌ Generic modules: `utils/helpers.py`, `common/`, `misc/` (be domain-specific)
- ❌ Abbreviations: `cfg` → `configuration`, `msg` → `message`, `repo` → `repository`
- ❌ Ignored exceptions: `except Exception: pass`
- ❌ Global state: Avoid shared mutable objects across modules
- ❌ Hardcoded values: Always use config files
- ❌ Nested subscriptions: In reactive flows, avoid deep nesting
- ❌ Manager/Helper classes: These indicate unclear responsibility

## When to Ask for Clarification

Before making changes, escalate if:
1. Unclear which config file applies to the change
2. Feature seems to span multiple responsibilities (trainer + predictor)
3. Need to introduce a new entry point or module
4. Considering a new external dependency
5. Unsure whether to add to existing utility or create a new one

## Resources

- See [GEMINI.md](GEMINI.md) for project overview and local environment details
- See [README.md](README.md) for troubleshooting GPU/CUDA issues
- Check [notebooks/](notebooks/) for experimental exploration patterns
- Configuration examples in [configs/](configs/)
