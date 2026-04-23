# Engineering Agent Guidelines Skill

This skill encodes domain-independent design principles for consistent code quality across the instance-segmentation project. Use these guidelines during design, implementation, code review, and refactoring.

## Design Principles (SOLID + Pragmatism)

### Functional/Reactive First Principle
All implementations should prioritize functional and reactive paradigms where appropriate.

**Default direction**:
- Prefer pure functions and immutable data transformations.
- Prefer reactive flows for event-driven and async pipelines.
- Keep side effects at boundaries (I/O, network, GPU, filesystem, clock).
- Avoid nested subscriptions and hidden mutable global state.

### Python 3.13 First Principle
Use modern Python 3.13 syntax and language features aggressively when they improve clarity and maintainability.

**Application**:
- Prefer concise, expression-oriented code when readability is preserved.
- Prefer modern typing syntax (built-in generics, union operator `|`).
- Prefer standard-library solutions before introducing dependencies.

### Cohesion/Coupling Principle
Maintain high cohesion and tightly focused responsibilities, while keeping cross-module coupling explicitly controlled.

**Application**:
- A module should expose a small, stable contract.
- Internal logic can be tightly cohesive; external dependencies should remain loosely coupled.
- If a change in one module forces broad edits elsewhere, coupling is too high.

### Single Responsibility Principle
Every module, class, function, and service should change for exactly ONE reason.

**Application**: Each entry point (trainer, predictor, stream_analyzer) owns a single concern. If you need to add logic, determine: "Which module's responsibility does this fall under?" If unclear, the design isn't ready.

**Check**: When reviewing code, ask: "What would require changing this module?" If the answer includes multiple domains (training AND rendering), it violates SRP.

### Open/Closed Principle
Enable extension without modifying existing code.

**Application**: Configuration files are the extension point. New experiments → new YAML config, not code changes. New models → add to existing trainer, don't fork it.

### Liskov Substitution Principle
Subtypes must maintain the contract of their base type.

**Application**: When inheriting from `Trainer` or creating model wrappers, ensure they preserve the original behavior contract.

### Interface Segregation Principle
Clients should not depend on unused capabilities.

**Application**: Keep function signatures focused. Pass only what's needed (e.g., `image_processor` and `model` separately, not bundled in a config object).

### Dependency Inversion Principle
Depend on abstractions, not concrete implementations.

**Application**: Import `AutoModelForInstanceSegmentation` from transformers (the abstraction), not a specific implementation class.

## Naming & Clarity

### Domain-Centric Names (Not Generic)

✅ **Good names reveal intent:**
- `normalize_instance_results` - Transform raw model output to standard format
- `image_batch` - A batch of images being processed
- `segmentation_trainer` - Trains segmentation models
- `cuda_available` - GPU is ready

❌ **Generic names hide intent:**
- `process_data` - What kind of data? What does processing do?
- `temp` - Temporary what?
- `result` - What type of result?
- `is_active` - Active in what context?

**Rule**: Names should be self-documenting. If you need a comment to explain what a variable is, the name is wrong.

### Boolean States (NO Prefixes)

Use domain terminology, NOT `is_`, `has_`, `can_`, `should_`.

✅ **Good:**
- `model_initialized` 
- `training_active`
- `cuda_available`
- `checkpoint_exists`

❌ **Bad:**
- `is_model_initialized`
- `is_training_active`
- `has_cuda`
- `can_checkpoint`

**Rationale**: The prefix adds no information and clutters reading. Domain terms are clearer.

### Abbreviations (Prohibited Except Standard Terms)

Standard abbreviations are universal and acceptable:
- `http`, `url`, `id`, `api`, `sql`, `json`, `csv`, `yaml`

Everything else must be spelled out:
- ❌ `cfg` → ✅ `configuration`
- ❌ `msg` → ✅ `message`
- ❌ `pwd` → ✅ `password`
- ❌ `ctx` → ✅ `context`
- ❌ `req` → ✅ `request`
- ❌ `repo` → ✅ `repository`

**Rationale**: Short names save nothing; full names prevent misreading.

### Function & Variable Naming (snake_case)

```python
# ✅ Good
def normalize_instance_results(raw_output):
    mask_array = _to_numpy_mask(raw_output)
    return processed_instances

segmentation_trainer = SegmentationTrainer(config)

# ❌ Bad
def normalizeInstanceResults(rawOutput):
    maskArray = toNumpyMask(rawOutput)
    return processedInstances

segmentationTrainer = segmentationTrainer(config)
```

### Class Naming (PascalCase)

```python
# ✅ Good
class ImagePredictor:
    pass

class SegmentationTrainer:
    pass

class StreamAnalyzer:
    pass

# ❌ Bad
class image_predictor:
    pass

class segmentation_trainer:
    pass
```

## Python 3.13 Typing Standard

### Use Built-in Generic Types

Do not use legacy `typing` collection aliases when a built-in generic exists.

✅ **Required style**:
- `list[str]`
- `dict[str, int]`
- `tuple[int, ...]`
- `set[str]`
- `str | None`

❌ **Disallowed style**:
- `List[str]`
- `Dict[str, int]`
- `Tuple[int, ...]`
- `Set[str]`
- `Optional[str]`

### Import Rule for `typing`

- Avoid importing `typing` only for container aliases.
- Import from `typing` only when there is no built-in equivalent, such as:
    - `Protocol`
    - `TypedDict`
    - `TypeAlias`
    - `Literal`
    - `Callable` (when function signatures need explicit typing)

## Structure & Layering

### Avoid Generic Modules

These module names are red flags—they indicate unclear responsibility:
- ❌ `utils/` (unless for truly shared utilities like `config_loader.py`)
- ❌ `common/`
- ❌ `helpers/`
- ❌ `managers/`
- ❌ `processors/`

Each module must have a specific domain purpose:
- ✅ `image_predictor.py` - Predicts instances on single images
- ✅ `stream_analyzer.py` - Analyzes streaming camera input
- ✅ `segmentation_trainer.py` - Trains segmentation models

### Layered Architecture (System-Agnostic)

Separate concerns into layers:

1. **Input/Output Layer**: User requests, CLI args, HTTP endpoints, display
2. **Application Layer**: Orchestration, configuration loading, flow control
3. **Domain Layer**: Core rules (e.g., how to normalize segmentation masks)
4. **Infrastructure Layer**: File I/O, model loading, GPU access, external services

**Application**: Put GPU operations and file loading at boundaries, not in domain logic. Make domain logic pure and testable.

### Module Size & Cohesion

- One file = one central responsibility
- One function = one behavior
- One class = one domain concept

**Check**: If you can describe a module in one sentence without "and" or "or", cohesion is good.

❌ **Bad**: "This module loads data and trains models and saves results"
✅ **Good**: "This module orchestrates model training using Hugging Face Trainer"

## Function Design

### Single Behavior Per Function

```python
# ✅ Good: Clear, single purpose
def normalize_instance_results(raw_output):
    """Convert raw model output to standard format."""
    processed = []
    for item in raw_output.get("segments_info", []):
        processed.append({
            "score": item.get("score", 1.0),
            "mask": _to_numpy_mask(raw_output["segmentation"]),
        })
    return processed

# ❌ Bad: Multiple concerns mixed
def process_and_save_and_display(model, image_path, config):
    # Load image
    # Run inference
    # Normalize output
    # Save to disk
    # Display result
    # Log metrics
    # This does too many things!
```

### Function Length

No absolute limit, but these signal it's too long:
- Multiple conditional branches (more than 3)
- Impossible to describe in one sentence
- Needs comments to explain flow
- Mixes data transformation with I/O
- Exception handling tangled with logic

**Solution**: Extract into focused functions.

### Parameter Count

Minimize parameters. Pass related values as objects:

```python
# ❌ Bad: Many parameters
def create_training_job(
    model_name,
    learning_rate,
    batch_size,
    num_epochs,
    warmup_steps,
    weight_decay,
    output_dir,
):
    pass

# ✅ Good: Structured parameters
def create_training_job(model_name, training_config):
    # training_config has learning_rate, batch_size, etc.
    pass
```

Avoid boolean flags:

```python
# ❌ Bad: What does the boolean mean?
def load_model(model_name, use_cache=True):
    pass

# ✅ Good: Intent is clear
def load_model(model_name, cache_behavior="use_local"):
    pass
```

## Functional Programming Principles

### Pure Functions First

A pure function:
- Returns same output for same input (no randomness unless seeded)
- Doesn't modify external state
- Is testable without mocks

```python
# ✅ Good: Pure
def calculate_mask_area(mask):
    return int(mask.sum())

# ❌ Bad: Side effects
global_cache = {}
def calculate_mask_area(mask):
    global_cache['last_area'] = int(mask.sum())  # mutation
    return global_cache['last_area']
```

### Transformations Over Mutations

```python
# ❌ Bad: Mutates input
instances.sort(key=lambda x: x['score'], reverse=True)
for inst in instances:
    inst['rank'] = len(instances) - instances.index(inst)

# ✅ Good: Returns new data
ranked_instances = sorted(instances, key=lambda x: x['score'], reverse=True)
ranked_with_position = [
    {**inst, 'rank': idx}
    for idx, inst in enumerate(ranked_instances)
]
```

### Map, Filter, Reduce Patterns

```python
# ✅ Declarative transformation
high_confidence_masks = [
    inst['mask']
    for inst in instances
    if inst['score'] > 0.7
]

# ✅ With map/filter
high_confidence_masks = list(map(
    lambda inst: inst['mask'],
    filter(lambda inst: inst['score'] > 0.7, instances)
))
```

### Push Side Effects to Boundaries

Pure domain logic lives in the center. I/O and mutations at the edges:

```
┌─────────────────────────────┐
│   Input/Output Layer        │  File I/O, GPU, networking
├─────────────────────────────┤
│  Application/Orchestration  │  Config loading, sequencing
├─────────────────────────────┤
│   Domain Layer (PURE)       │  Core logic, transformations
├─────────────────────────────┤
│  Infrastructure             │  Database, APIs, OS resources
└─────────────────────────────┘
```

**Example**:
```python
# ✅ Pure domain function
def normalize_instance_results(raw_output):
    # NO file I/O, NO GPU operations, NO global state
    return processed_results

# Infrastructure layer calls it
def main():
    raw_model_output = model(**inputs)  # GPU operation (infrastructure)
    instances = normalize_instance_results(raw_model_output)  # Pure
    save_results(instances)  # I/O (infrastructure)
```

## Data & State Management

### Minimize State

- Don't store what you can compute
- Avoid duplicate copies of the same data
- Prefer immutable data structures

```python
# ❌ Bad: Redundant state
class ImageBatch:
    def __init__(self, images):
        self.images = images
        self.count = len(images)  # Redundant, computable
        self.has_images = len(images) > 0  # Redundant

# ✅ Good: Computed on demand
class ImageBatch:
    def __init__(self, images):
        self.images = images
    
    @property
    def count(self):
        return len(self.images)
```

### State Naming Reveals Purpose

✅ **Good**: `selected_product_id`, `session_timeout_remaining`, `error_message`
❌ **Bad**: `value`, `item`, `data`, `flag`

### Prefer Immutability

```python
# ❌ Bad: Mutating config dict
config['learning_rate'] = 0.0001
config['batch_size'] = 32

# ✅ Good: Create new config
updated_config = {
    **config,
    'learning_rate': 0.0001,
    'batch_size': 32,
}
```

## When to Use OOP

Use objects when:
1. State and behavior naturally pair (e.g., `SegmentationTrainer` holds state and orchestrates training)
2. Invariants must be enforced internally (e.g., a config object ensures valid values)
3. Inheritance simplifies implementation (extending HF `Trainer`)
4. Encapsulation prevents accidental misuse

**Don't use objects for**:
- Data containers (use dicts or dataclasses)
- Grouping unrelated functions (that's what modules are for)
- Generic "Manager" or "Handler" concepts

## Exception Handling

### Meaningful Error Types

```python
# ❌ Bad: Silent failure
try:
    model = load_model(path)
except Exception:
    pass  # Ignored!

# ✅ Good: Named error, preserved context
class ModelLoadingError(Exception):
    """Raised when a model checkpoint cannot be loaded."""
    pass

try:
    model = load_model(path)
except FileNotFoundError as e:
    raise ModelLoadingError(f"Checkpoint not found: {path}") from e
except RuntimeError as e:
    raise ModelLoadingError(f"Model loading failed: {e}") from e
```

### Error Naming

Errors should describe the failure:
- ✅ `AuthenticationFailedError`, `InvalidConfigurationError`, `GPUNotAvailableError`
- ❌ `Error`, `BadInput`, `FailedOperation`

### Layer Responsibilities

- **Domain layer**: Throw domain-specific errors (e.g., `InvalidConfigurationError`)
- **Infrastructure layer**: Translate external errors (e.g., `FileNotFoundError` → `DataLoadingError`)
- **Input/Output layer**: Convert errors to user responses (e.g., JSON error responses, CLI messages)

## Documentation & Comments

### Comments Explain WHY, Not WHAT

```python
# ❌ Bad: Restates code
# Loop through instances and add to list
for inst in instances:
    filtered.append(inst)

# ✅ Good: Explains reasoning
# Preserve instance order for visualization consistency.
# Sorting would break visual cue for model debugging.
for inst in instances:
    filtered.append(inst)
```

### Language: Korean

All comments, docstrings, and documentation should be in Korean:

```python
# ✅ Good
def normalize_instance_results(raw_output):
    """
    모델 출력을 표준 포맷으로 변환합니다.
    
    이 함수는 Mask2Former, OneFormer 등 다양한 모델의 출력을 통일된 형식으로
    변환하여 시각화 파이프라인에서 사용할 수 있게 합니다.
    """
    ...
```

## Testing Principles

### Design for Testability

Hide implementation details that require mocking:

```python
# ❌ Hard to test: Hidden dependencies
def predict(image_path):
    model = GLOBAL_MODEL  # Hidden dependency
    config = load_config()  # Hidden file I/O
    return model.predict(image)

# ✅ Easy to test: Explicit dependencies
def predict(image, model, config):
    return model.predict(image, config)

# In tests: Pass mock model and config
```

### Test Naming

Tests should read like specifications:

```python
# ✅ Good: Describes the behavior
def test_높은_신뢰도의_마스크만_필터링합니다():
    high_conf = [{'score': 0.9}, {'score': 0.5}]
    result = filter_by_confidence(high_conf, threshold=0.7)
    assert len(result) == 1

# ❌ Bad: Meaningless
def test_filter():
    ...
```

### One Assertion Per Behavior

```python
# ✅ Good: Clear what's being tested
def test_마스크_정규화():
    raw = {"segmentation": tensor, "segments_info": [...]}
    result = normalize_instance_results(raw)
    assert len(result) == 2
    assert all('mask' in r for r in result)

# ❌ Bad: Multiple unrelated checks
def test_normalize():
    result = normalize_instance_results(raw)
    assert result is not None
    assert len(result) > 0
    assert result[0]['score'] > 0
    assert 'label' in result[0]
    # What am I really testing?
```

## Dependency Management

### Prefer Standard Library

Only adopt external libraries when they:
1. Reduce maintenance burden significantly
2. Have active community and stable maintenance
3. Directly reduce project complexity
4. Don't conflict with existing tools

### Current Tech Stack (Justified)

- **PyTorch, Transformers**: Necessary for deep learning research (core value)
- **OpenCV**: Essential for image processing (standard in CV)
- **Hugging Face Trainer**: Consistent with ecosystem choice (reduces code)
- **YAML**: Configuration format (simple, readable, standard)
- **uv**: Package management (fast, deterministic, modern Python standard)

### Internal Dependencies

Minimize cross-module dependencies. Prefer dependency injection:

```python
# ❌ Bad: Hidden dependency
def predict(image_path):
    from src.utils.visualization import draw_instance_overlay
    # ...

# ✅ Good: Explicit dependency
def predict(image, model, draw_visualization=None):
    results = model.predict(image)
    if draw_visualization:
        return draw_visualization(image, results)
    return results
```

## Code Review Checklist

Use these questions during review:

### Naming & Clarity
- [ ] Can I understand the intent from names alone?
- [ ] Are there any `is_`, `has_`, `can_`, `should_` prefixes?
- [ ] Are variable names domain-specific, not generic?
- [ ] Are abbreviations limited to standard terms (url, id, api, etc.)?

### Structure
- [ ] Does each file have one central responsibility?
- [ ] Does each function do one thing?
- [ ] Is state minimized (not stored if computable)?
- [ ] Are there any `utils/`, `helpers/`, `common/` modules created?

### Functional Purity
- [ ] Are side effects pushed to boundaries?
- [ ] Is domain logic free of I/O operations?
- [ ] Are data transformations declarative (map/filter) where appropriate?
- [ ] Are there any global state mutations?

### Testing & Dependency
- [ ] Can this be tested without mocking the filesystem or network?
- [ ] Are dependencies explicitly passed, not hidden?
- [ ] Are configuration values in YAML, not hardcoded?
- [ ] Is error handling meaningful (not silent failures)?

### Design Principles
- [ ] Does this change for exactly one reason (SRP)?
- [ ] Can existing code extend this without modification (OCP)?
- [ ] Are there unnecessary abstractions (YAGNI)?
- [ ] Is the code simpler than possible? (KISS)

## Prohibited Patterns

Never do these without documented justification:

- ❌ Boolean name prefixes (`is_`, `has_`, `can_`, `should_`)
- ❌ Generic abbreviations (`cfg`, `msg`, `pwd`, `ctx`, `req`, `repo`)
- ❌ Utility modules (`utils/helpers.py`, `common/`, `misc/`)
- ❌ Ignored exceptions (`except: pass`, `except Exception: pass`)
- ❌ Premature abstraction (Don't create base classes before 3 uses)
- ❌ Hardcoded values (Everything configurable should be in YAML)
- ❌ Hidden dependencies (Inject them explicitly)
- ❌ Manager/Helper/Processor classes (Be domain-specific)
- ❌ Nested subscriptions (In reactive flows)
- ❌ Global mutable state

## Decision Framework

When uncertain about design:

1. **Can I describe it in one sentence without "and"?** (SRP check)
2. **Would a smart colleague immediately understand this name?** (Naming check)
3. **Can I test this without mocking?** (Testability check)
4. **Am I solving today's problem or tomorrow's?** (YAGNI check)
5. **Is this simpler than yesterday's version?** (KISS check)

If all five are "yes", the design is solid.

## References

- [AGENTS.md](../AGENTS.md) - Project-specific guidelines
- [GEMINI.md](../GEMINI.md) - Project overview and local setup
- [src/instance_segmentation/utils/visualization.py](../src/instance_segmentation/utils/visualization.py) - Example of focused module
- [segmentation_trainer.py](../src/instance_segmentation/segmentation_trainer.py) - Example of single-responsibility orchestration
