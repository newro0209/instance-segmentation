import argparse
from pathlib import Path
from typing import Literal

import torch
import yaml
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)

from instance_segmentation.datasets import (
    Mask2FormerTrainingCollator,
    SegmentationTrainingDataset,
    dataset_split_configured,
    load_segmentation_dataset,
)
from instance_segmentation.runtime import (
    load_segmentation_runtime,
    runtime_image_processor,
    runtime_model,
    runtime_model_name,
)
from instance_segmentation.utils.config_loader import resolve_config_path


type TrainingMode = Literal["full_training", "full_adjustment", "fine_tuning"]


def _get_config_section(config: dict[str, object], section_name: str) -> dict[str, object]:
    section = config.get(section_name)
    if isinstance(section, dict):
        return section
    return {}


def _require_mask2former_model(config: dict[str, object]) -> None:
    model_config = _get_config_section(config, "model")
    model_type = str(model_config.get("type", "mask2former")).strip().lower()
    if model_type != "mask2former":
        raise ValueError("Mask2Former 전용 프로젝트는 model.type='mask2former'만 지원합니다.")


def _resolve_segmentation_task(config: dict[str, object]) -> str:
    inference_config = _get_config_section(config, "inference")
    task_name = str(inference_config.get("task", "panoptic")).strip().lower()
    if task_name in {"panoptic", "instance"}:
        return task_name
    return "panoptic"


def _resolve_training_mode(training_config: dict[str, object], cli_mode: str | None) -> TrainingMode:
    raw_mode = cli_mode or str(training_config.get("mode", "full_adjustment"))
    normalized_mode = raw_mode.strip().lower().replace("-", "_")
    mode_aliases: dict[str, TrainingMode] = {
        "full": "full_training",
        "scratch": "full_training",
        "from_scratch": "full_training",
        "full_training": "full_training",
        "전체학습": "full_training",
        "full_adjustment": "full_adjustment",
        "full_finetuning": "full_adjustment",
        "full_fine_tuning": "full_adjustment",
        "전체조정": "full_adjustment",
        "fine_tuning": "fine_tuning",
        "finetuning": "fine_tuning",
        "head_tuning": "fine_tuning",
        "미세조정": "fine_tuning",
    }
    resolved_mode = mode_aliases.get(normalized_mode)
    if resolved_mode is None:
        raise ValueError("training.mode는 full_training, full_adjustment, fine_tuning 중 하나여야 합니다.")
    return resolved_mode


def _config_float(value: object, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _config_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _build_training_arguments(
    training_config: dict[str, object],
    dataset_config: dict[str, object],
    evaluation_available: bool,
) -> TrainingArguments:
    evaluation_strategy = str(training_config.get("eval_strategy", "steps" if evaluation_available else "no"))
    return TrainingArguments(
        output_dir=str(training_config.get("output_dir", "./results/mask2former")),
        num_train_epochs=_config_float(training_config.get("num_epochs"), 50.0),
        per_device_train_batch_size=_config_int(dataset_config.get("batch_size"), 1),
        per_device_eval_batch_size=_config_int(
            dataset_config.get("eval_batch_size"),
            _config_int(dataset_config.get("batch_size"), 1),
        ),
        learning_rate=_config_float(training_config.get("learning_rate"), 5e-5),
        weight_decay=_config_float(training_config.get("weight_decay"), 0.0),
        lr_scheduler_type=str(training_config.get("lr_scheduler_type", "linear")),
        warmup_steps=_config_int(training_config.get("warmup_steps"), 0),
        max_steps=_config_int(training_config.get("max_steps"), -1),
        gradient_accumulation_steps=_config_int(training_config.get("gradient_accumulation_steps"), 1),
        logging_steps=_config_int(training_config.get("logging_steps"), 10),
        eval_strategy=evaluation_strategy,
        eval_steps=_config_int(training_config.get("eval_steps"), 500),
        save_strategy="steps",
        save_steps=_config_int(training_config.get("save_steps"), 1000),
        save_total_limit=_config_int(training_config.get("save_total_limit"), 2),
        dataloader_num_workers=_config_int(dataset_config.get("num_workers"), 0),
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        remove_unused_columns=False,
        report_to=str(training_config.get("report_to", "none")),
    )


def _build_training_components(
    config: dict[str, object],
    training_mode: TrainingMode,
) -> tuple[Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation, str]:
    model_config = _get_config_section(config, "model")
    model_name = str(model_config["name"])

    if training_mode == "full_training":
        image_processor = AutoImageProcessor.from_pretrained(model_name)
        if not isinstance(image_processor, Mask2FormerImageProcessor):
            raise TypeError("Mask2Former 학습에는 Mask2FormerImageProcessor가 필요합니다.")
        model_configuration = AutoConfig.from_pretrained(model_name)
        if not isinstance(model_configuration, Mask2FormerConfig):
            raise TypeError("Mask2Former 전체학습에는 Mask2FormerConfig가 필요합니다.")
        model = Mask2FormerForUniversalSegmentation(model_configuration)
        return image_processor, model, model_name

    runtime = load_segmentation_runtime(config)
    return runtime_image_processor(runtime), runtime_model(runtime), runtime_model_name(runtime)


def _configure_trainable_parameters(
    model: Mask2FormerForUniversalSegmentation,
    training_mode: TrainingMode,
    training_config: dict[str, object],
) -> None:
    for _, parameter in model.named_parameters():
        parameter.requires_grad = True

    if training_mode != "fine_tuning":
        return

    for _, parameter in model.named_parameters():
        parameter.requires_grad = False

    trainable_patterns = _resolve_string_list(
        training_config.get("fine_tuning_trainable_patterns"),
        [
            "model.pixel_level_module.decoder",
            "model.transformer_module",
            "class_predictor",
        ],
    )
    for parameter_name, parameter in model.named_parameters():
        parameter.requires_grad = any(pattern in parameter_name for pattern in trainable_patterns)


def _resolve_string_list(value: object, default: list[str]) -> list[str]:
    if not isinstance(value, list):
        return default
    resolved_values = [str(item) for item in value if str(item).strip()]
    return resolved_values or default


def _count_parameters(model: Mask2FormerForUniversalSegmentation) -> tuple[int, int]:
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total_parameters, trainable_parameters


def _load_optional_eval_dataset(
    dataset_config: dict[str, object],
    split_name: str,
    config_path: Path,
) -> SegmentationTrainingDataset | None:
    if not dataset_split_configured(dataset_config, split_name):
        return None
    return load_segmentation_dataset(dataset_config, split_name, config_path)


def _resolve_processor_size(config: dict[str, object]) -> dict[str, int] | None:
    augmentation_config = _get_config_section(config, "augmentation")
    resize = augmentation_config.get("resize")
    if isinstance(resize, list) and len(resize) == 2:
        height = _config_int(resize[0], 0)
        width = _config_int(resize[1], 0)
        if height > 0 and width > 0:
            return {"height": height, "width": width}
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mask2former_config.yaml")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Training mode: full_training, full_adjustment, fine_tuning",
    )
    parser.add_argument("--dry_run", action="store_true", help="Validate setup without calling Trainer.train().")
    args = parser.parse_args()

    # 1. 설정 파일을 읽고 Mask2Former 전용 실행 조건을 먼저 검증합니다.
    config_path = resolve_config_path(args.config)
    with open(config_path, "r", encoding="utf-8") as file_handle:
        config = yaml.safe_load(file_handle)
    if not isinstance(config, dict):
        raise ValueError("설정 파일은 YAML mapping 형식이어야 합니다.")

    _require_mask2former_model(config)

    training_config = _get_config_section(config, "training")
    dataset_config = _get_config_section(config, "dataset")
    set_seed(_config_int(training_config.get("seed"), 42))
    training_mode = _resolve_training_mode(training_config, args.mode)

    # 2. 선택한 학습 모드에 맞게 모델 초기화 방식과 학습 대상 파라미터를 확정합니다.
    image_processor, model, model_name = _build_training_components(config, training_mode)
    _configure_trainable_parameters(model, training_mode, training_config)

    train_split = str(dataset_config.get("train_split", "train"))
    eval_split = str(dataset_config.get("test_split", dataset_config.get("eval_split", "validation")))
    train_dataset = load_segmentation_dataset(dataset_config, train_split, config_path)
    eval_dataset = _load_optional_eval_dataset(dataset_config, eval_split, config_path)
    training_args = _build_training_arguments(training_config, dataset_config, eval_dataset is not None)
    segmentation_task = _resolve_segmentation_task(config)
    data_collator = Mask2FormerTrainingCollator(
        image_processor=image_processor,
        ignore_index=_config_int(dataset_config.get("ignore_index"), 255),
        reduce_labels=bool(dataset_config.get("reduce_labels", False)),
        image_size=_resolve_processor_size(config),
    )
    total_parameters, trainable_parameters = _count_parameters(model)

    print(f"Mask2Former training setup complete: {model_name}")
    print(f"Training mode: {training_mode}")
    print(f"Segmentation task: {segmentation_task}")
    print(f"Output directory: {training_args.output_dir}")
    print(f"Train samples: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"Eval samples: {len(eval_dataset)}")
    print(f"Trainable parameters: {trainable_parameters:,} / {total_parameters:,}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    if args.dry_run:
        training_batch = next(iter(trainer.get_train_dataloader()))
        print(f"Dry run batch keys: {', '.join(sorted(training_batch.keys()))}")
        print("Dry run complete. Trainer is ready.")
        return

    trainer.train()


if __name__ == "__main__":
    main()
