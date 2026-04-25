from typing import Literal, TypedDict

import torch
from transformers import (
    AutoImageProcessor,
    BatchFeature,
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
    AutoModelForUniversalSegmentation,
)
from transformers.image_utils import ImageInput
from transformers.models.mask2former.modeling_mask2former import Mask2FormerForUniversalSegmentationOutput

from instance_segmentation.utils.visualization import normalize_segmentation_results


type DeviceLike = torch.device | str
type SegmentationTask = Literal["panoptic", "instance"]


class SegmentationRuntime(TypedDict):
    model_name: str
    model_type: str
    segmentation_task: SegmentationTask
    image_processor: Mask2FormerImageProcessor
    model: Mask2FormerForUniversalSegmentation
    inference_options: dict[str, object]
    label_ids_to_fuse: set[int]
    label_mapping: dict[int, str]


def load_segmentation_runtime(
    config: dict[str, object],
    checkpoint: str | None = None,
    device: DeviceLike | None = None,
) -> SegmentationRuntime:
    model_config = _require_mapping(config, "model")
    inference_config = _optional_mapping(config, "inference")
    model_name = checkpoint or str(model_config["name"])
    model_type = str(model_config.get("type", "mask2former")).strip().lower()
    _ensure_mask2former_model_type(model_type)
    segmentation_task = _resolve_segmentation_task(inference_config)
    label_ids_to_fuse = _resolve_label_ids_to_fuse(inference_config)

    image_processor = _load_image_processor(model_name, model_type)
    model = _load_mask2former_model(model_name)

    if device is not None:
        model = _move_model_to_device(model, device)

    return {
        "model_name": model_name,
        "model_type": model_type,
        "segmentation_task": segmentation_task,
        "image_processor": image_processor,
        "model": model,
        "inference_options": inference_config,
        "label_ids_to_fuse": label_ids_to_fuse,
        "label_mapping": _resolve_label_mapping(model),
    }


def runtime_model_name(runtime: SegmentationRuntime) -> str:
    return str(runtime["model_name"])


def runtime_image_processor(runtime: SegmentationRuntime) -> Mask2FormerImageProcessor:
    return runtime["image_processor"]


def runtime_model(runtime: SegmentationRuntime) -> Mask2FormerForUniversalSegmentation:
    return runtime["model"]


def prepare_segmentation_inputs(
    runtime: SegmentationRuntime,
    image: ImageInput,
    device: DeviceLike | None = None,
) -> BatchFeature:
    inputs = runtime_image_processor(runtime)(images=image, return_tensors="pt")
    if device is not None:
        inputs = inputs.to(device)

    return inputs


def predict_segmentation_segments(
    runtime: SegmentationRuntime,
    image: ImageInput,
    device: DeviceLike | None = None,
) -> list[dict[str, object]]:
    inputs = prepare_segmentation_inputs(runtime, image, device=device)

    with torch.no_grad(), torch.inference_mode():
        outputs = runtime_model(runtime)(**inputs)

    target_sizes = [_resolve_target_size(image)]
    raw_results = post_process_segmentation(runtime, outputs, target_sizes=target_sizes)
    return normalize_segmentation_results(raw_results)


def resolve_runtime_labels(runtime: SegmentationRuntime) -> dict[int, str]:
    label_mapping = runtime["label_mapping"]
    if label_mapping:
        return label_mapping

    return _resolve_label_mapping(runtime_model(runtime)) or {1: "object"}


def post_process_segmentation(
    runtime: SegmentationRuntime,
    outputs: Mask2FormerForUniversalSegmentationOutput,
    target_sizes: list[tuple[int, int]],
) -> dict[str, object]:
    image_processor = runtime_image_processor(runtime)
    segmentation_task = str(runtime["segmentation_task"])

    if segmentation_task == "panoptic":
        return post_process_panoptic_segmentation(runtime, outputs, target_sizes=target_sizes)

    if segmentation_task == "instance":
        return image_processor.post_process_instance_segmentation(outputs, target_sizes=target_sizes)[0]

    raise ValueError("Mask2Former 전용 런타임은 inference.task='panoptic' 또는 'instance'만 지원합니다.")


def post_process_panoptic_segmentation(
    runtime: SegmentationRuntime,
    outputs: Mask2FormerForUniversalSegmentationOutput,
    target_sizes: list[tuple[int, int]],
) -> dict[str, object]:
    image_processor = runtime_image_processor(runtime)

    return image_processor.post_process_panoptic_segmentation(
        outputs,
        target_sizes=target_sizes,
        label_ids_to_fuse=runtime["label_ids_to_fuse"],
    )[0]


def _load_image_processor(model_name: str, model_type: str) -> Mask2FormerImageProcessor:
    _ensure_mask2former_model_type(model_type)
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    if not isinstance(image_processor, Mask2FormerImageProcessor):
        raise TypeError("Mask2Former 설정에는 Mask2FormerImageProcessor가 필요합니다.")
    return image_processor


def _load_mask2former_model(model_name: str) -> Mask2FormerForUniversalSegmentation:
    model = AutoModelForUniversalSegmentation.from_pretrained(model_name)
    if not isinstance(model, Mask2FormerForUniversalSegmentation):
        raise TypeError("Mask2Former 설정에는 Mask2FormerForUniversalSegmentation 모델이 필요합니다.")
    return model


def _move_model_to_device(
    model: Mask2FormerForUniversalSegmentation,
    device: DeviceLike,
) -> Mask2FormerForUniversalSegmentation:
    target_device = torch.device(device) if isinstance(device, str) else device
    torch.nn.Module.to(model, target_device)
    return model


def _resolve_label_mapping(model: Mask2FormerForUniversalSegmentation) -> dict[int, str]:
    id2label = model.config.id2label or {}
    return {int(key): str(value) for key, value in id2label.items()}


def _ensure_mask2former_model_type(model_type: str) -> None:
    if model_type != "mask2former":
        raise ValueError("Mask2Former 전용 프로젝트는 model.type='mask2former'만 지원합니다.")


def _resolve_segmentation_task(inference_config: dict[str, object]) -> SegmentationTask:
    configured_task = inference_config.get("task")
    if isinstance(configured_task, str):
        normalized_task = configured_task.strip().lower()
        if normalized_task == "panoptic":
            return "panoptic"
        if normalized_task == "instance":
            return "instance"

    return "panoptic"


def _resolve_label_ids_to_fuse(inference_config: dict[str, object]) -> set[int]:
    configured_labels = inference_config.get("label_ids_to_fuse")
    if not isinstance(configured_labels, list):
        return set()

    resolved_labels: set[int] = set()
    for label_id in configured_labels:
        if isinstance(label_id, bool):
            resolved_labels.add(int(label_id))
        elif isinstance(label_id, int):
            resolved_labels.add(label_id)
        elif isinstance(label_id, float):
            resolved_labels.add(int(label_id))
        elif isinstance(label_id, str):
            try:
                resolved_labels.add(int(label_id))
            except ValueError:
                continue

    return resolved_labels


def _optional_mapping(config: dict[str, object], key: str) -> dict[str, object]:
    value = config.get(key)
    if isinstance(value, dict):
        return value
    return {}


def _resolve_target_size(image: object) -> tuple[int, int]:
    image_size = getattr(image, "size", None)
    if isinstance(image_size, tuple) and len(image_size) == 2:
        return int(image_size[1]), int(image_size[0])

    image_shape = getattr(image, "shape", None)
    if image_shape is not None and len(image_shape) >= 2:
        return int(image_shape[0]), int(image_shape[1])

    raise ValueError("Could not infer target size from input image.")


def _require_mapping(config: dict[str, object], key: str) -> dict[str, object]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must include a '{key}' mapping.")
    return value