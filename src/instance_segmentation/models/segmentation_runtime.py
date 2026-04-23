from dataclasses import dataclass
from typing import Protocol

import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForInstanceSegmentation,
    AutoModelForUniversalSegmentation,
)


@dataclass(frozen=True)
class SegmentationRuntime:
    model_name: str
    model_type: str
    image_processor: object
    model: object

    def post_process_instance_segmentation(
        self,
        outputs: object,
        target_sizes: list[tuple[int, int]],
    ) -> dict[str, object]:
        post_process = getattr(self.image_processor, "post_process_instance_segmentation", None)
        if post_process is None:
            raise AttributeError(
                f"Image processor for '{self.model_type}' does not support instance segmentation post-processing."
            )

        return post_process(outputs, target_sizes=target_sizes)[0]


class _PretrainedModelLoader(Protocol):
    @staticmethod
    def from_pretrained(model_name: str) -> "_RuntimeModel": ...


class _RuntimeModel(Protocol):
    def to(self, device: torch.device | str) -> "_RuntimeModel": ...


def load_segmentation_runtime(
    config: dict[str, object],
    checkpoint: str | None = None,
    device: torch.device | str | None = None,
) -> SegmentationRuntime:
    model_config = _require_mapping(config, "model")
    model_name = checkpoint or str(model_config["name"])
    model_type = str(model_config.get("type", "instance_segmentation")).lower()

    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model_loader = _resolve_model_loader(model_type)
    model = model_loader.from_pretrained(model_name)

    if device is not None:
        model = model.to(device)

    return SegmentationRuntime(
        model_name=model_name,
        model_type=model_type,
        image_processor=image_processor,
        model=model,
    )


def _resolve_model_loader(model_type: str) -> _PretrainedModelLoader:
    universal_model_types = {"mask2former", "oneformer", "universal_segmentation"}
    instance_model_types = {"maskrcnn", "instance_segmentation"}

    if model_type in universal_model_types:
        return AutoModelForUniversalSegmentation

    if model_type in instance_model_types:
        return AutoModelForInstanceSegmentation

    raise ValueError(f"Unsupported model.type '{model_type}'.")


def _require_mapping(config: dict[str, object], key: str) -> dict[str, object]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must include a '{key}' mapping.")
    return value