from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from transformers import Mask2FormerImageProcessor


type SemanticMapping = dict[int, int]


class SegmentationTrainingDataset(Dataset[dict[str, object]]):
    def __init__(self, samples: list[dict[str, object]]) -> None:
        if not samples:
            raise ValueError("학습 데이터셋에 사용할 샘플이 없습니다.")
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self._samples[index]
        image_path = _require_path(sample, "image_path")
        image = Image.open(image_path).convert("RGB")

        segmentation_map = sample.get("segmentation_map")
        if segmentation_map is None:
            segmentation_map_path = _require_path(sample, "segmentation_map_path")
            segmentation_map = Image.open(segmentation_map_path)

        semantic_mapping = sample.get("instance_id_to_semantic_id")
        if not isinstance(semantic_mapping, dict):
            semantic_mapping = infer_semantic_mapping(segmentation_map)

        return {
            "image": image,
            "segmentation_map": segmentation_map,
            "instance_id_to_semantic_id": semantic_mapping,
        }


class Mask2FormerTrainingCollator:
    def __init__(
        self,
        image_processor: Mask2FormerImageProcessor,
        ignore_index: int = 255,
        reduce_labels: bool = False,
        image_size: dict[str, int] | None = None,
    ) -> None:
        self._image_processor = image_processor
        self._ignore_index = ignore_index
        self._reduce_labels = reduce_labels
        self._image_size = image_size

    def __call__(self, features: list[dict[str, object]]) -> dict[str, object]:
        images = [feature["image"] for feature in features]
        segmentation_maps = [feature["segmentation_map"] for feature in features]
        semantic_mappings = [
            self._prepare_semantic_mapping(feature["instance_id_to_semantic_id"])
            for feature in features
        ]

        encoded_batch = self._image_processor(
            images=images,
            segmentation_maps=segmentation_maps,
            instance_id_to_semantic_id=semantic_mappings,
            ignore_index=self._ignore_index,
            do_reduce_labels=self._reduce_labels,
            size=self._image_size,
            return_tensors="pt",
        )
        return dict(encoded_batch)

    def _prepare_semantic_mapping(self, value: object) -> SemanticMapping:
        semantic_mapping = dict(value) if isinstance(value, dict) else {}
        semantic_mapping.setdefault(0, self._ignore_index)
        return {int(key): int(mapping_value) for key, mapping_value in semantic_mapping.items()}


def load_segmentation_dataset(
    dataset_config: dict[str, object],
    split_name: str,
    config_path: Path,
) -> SegmentationTrainingDataset:
    split_config = resolve_split_config(dataset_config, split_name)
    dataset_format = str(split_config.get("format", dataset_config.get("format", "segmentation_folder"))).lower()

    if dataset_format == "segmentation_folder":
        samples = load_segmentation_folder_samples(split_config, config_path)
    elif dataset_format == "segmentation_manifest":
        samples = load_segmentation_manifest_samples(split_config, config_path)
    elif dataset_format == "coco_instance":
        samples = load_coco_instance_samples(split_config, config_path)
    else:
        raise ValueError(
            "dataset.format은 'segmentation_folder', 'segmentation_manifest', "
            "'coco_instance' 중 하나여야 합니다."
        )

    return SegmentationTrainingDataset(samples)


def dataset_split_configured(dataset_config: dict[str, object], split_name: str) -> bool:
    splits = dataset_config.get("splits")
    if isinstance(splits, dict) and isinstance(splits.get(split_name), dict):
        return True
    return any(key in dataset_config for key in ["image_dir", "annotation_file", "manifest_file"])


def resolve_split_config(dataset_config: dict[str, object], split_name: str) -> dict[str, object]:
    splits = dataset_config.get("splits")
    if isinstance(splits, dict):
        split_config = splits.get(split_name)
        if isinstance(split_config, dict):
            return {**dataset_config, **split_config}
    return dataset_config


def load_segmentation_folder_samples(
    dataset_config: dict[str, object],
    config_path: Path,
) -> list[dict[str, object]]:
    image_dir = resolve_dataset_path(_require_string(dataset_config, "image_dir"), config_path)
    segmentation_map_dir = resolve_dataset_path(_require_string(dataset_config, "segmentation_map_dir"), config_path)
    image_extensions = _resolve_extensions(dataset_config.get("image_extensions"), [".jpg", ".jpeg", ".png"])
    mask_extension = str(dataset_config.get("segmentation_map_extension", ".png"))

    _require_directory(image_dir, "dataset.image_dir")
    _require_directory(segmentation_map_dir, "dataset.segmentation_map_dir")

    samples: list[dict[str, object]] = []
    for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in image_extensions):
        segmentation_map_path = segmentation_map_dir / f"{image_path.stem}{mask_extension}"
        if not segmentation_map_path.exists():
            raise FileNotFoundError(f"세그멘테이션 맵을 찾을 수 없습니다: {segmentation_map_path}")
        samples.append({"image_path": image_path, "segmentation_map_path": segmentation_map_path})

    return samples


def load_segmentation_manifest_samples(
    dataset_config: dict[str, object],
    config_path: Path,
) -> list[dict[str, object]]:
    manifest_path = resolve_dataset_path(_require_string(dataset_config, "manifest_file"), config_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest_file을 찾을 수 없습니다: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    manifest_items = payload.get("samples") if isinstance(payload, dict) else payload
    if not isinstance(manifest_items, list):
        raise ValueError("segmentation_manifest는 리스트 또는 {'samples': [...]} 형식이어야 합니다.")

    samples: list[dict[str, object]] = []
    for item in manifest_items:
        if not isinstance(item, dict):
            continue
        semantic_mapping = item.get("instance_id_to_semantic_id")
        samples.append(
            {
                "image_path": resolve_dataset_path(_require_string(item, "image"), config_path),
                "segmentation_map_path": resolve_dataset_path(_require_string(item, "segmentation_map"), config_path),
                "instance_id_to_semantic_id": _normalize_semantic_mapping(semantic_mapping),
            }
        )

    return samples


def load_coco_instance_samples(
    dataset_config: dict[str, object],
    config_path: Path,
) -> list[dict[str, object]]:
    image_dir = resolve_dataset_path(_require_string(dataset_config, "image_dir"), config_path)
    annotation_file = resolve_dataset_path(_require_string(dataset_config, "annotation_file"), config_path)
    _require_directory(image_dir, "dataset.image_dir")

    if not annotation_file.exists():
        raise FileNotFoundError(f"COCO annotation_file을 찾을 수 없습니다: {annotation_file}")

    with open(annotation_file, "r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("COCO annotation_file에는 images와 annotations 리스트가 필요합니다.")

    annotations_by_image_id: dict[int, list[dict[str, object]]] = {}
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        image_id = int(annotation.get("image_id", -1))
        annotations_by_image_id.setdefault(image_id, []).append(annotation)

    samples: list[dict[str, object]] = []
    for image_info in images:
        if not isinstance(image_info, dict):
            continue
        image_id = int(image_info.get("id", -1))
        file_name = str(image_info.get("file_name", ""))
        width = int(image_info.get("width", 0))
        height = int(image_info.get("height", 0))
        image_annotations = annotations_by_image_id.get(image_id, [])
        if not file_name or width <= 0 or height <= 0 or not image_annotations:
            continue

        segmentation_map, semantic_mapping = build_coco_instance_segmentation_map(width, height, image_annotations)
        samples.append(
            {
                "image_path": image_dir / file_name,
                "segmentation_map": segmentation_map,
                "instance_id_to_semantic_id": semantic_mapping,
            }
        )

    return samples


def build_coco_instance_segmentation_map(
    width: int,
    height: int,
    annotations: list[dict[str, object]],
) -> tuple[np.ndarray, SemanticMapping]:
    segmentation_image = Image.new("I", (width, height), 0)
    draw = ImageDraw.Draw(segmentation_image)
    semantic_mapping: SemanticMapping = {}
    instance_id = 1

    for annotation in annotations:
        category_id = int(annotation.get("category_id", 0))
        segmentation = annotation.get("segmentation")
        bbox = annotation.get("bbox")

        drawn = draw_coco_segmentation(draw, segmentation, instance_id)
        if not drawn and isinstance(bbox, list) and len(bbox) == 4:
            x0, y0, box_w, box_h = [float(value) for value in bbox]
            draw.rectangle((x0, y0, x0 + box_w, y0 + box_h), fill=instance_id)
            drawn = True

        if drawn:
            semantic_mapping[instance_id] = category_id
            instance_id += 1

    return np.asarray(segmentation_image, dtype=np.int32), semantic_mapping


def draw_coco_segmentation(draw: ImageDraw.ImageDraw, segmentation: object, instance_id: int) -> bool:
    if not isinstance(segmentation, list):
        return False

    drawn = False
    for polygon in segmentation:
        if not isinstance(polygon, list) or len(polygon) < 6:
            continue
        points = [(float(polygon[index]), float(polygon[index + 1])) for index in range(0, len(polygon) - 1, 2)]
        draw.polygon(points, fill=instance_id)
        drawn = True
    return drawn


def infer_semantic_mapping(segmentation_map: object) -> SemanticMapping:
    segmentation_array = np.asarray(segmentation_map)
    segment_ids = np.unique(segmentation_array)
    return {int(segment_id): int(segment_id) for segment_id in segment_ids if int(segment_id) > 0}


def resolve_dataset_path(path_value: str, config_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path

    current_directory_path = Path.cwd() / path
    if current_directory_path.exists():
        return current_directory_path

    config_relative_path = config_path.parent / path
    if config_relative_path.exists():
        return config_relative_path

    return current_directory_path


def _normalize_semantic_mapping(value: object) -> SemanticMapping:
    if not isinstance(value, dict):
        return {}
    return {int(key): int(mapping_value) for key, mapping_value in value.items()}


def _resolve_extensions(value: object, default: list[str]) -> set[str]:
    if not isinstance(value, list):
        return set(default)
    return {str(extension).lower() for extension in value}


def _require_directory(path: Path, config_key: str) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{config_key} 디렉터리를 찾을 수 없습니다: {path}")


def _require_string(mapping: dict[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"dataset 설정에 '{key}' 문자열 값이 필요합니다.")
    return value


def _require_path(sample: dict[str, object], key: str) -> Path:
    value = sample.get(key)
    if not isinstance(value, Path):
        raise ValueError(f"샘플에 '{key}' 경로가 필요합니다.")
    if not value.exists():
        raise FileNotFoundError(f"샘플 파일을 찾을 수 없습니다: {value}")
    return value