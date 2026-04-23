import numpy as np


def infer_model_target_size(config: dict[str, object], image_processor: object) -> tuple[int, int]:
    augmentation = config.get("augmentation", {}) if isinstance(config, dict) else {}
    resize = augmentation.get("resize") if isinstance(augmentation, dict) else None

    if isinstance(resize, (list, tuple)) and len(resize) == 2:
        resize_h, resize_w = int(resize[0]), int(resize[1])
        if resize_h > 0 and resize_w > 0:
            return resize_w, resize_h

    processor_size = getattr(image_processor, "size", None)

    if isinstance(processor_size, dict):
        if "width" in processor_size and "height" in processor_size:
            return int(processor_size["width"]), int(processor_size["height"])
        if "longest_edge" in processor_size:
            edge = int(processor_size["longest_edge"])
            return edge, edge
        if "shortest_edge" in processor_size:
            edge = int(processor_size["shortest_edge"])
            return edge, edge

    if isinstance(processor_size, int):
        return int(processor_size), int(processor_size)

    return 512, 512


def select_best_camera_mode(
    modes: list[tuple[int, int, float]],
    target_w: int,
    target_h: int,
) -> tuple[int, int, float] | None:
    if not modes:
        return None

    target_aspect = target_w / target_h
    target_sum = max(target_w + target_h, 1)

    def mode_sort_key(mode: tuple[int, int, float]) -> tuple[float, float, float, float, float]:
        width, height, fps = mode
        if width <= 0 or height <= 0:
            return (2, 9999.0, 9999.0, 0.0, 0.0)

        aspect = width / height
        aspect_diff = abs(aspect - target_aspect)

        if width >= target_w and height >= target_h:
            overshoot = ((width - target_w) + (height - target_h)) / target_sum
            return (0, aspect_diff, overshoot, 0.0, -fps)

        shortfall = (max(0, target_w - width) + max(0, target_h - height)) / target_sum
        return (1, aspect_diff, shortfall, -(width * height), -fps)

    return min(modes, key=mode_sort_key)


def center_crop_to_aspect(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    height, width = frame.shape[:2]
    target_aspect = target_w / target_h
    frame_aspect = width / height

    if frame_aspect > target_aspect:
        crop_w = max(1, int(round(height * target_aspect)))
        x0 = max(0, (width - crop_w) // 2)
        return frame[:, x0:x0 + crop_w]

    if frame_aspect < target_aspect:
        crop_h = max(1, int(round(width / target_aspect)))
        y0 = max(0, (height - crop_h) // 2)
        return frame[y0:y0 + crop_h, :]

    return frame
