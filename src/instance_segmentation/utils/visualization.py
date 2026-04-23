import cv2
import numpy as np


def _to_numpy_mask(mask: object) -> np.ndarray:
    if hasattr(mask, "cpu"):
        return mask.cpu().numpy()
    return np.asarray(mask)


def _normalize_from_segmentation_map(result_dict: dict[str, object]) -> list[dict[str, object]]:
    processed_results: list[dict[str, object]] = []
    segmentation = result_dict.get("segmentation")
    segments_info = result_dict.get("segments_info", [])

    if segmentation is None or not segments_info:
        return processed_results

    segmentation_array = _to_numpy_mask(segmentation)
    for segment in segments_info:
        if not isinstance(segment, dict):
            continue
        segment_id = segment.get("id")
        if segment_id is None:
            continue

        processed_results.append(
            {
                "score": float(segment.get("score", 1.0)),
                "label": int(segment.get("label_id", segment.get("label", -1))),
                "mask": segmentation_array == int(segment_id),
            }
        )

    return processed_results


def normalize_instance_results(raw_results: object) -> list[dict[str, object]]:
    """Normalize post-processed instance segmentation output to a list of dicts."""
    processed_results: list[dict[str, object]] = []

    if isinstance(raw_results, dict) and "masks" in raw_results:
        masks = raw_results.get("masks", [])
        labels = raw_results.get("labels", [])
        scores = raw_results.get("scores", [])

        for idx in range(len(masks)):
            score = scores[idx].item() if len(scores) > idx else 1.0
            label = labels[idx].item() if len(labels) > idx else -1
            mask = masks[idx]

            mask = _to_numpy_mask(mask)

            processed_results.append(
                {
                    "score": float(score),
                    "label": int(label),
                    "mask": mask,
                }
            )

    elif isinstance(raw_results, dict) and "segmentation" in raw_results and "segments_info" in raw_results:
        processed_results = _normalize_from_segmentation_map(raw_results)

    elif isinstance(raw_results, list):
        for item in raw_results:
            if isinstance(item, dict) and "mask" in item:
                processed_results.append(item)
            elif isinstance(item, dict) and "segmentation" in item and "segments_info" in item:
                processed_results.extend(_normalize_from_segmentation_map(item))

    return processed_results


def get_label_color(label: int) -> tuple[int, int, int]:
    """Generate a deterministic BGR color for a class label."""
    label = int(label)
    b = (37 * (label + 1)) % 256
    g = (17 * (label + 13)) % 256
    r = (29 * (label + 29)) % 256
    return int(b), int(g), int(r)


def draw_instance_overlay(
    frame: np.ndarray,
    instance_results: list[dict[str, object]],
    id2label: dict[int, str],
    threshold: float = 0.7,
) -> np.ndarray:
    """Draw masks, bounding boxes, and labels on a BGR image."""
    overlay = frame.copy()
    drawn_count = 0

    for result in instance_results:
        score = float(result.get("score", 0.0))
        if score <= threshold:
            continue

        label = int(result.get("label", -1))
        mask = result.get("mask")
        if mask is None:
            continue

        frame_h, frame_w = frame.shape[:2]
        mask_array = np.asarray(mask)

        # Some models can return masks at a different resolution than the frame.
        if mask_array.ndim == 2 and (mask_array.shape[0] != frame_h or mask_array.shape[1] != frame_w):
            mask_array = cv2.resize(mask_array, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)

        mask_uint8 = (mask_array > 0).astype(np.uint8) * 255
        if not np.any(mask_uint8):
            continue

        color = get_label_color(label)
        mask_indices = mask_uint8 > 0
        overlay_region = overlay[mask_indices].astype(np.float32)
        color_region = np.array(color, dtype=np.float32)
        blended_region = overlay_region * 0.7 + color_region * 0.3
        overlay[mask_indices] = blended_region.astype(np.uint8)

        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        merged = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(merged)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

        label_name = id2label.get(label, str(label))
        label_text = f"{label_name}: {score:.2f}"

        (text_w, text_h), baseline = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1,
        )

        text_x = x
        text_y = max(y - 8, text_h + 4)

        cv2.rectangle(
            overlay,
            (text_x, text_y - text_h - baseline - 4),
            (text_x + text_w + 6, text_y + baseline - 2),
            color,
            -1,
        )

        cv2.putText(
            overlay,
            label_text,
            (text_x + 3, text_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        drawn_count += 1

    status_text = f"Detections: {drawn_count} | Threshold: {threshold:.2f}"
    cv2.rectangle(overlay, (10, 10), (330, 38), (0, 0, 0), -1)
    cv2.putText(
        overlay,
        status_text,
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return overlay
