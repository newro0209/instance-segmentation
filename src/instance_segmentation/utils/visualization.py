import cv2
import numpy as np


def _to_numpy_mask(mask: object) -> np.ndarray:
    if hasattr(mask, "cpu"):
        return mask.cpu().numpy()
    return np.asarray(mask)


def _to_float_score(value: object, default: float = 1.0) -> float:
    if hasattr(value, "item"):
        return float(value.item())

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default

    return default


def _to_label_index(value: object, default: int = 1) -> int:
    if hasattr(value, "item"):
        return int(value.item())

    if isinstance(value, bool):
        return int(value)

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


def normalize_segmentation_results(raw_results: object) -> list[dict[str, object]]:
    """후처리 결과를 공통 세그멘테이션 항목 목록 형식으로 정규화합니다."""
    processed_results: list[dict[str, object]] = []

    if isinstance(raw_results, dict) and "masks" in raw_results:
        # 1. 마스크, 라벨, 점수가 분리된 출력은 인덱스를 기준으로 같은 세그먼트로 묶습니다.
        masks = raw_results.get("masks", [])
        labels = raw_results.get("labels", [])
        scores = raw_results.get("scores", [])

        for idx in range(len(masks)):
            score = _to_float_score(scores[idx], 1.0) if len(scores) > idx else 1.0
            label = _to_label_index(labels[idx], 1) if len(labels) > idx else 1
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
        # 2. segmentation map 기반 출력은 segment id를 실제 이진 마스크로 펼쳐 변환합니다.
        processed_results = _normalize_from_segmentation_map(raw_results)

    elif isinstance(raw_results, list):
        # 3. 리스트 입력은 이미 정규화된 항목과 map 기반 항목이 섞여 있을 수 있으므로
        #    항목별 형식을 판별해 같은 공통 구조로 합칩니다.
        for item in raw_results:
            if isinstance(item, dict) and "mask" in item:
                processed_results.append(
                    {
                        "score": _to_float_score(item.get("score", 1.0), 1.0),
                        "label": _to_label_index(item.get("label", 1), 1),
                        "mask": _to_numpy_mask(item["mask"]),
                    }
                )
            elif isinstance(item, dict) and "segmentation" in item and "score" in item:
                processed_results.append(
                    {
                        "score": _to_float_score(item.get("score", 1.0), 1.0),
                        "label": _to_label_index(item.get("label", 1), 1),
                        "mask": _to_numpy_mask(item["segmentation"]),
                    }
                )
            elif isinstance(item, dict) and "segmentation" in item and "segments_info" in item:
                processed_results.extend(_normalize_from_segmentation_map(item))

    return processed_results


def get_label_color(label: int) -> tuple[int, int, int]:
    """클래스 라벨마다 항상 같은 BGR 색상을 생성합니다."""
    label = int(label)
    b = (37 * (label + 1)) % 256
    g = (17 * (label + 13)) % 256
    r = (29 * (label + 29)) % 256
    return int(b), int(g), int(r)


def draw_segmentation_overlay(
    frame: np.ndarray,
    segmentation_results: list[dict[str, object]],
    id2label: dict[int, str],
    threshold: float = 0.7,
) -> np.ndarray:
    """BGR 이미지 위에 마스크, 박스, 라벨을 일관된 규칙으로 그립니다."""
    overlay = frame.copy()
    drawn_count = 0

    for result in segmentation_results:
        # 1. 임계값 이하 결과는 초기에 제외해 이후 렌더링 비용을 줄입니다.
        score = float(result.get("score", 0.0))
        if score <= threshold:
            continue

        label = int(result.get("label", -1))
        mask = result.get("mask")
        if mask is None:
            continue

        frame_h, frame_w = frame.shape[:2]
        mask_array = np.asarray(mask)

        # 2. 일부 모델은 프레임과 다른 해상도의 마스크를 반환하므로
        #    현재 프레임 크기로 최근접 리사이즈해 좌표계를 맞춥니다.
        if mask_array.ndim == 2 and (mask_array.shape[0] != frame_h or mask_array.shape[1] != frame_w):
            mask_array = cv2.resize(mask_array, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)

        # 3. 마스크가 비어 있으면 오버레이와 박스를 그리지 않습니다.
        mask_uint8 = (mask_array > 0).astype(np.uint8) * 255
        if not np.any(mask_uint8):
            continue

        # 4. 마스크 영역만 반투명 혼합해 원본 장면 정보를 유지합니다.
        color = get_label_color(label)
        mask_indices = mask_uint8 > 0
        overlay_region = overlay[mask_indices].astype(np.float32)
        color_region = np.array(color, dtype=np.float32)
        blended_region = overlay_region * 0.7 + color_region * 0.3
        overlay[mask_indices] = blended_region.astype(np.uint8)

        # 5. 외곽선을 합쳐 단일 bounding box를 만들고, 그 위에 라벨 배경과 텍스트를 올립니다.
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

    # 6. 마지막에 전체 검출 수와 현재 임계값을 상태 바 형태로 요약합니다.
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
