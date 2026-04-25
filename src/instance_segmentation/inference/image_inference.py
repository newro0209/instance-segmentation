import argparse

import yaml
import torch
from PIL import Image
import cv2
import numpy as np
from instance_segmentation.runtime import (
    load_segmentation_runtime,
    predict_segmentation_segments,
    resolve_runtime_labels,
    runtime_model_name,
)
from instance_segmentation.utils.visualization import (
    draw_segmentation_overlay,
)
from instance_segmentation.utils.config_loader import resolve_config_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to specific checkpoint")
    parser.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold")
    parser.add_argument("--output", type=str, default=None, help="Path to save visualized result")
    args = parser.parse_args()

    # 1. 설정 파일을 해석해 모델과 추론 옵션의 기준값을 확보합니다.
    config_path = resolve_config_path(args.config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    runtime = load_segmentation_runtime(config, checkpoint=args.checkpoint, device=device)
    id2label = resolve_runtime_labels(runtime)

    print(f"Loading model: {runtime_model_name(runtime)}")

    # 2. 입력 이미지를 RGB 기준으로 열고, 런타임 공용 세그멘테이션 경로를 실행합니다.
    image = Image.open(args.image).convert("RGB")
    segmentation_results = predict_segmentation_segments(runtime, image, device=device)

    # 5. 공용 OpenCV 시각화 파이프라인으로 오버레이를 생성합니다.
    #    이미지 추론과 스트림 추론이 같은 렌더링 규칙을 쓰도록 맞춥니다.
    frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    overlay = draw_segmentation_overlay(
        frame=frame_bgr,
        segmentation_results=segmentation_results,
        id2label=id2label,
        threshold=args.threshold,
    )

    for result in segmentation_results:
        score_value = result.get("score", 0.0)
        score = 0.0
        if isinstance(score_value, (int, float, str)) and not isinstance(score_value, bool):
            try:
                score = float(score_value)
            except ValueError:
                score = 0.0

        if score > args.threshold:
            label_value = result.get("label", -1)
            label = -1
            if isinstance(label_value, (int, float, str)) and not isinstance(label_value, bool):
                try:
                    label = int(label_value)
                except ValueError:
                    label = -1
            label_name = id2label.get(label, str(label))
            print(f"Detected {label_name} with score {score:.3f}")

    if args.output:
        cv2.imwrite(args.output, overlay)
        print(f"Saved visualization to: {args.output}")

    window_name = f"Mask2Former Segmentation ({device})"
    cv2.imshow(window_name, overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
