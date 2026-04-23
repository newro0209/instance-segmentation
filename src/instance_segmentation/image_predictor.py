import argparse
import yaml
import torch
from PIL import Image
import cv2
import numpy as np
from instance_segmentation.models import load_segmentation_runtime
from instance_segmentation.utils.visualization import (
    normalize_instance_results,
    draw_instance_overlay,
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

    # Load configuration
    config_path = resolve_config_path(args.config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    runtime = load_segmentation_runtime(config, checkpoint=args.checkpoint, device=device)
    model = runtime.model
    image_processor = runtime.image_processor

    print(f"Loading model: {runtime.model_name}")

    # 2. Prepare Image
    image = Image.open(args.image).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt").to(device)

    # 3. Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # 4. Post-processing
    target_sizes = [image.size[::-1]]
    raw_results = runtime.post_process_instance_segmentation(outputs, target_sizes=target_sizes)
    processed_results = normalize_instance_results(raw_results)

    # 5. Shared OpenCV visualization pipeline
    frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    overlay = draw_instance_overlay(
        frame=frame_bgr,
        instance_results=processed_results,
        id2label=model.config.id2label,
        threshold=args.threshold,
    )

    for result in processed_results:
        score = float(result.get("score", 0.0))
        if score > args.threshold:
            label = int(result.get("label", -1))
            label_name = model.config.id2label.get(label, str(label))
            print(f"Detected {label_name} with score {score:.3f}")

    if args.output:
        cv2.imwrite(args.output, overlay)
        print(f"Saved visualization to: {args.output}")

    window_name = f"Instance Segmentation ({device})"
    cv2.imshow(window_name, overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
