import argparse
from pathlib import Path
import yaml
import torch
import cv2
from PIL import Image
from instance_segmentation.domain.camera_selection import (
    center_crop_to_aspect,
    infer_model_target_size,
    select_best_camera_mode,
)
from instance_segmentation.models import load_segmentation_runtime
from instance_segmentation.infrastructure.camera_probe import probe_camera_modes
from instance_segmentation.utils.visualization import (
    normalize_instance_results,
    draw_instance_overlay,
)
from instance_segmentation.utils.config_loader import resolve_config_path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="mask2former_config.yaml",
        help="Path to config yaml (default: mask2former_config.yaml)",
    )
    parser.add_argument("--camera_id", type=int, default=0, help="ID of the camera to use")
    parser.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold")
    parser.add_argument("--no_show", action="store_true", help="Run without OpenCV display window")
    parser.add_argument("--output", type=str, default=None, help="Path to save output video")
    parser.add_argument("--max_frames", type=int, default=0, help="Stop after N frames (0 means infinite)")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Device selection with logging
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"[*] CUDA Available: {cuda_available}")
    print(f"[*] Target Device: {device}")

    runtime = load_segmentation_runtime(config, device=device)
    print(f"[*] Loading model: {runtime.model_name}...")

    image_processor = runtime.image_processor
    model = runtime.model
    model.eval()

    target_w, target_h = infer_model_target_size(config, image_processor)
    print(f"[*] Model target size: {target_w}x{target_h}")

    available_modes = probe_camera_modes(args.camera_id)
    selected_mode = select_best_camera_mode(available_modes, target_w, target_h)

    # 2. Camera Setup
    cap = cv2.VideoCapture(args.camera_id)

    if not cap.isOpened():
        print(f"[!] Error: Could not open camera {args.camera_id}")
        return

    if selected_mode:
        selected_w, selected_h, selected_fps = selected_mode
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, selected_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, selected_h)
        print(
            f"[*] Selected camera mode: {selected_w}x{selected_h} @ {selected_fps:.1f}fps "
            f"(closest larger match to model target)"
        )

    print(f"[*] Camera {args.camera_id} started. Device: {device}. Press 'q' to quit.")

    if args.no_show:
        print("[*] Running in no-show mode.")

    if args.no_show and not args.output:
        print("[*] No display and no output path: running inference-only mode.")

    video_writer = None

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cropped_frame = center_crop_to_aspect(frame, target_w, target_h)

            rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # 3. Explicitly move inputs to GPU
            inputs = image_processor(images=pil_image, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            frame_count += 1
            if frame_count % 10 == 0:
                mem = torch.cuda.memory_allocated(device) / 1024**2 if cuda_available else 0
                # Check model parameter device to be 100% sure
                actual_device = next(model.parameters()).device
                print(f"[*] [Frame {frame_count}] Model Device: {actual_device} | GPU Mem: {mem:.2f}MB")

            # 4. Post-processing
            target_sizes = [(cropped_frame.shape[0], cropped_frame.shape[1])]
            raw_results = runtime.post_process_instance_segmentation(outputs, target_sizes=target_sizes)
            processed_results = normalize_instance_results(raw_results)
            overlay = draw_instance_overlay(
                frame=cropped_frame,
                instance_results=processed_results,
                id2label=model.config.id2label,
                threshold=args.threshold,
            )

            if args.output:
                if video_writer is None:
                    output_path = Path(args.output)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    frame_h, frame_w = overlay.shape[:2]
                    output_fps = cap.get(cv2.CAP_PROP_FPS)
                    if output_fps <= 1.0 or output_fps > 240:
                        output_fps = 30.0
                    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (frame_w, frame_h))
                    print(f"[*] Saving stream output to: {output_path}")
                video_writer.write(overlay)

            if not args.no_show:
                cv2.imshow(f"Instance Segmentation ({device})", overlay)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if args.max_frames > 0 and frame_count >= args.max_frames:
                print(f"[*] Reached max_frames={args.max_frames}, stopping stream.")
                break
    except KeyboardInterrupt:
        print("[*] Stream interrupted by user.")
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
