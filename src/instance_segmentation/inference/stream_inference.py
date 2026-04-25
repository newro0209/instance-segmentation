import argparse
from time import perf_counter
from pathlib import Path
from collections.abc import Iterator
from typing import cast
import yaml
import torch
import cv2
from PIL import Image
from instance_segmentation.camera.selection import (
    compute_center_crop_bounds,
    center_crop_to_aspect,
    infer_model_target_size,
    select_crop_source_camera_mode,
)
from instance_segmentation.runtime import (
    load_segmentation_runtime,
    predict_segmentation_segments,
    resolve_runtime_labels,
    runtime_image_processor,
    runtime_model,
    runtime_model_name,
)
from instance_segmentation.infrastructure.camera_probe import COMMON_CAMERA_RESOLUTIONS
from instance_segmentation.utils.visualization import (
    draw_segmentation_overlay,
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

    # 1. 실행 장치를 먼저 확정합니다.
    #    이후 런타임 생성, 입력 텐서 준비, 메모리 출력까지 모두 이 장치 기준으로 맞춥니다.
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"[*] CUDA Available: {cuda_available}")
    print(f"[*] Target Device: {device}")

    runtime = load_segmentation_runtime(config, device=device)
    print(f"[*] Loading model: {runtime_model_name(runtime)}...")

    image_processor = runtime_image_processor(runtime)
    model = runtime_model(runtime)
    id2label = resolve_runtime_labels(runtime)
    evaluate_model = getattr(model, "eval", None)
    if callable(evaluate_model):
        evaluate_model()

    target_w, target_h = infer_model_target_size(config, image_processor)
    print(f"[*] Model target size: {target_w}x{target_h}")

    candidate_modes = [(width, height, 30.0) for width, height in COMMON_CAMERA_RESOLUTIONS]
    selected_mode = select_crop_source_camera_mode(candidate_modes, target_w, target_h)

    # 2. 카메라 입력 모드를 준비합니다.
    #    모델 입력 비율에 최대한 유리한 해상도를 먼저 고르고,
    #    실제 스트림에서는 중앙 크롭으로 종횡비를 다시 정렬합니다.
    cap = cv2.VideoCapture(args.camera_id)

    if not cap.isOpened():
        print(f"[!] Error: Could not open camera {args.camera_id}")
        return

    if selected_mode:
        selected_w, selected_h, selected_fps = selected_mode
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, selected_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, selected_h)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or selected_w
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or selected_h
        actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if actual_fps <= 1.0 or actual_fps > 240:
            actual_fps = selected_fps
        print(
            f"[*] Selected camera mode: {actual_w}x{actual_h} @ {actual_fps:.1f}fps "
            f"(first larger crop source from candidate list)"
        )

    print(f"[*] Camera {args.camera_id} started. Device: {device}. Press 'q' to quit.")

    if args.no_show:
        print("[*] Running in no-show mode.")

    if args.no_show and not args.output:
        print("[*] No display and no output path: running inference-only mode.")

    video_writer = None

    frame_count = 0
    previous_frame_time = perf_counter()
    smoothed_fps = 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            crop_y0, crop_y1, crop_x0, crop_x1 = compute_center_crop_bounds(
                frame.shape[1],
                frame.shape[0],
                target_w,
                target_h,
            )

            # 3. 원본 프레임을 모델 종횡비에 맞춰 중앙 크롭합니다.
            #    이렇게 해야 리사이즈 전에 시야 왜곡을 줄일 수 있습니다.
            cropped_frame = center_crop_to_aspect(frame, target_w, target_h)

            # 4. OpenCV BGR 프레임을 PIL RGB 이미지로 바꿔 전처리기에 전달합니다.
            rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # 5. 전처리 결과를 추론 장치로 명시적으로 이동합니다.
            #    모델과 입력 장치가 어긋나면 성능 저하나 런타임 오류가 발생할 수 있습니다.
            segmentation_results = predict_segmentation_segments(runtime, pil_image, device=device)

            frame_count += 1
            if frame_count % 10 == 0:
                mem = torch.cuda.memory_allocated(device) / 1024**2 if cuda_available else 0
                # 장치 불일치 문제를 빨리 찾기 위해 실제 모델 파라미터 위치를 함께 출력합니다.
                actual_device = _resolve_model_device(model, fallback_device=device)
                print(f"[*] [Frame {frame_count}] Model Device: {actual_device} | GPU Mem: {mem:.2f}MB")

            # 6. 모델 출력을 프레임 좌표계 기준 결과로 후처리합니다.
            #    1) 현재 크롭 프레임 크기를 기준으로 마스크를 복원하고
            #    2) 공통 포맷으로 정규화한 뒤
            #    3) 임계값을 적용해 시각화 오버레이를 만듭니다.
            overlay = draw_segmentation_overlay(
                frame=cropped_frame,
                segmentation_results=segmentation_results,
                id2label=id2label,
                threshold=args.threshold,
            )

            current_frame_time = perf_counter()
            frame_delta = max(current_frame_time - previous_frame_time, 1e-6)
            current_fps = 1.0 / frame_delta
            smoothed_fps = current_fps if smoothed_fps == 0.0 else (smoothed_fps * 0.9) + (current_fps * 0.1)
            previous_frame_time = current_frame_time

            display_frame = frame.copy()
            display_frame[crop_y0:crop_y1, crop_x0:crop_x1] = overlay
            _draw_fps_overlay(display_frame, smoothed_fps)

            if args.output:
                if video_writer is None:
                    # 7. 출력 비디오 라이터는 첫 프레임 크기가 확정된 뒤 한 번만 초기화합니다.
                    output_path = Path(args.output)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    frame_h, frame_w = display_frame.shape[:2]
                    output_fps = cap.get(cv2.CAP_PROP_FPS)
                    if output_fps <= 1.0 or output_fps > 240:
                        output_fps = 30.0
                    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (frame_w, frame_h))
                    print(f"[*] Saving stream output to: {output_path}")
                video_writer.write(display_frame)

            if not args.no_show:
                cv2.imshow(f"Mask2Former Segmentation ({device})", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 8. 테스트나 배치 실행에서는 지정한 프레임 수만 처리하고 종료합니다.
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


def _resolve_model_device(model: object, fallback_device: torch.device) -> torch.device:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return fallback_device

    try:
        parameter_iterator = cast(Iterator[object], parameters())
        first_parameter = next(parameter_iterator)
        parameter_device = getattr(first_parameter, "device", fallback_device)
        if isinstance(parameter_device, torch.device):
            return parameter_device
        return torch.device(str(parameter_device))
    except (StopIteration, TypeError):
        return fallback_device


def _draw_fps_overlay(frame: cv2.typing.MatLike, fps: float) -> None:
    frame_height, frame_width = frame.shape[:2]
    fps_text = f"FPS: {fps:.1f}"
    (text_width, text_height), baseline = cv2.getTextSize(
        fps_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        2,
    )

    padding_x = 12
    padding_y = 10
    origin_x = max(10, frame_width - text_width - (padding_x * 2) - 10)
    origin_y = min(frame_height - 10, 18 + text_height)

    cv2.rectangle(
        frame,
        (origin_x, origin_y - text_height - baseline - padding_y),
        (origin_x + text_width + (padding_x * 2), origin_y + baseline),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        frame,
        fps_text,
        (origin_x + padding_x, origin_y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (80, 255, 80),
        2,
        cv2.LINE_AA,
    )

if __name__ == "__main__":
    main()
