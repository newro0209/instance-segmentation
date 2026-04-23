import json
import subprocess

import cv2

COMMON_CAMERA_RESOLUTIONS: list[tuple[int, int]] = [
    (3840, 2160),
    (2560, 1440),
    (1920, 1080),
    (1600, 900),
    (1280, 720),
    (1024, 768),
    (1024, 576),
    (960, 540),
    (800, 600),
    (640, 480),
]


def get_all_pnp_devices() -> list[dict[str, str]]:
    ps_cmd = (
        'pwsh -NoProfile -Command "Get-PnpDevice -Class Image, Camera -Status OK '
        '| Select-Object FriendlyName, Class, Status | ConvertTo-Json"'
    )
    try:
        output = subprocess.check_output(ps_cmd, shell=True).decode("utf-8", errors="ignore")
        payload = json.loads(output)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return []

    device_items = [payload] if isinstance(payload, dict) else payload
    return [
        {
            "FriendlyName": str(item.get("FriendlyName", "Unknown System Name")),
            "Class": str(item.get("Class", "Unknown")),
            "Status": str(item.get("Status", "Unknown")),
        }
        for item in device_items
        if isinstance(item, dict)
    ]


def probe_openable_indices(max_tested: int = 10) -> list[int]:
    openable_indices: list[int] = []

    for index in range(max_tested):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            openable_indices.append(index)
        cap.release()

    return openable_indices


def list_cameras(max_tested: int = 5) -> list[int]:
    available_cameras: list[int] = []
    for camera_index in range(max_tested):
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            available_cameras.append(camera_index)
            cap.release()
    return available_cameras


def probe_camera_modes(
    camera_id: int,
    candidate_resolutions: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int, float]]:
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        return []

    detected_modes: dict[tuple[int, int, int], tuple[int, int, float]] = {}
    resolutions = candidate_resolutions or COMMON_CAMERA_RESOLUTIONS

    for width, height in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(cap.get(cv2.CAP_PROP_FPS))

        if actual_w <= 0 or actual_h <= 0:
            continue
        if actual_fps <= 1.0 or actual_fps > 240:
            actual_fps = 30.0

        dedup_key = (actual_w, actual_h, int(round(actual_fps)))
        detected_modes[dedup_key] = (actual_w, actual_h, actual_fps)

    if not detected_modes:
        fallback_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fallback_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fallback_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fallback_w > 0 and fallback_h > 0:
            if fallback_fps <= 1.0 or fallback_fps > 240:
                fallback_fps = 30.0
            dedup_key = (fallback_w, fallback_h, int(round(fallback_fps)))
            detected_modes[dedup_key] = (fallback_w, fallback_h, fallback_fps)

    cap.release()
    return list(detected_modes.values())
