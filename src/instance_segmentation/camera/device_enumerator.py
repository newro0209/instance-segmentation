import sys
from instance_segmentation.camera.device_catalog import build_summary_rows, render_horizontal_table
from instance_segmentation.infrastructure.camera_probe import (
    get_all_pnp_devices,
    probe_openable_indices,
)

# 콘솔 출력이 깨지지 않도록 표 렌더링 전에 UTF-8 인코딩을 강제합니다.
stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
if callable(stdout_reconfigure):
    stdout_reconfigure(encoding='utf-8')

def main() -> None:
    # 1. 운영체제가 인식한 PnP 장치 목록과 실제 OpenCV로 열 수 있는 인덱스를 각각 수집합니다.
    pnp_devices = get_all_pnp_devices()
    openable_indices = probe_openable_indices(max_tested=len(pnp_devices))

    # 2. 두 소스를 합쳐 사람이 읽기 쉬운 요약 행으로 정리한 뒤 표 형태로 출력합니다.
    summary_rows = build_summary_rows(pnp_devices=pnp_devices, openable_indices=openable_indices)
    for line in render_horizontal_table(summary_rows):
        print(line)

if __name__ == "__main__":
    main()
