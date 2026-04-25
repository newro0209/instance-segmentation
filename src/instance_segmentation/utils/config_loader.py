from pathlib import Path


def resolve_config_path(config_arg: str) -> Path:
    """명시 경로나 configs/<name> 후보를 순서대로 확인해 설정 파일 경로를 해석합니다."""
    config_path = Path(config_arg)

    # 1. 사용자가 절대/상대 경로를 정확히 넘긴 경우 그대로 사용합니다.
    if config_path.exists():
        return config_path

    # 2. 파일명만 넘긴 경우에는 기본 설정 디렉터리 아래에서 다시 찾습니다.
    if not config_path.is_absolute():
        candidate = Path("configs") / config_path
        if candidate.exists():
            return candidate

    # 3. 둘 다 실패하면 원래 값을 반환해 호출 측에서 오류를 드러내게 합니다.
    return config_path
