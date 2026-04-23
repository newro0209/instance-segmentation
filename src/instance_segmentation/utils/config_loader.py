from pathlib import Path


def resolve_config_path(config_arg: str) -> Path:
    """Resolve config path from explicit path or configs/<name>."""
    config_path = Path(config_arg)

    if config_path.exists():
        return config_path

    if not config_path.is_absolute():
        candidate = Path("configs") / config_path
        if candidate.exists():
            return candidate

    return config_path
