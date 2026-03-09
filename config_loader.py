from pathlib import Path

import yaml


def load_config(config_path: str | None = None) -> dict:
    path = Path(config_path) if config_path else Path(__file__).parent / "cfg.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        return yaml.safe_load(f)


train_config: dict = load_config()
