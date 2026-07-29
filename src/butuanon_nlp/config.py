import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ProjectConfig:
    workspace_root: Path
    data_path: Path
    output_dir: Path
    max_samples: int = 100
    default_backend: str = "nllb"

    @classmethod
    def from_workspace(cls, workspace_root: str | Path) -> "ProjectConfig":
        root = Path(workspace_root).resolve()
        config_path = root / "config.json"
        loaded = load_config(config_path)
        default_backend = loaded.get("default_backend", "nllb") if loaded else "nllb"
        return cls(
            workspace_root=root,
            data_path=root / "data" / "sample_data.jsonl",
            output_dir=root / "outputs",
            default_backend=default_backend,
        )


def save_config(path: str | Path, values: Dict[str, Any]) -> Path:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_config(config_path)
    existing.update(values)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2, ensure_ascii=False)
    return config_path


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
