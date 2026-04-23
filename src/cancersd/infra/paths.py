from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = PROJECT_ROOT / "configs"


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)

    return path


@dataclass(frozen=True)
class RuntimePaths:
    project_root: Path
    config_root: Path
    data_root: Path
    output_root: Path
    log_root: Path

    def ensure_dirs(self) -> None:
        for path in (self.data_root, self.log_root, self.config_root):
            mkdir(path)


def resolve_path(path_str: str | None, base: Path = PROJECT_ROOT) -> Path:
    if path_str is None or str(path_str).strip() == "":
        return base.resolve()

    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()

    return (base / path).resolve()


def prepare_runtime_paths(cfg: dict[str, Any]) -> RuntimePaths:
    paths_cfg = cfg.setdefault('paths', {})

    project_root_value = paths_cfg.get('project_root')
    if project_root_value  is None:
        project_root = PROJECT_ROOT
    else:
        project_root = resolve_path(project_root_value, PROJECT_ROOT)

    config_root = project_root / 'configs'

    data_root = resolve_path(paths_cfg.get('data_root', 'data'), project_root)
    output_root = resolve_path(paths_cfg.get('output_root', 'outputs'), project_root)
    log_root = resolve_path(paths_cfg.get('log_root', 'logs'), project_root)

    runtime_paths = RuntimePaths(
        project_root=project_root,
        config_root=config_root,
        data_root=data_root,
        output_root=output_root,
        log_root=log_root,
    )
    runtime_paths.ensure_dirs()

    return runtime_paths
