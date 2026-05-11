import os
import pickle
import sys
from pathlib import Path

from ..utils import load_yaml


class ScUnifyConfig:
    def __init__(self, adata_dir: str, config_dir: str, save_dir: str = "./scunify_results", task_name=None):
        self.adata_dir = Path(adata_dir)
        self.config_dir = Path(config_dir)
        self.config = load_yaml(self.config_dir)
        self._parse_configs()
        self._architecture_dir = Path(__file__).resolve().parent / "architecture" / f"{self.model_name.lower()}.yaml"
        self.task_name = (
            task_name if task_name is not None else f"{self.adata_dir.name.split('.')[0]}_{self.model_name}"
        )
        self.save_key = self.adata_dir.name
        self.save_dir = Path(save_dir).expanduser().resolve()
        self.save_dir.mkdir(exist_ok=True, parents=True)
        # Auto-discover ``Codes/`` next to the yaml so user mixins referenced
        # by Python import path in ``training.task_param.mixin`` are loadable
        # in both this (main) process and Ray workers.
        self._discover_user_codes()

    def _discover_user_codes(self):
        """Find ``Codes/`` near the yaml so Python imports like
        ``my_mil.MILMixin`` resolve. Looks both next to the yaml AND one
        level up (typical layout: yaml in ``Configs/``, mixin in ``Codes/``,
        both under one workspace root).

        Adds to:
          - this process's ``sys.path``
          - ``SCUNIFY_USER_CODES_PATHS`` env var (Ray workers inherit)
        """
        candidates = [
            self.config_dir.parent / "Codes",         # yaml 옆 (Configs 없이 평평한 layout)
            self.config_dir.parent.parent / "Codes",  # workspace root (Configs/ + Codes/ 형제)
        ]
        for candidate in candidates:
            if not candidate.is_dir():
                continue
            codes_str = str(candidate.resolve())
            if codes_str not in sys.path:
                sys.path.insert(0, codes_str)
            existing = os.environ.get("SCUNIFY_USER_CODES_PATHS", "")
            paths = [p for p in existing.split(os.pathsep) if p]
            if codes_str not in paths:
                paths.insert(0, codes_str)
            os.environ["SCUNIFY_USER_CODES_PATHS"] = os.pathsep.join(paths)
            return  # use first match

    def _parse_configs(self):
        for key, value in self.config.items():
            setattr(self, key, value)

    def save(self, output_path: str):
        with open(output_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self):
        return f"{self.__class__.__name__}(task_name='{self.task_name}')"

    def get(self, key: str, default=None):
        if hasattr(self, key):
            return getattr(self, key)
        return self.config.get(key, default) if isinstance(self.config, dict) else default
