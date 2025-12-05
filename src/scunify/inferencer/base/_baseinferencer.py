# core/infer/base.py
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...utils import load_yaml


class BaseInferencer(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg.model_param = load_yaml(cfg._architecture_dir)

    @abstractmethod
    def build_dataset(self, adata): ...

    # ----- Model -----
    @abstractmethod
    def build_model(self): ...

    @abstractmethod
    def forward_step(self, model, batch): ...

    def build_dataloader(self, ds):
        from torch.utils.data import DataLoader

        inf = self.cfg.get("inference", {})
        bs = int(inf.get("batch_size", 32))
        nw = int(inf.get("num_workers", 0))

        collator = getattr(ds, "collator", None)
        sampler = getattr(ds, "sampler", None)

        return DataLoader(
            ds,
            batch_size=bs,
            num_workers=nw,
            shuffle=False,
            sampler=sampler,
            collate_fn=collator,
            pin_memory=True,
            persistent_workers=(nw > 0),
            drop_last=False,
        )

    # ----- Output handling -----
    def postprocess(self, gathered_outputs: Iterable[torch.Tensor]) -> np.ndarray | None:
        """워커별 모인 텐서들을 병합해 numpy로 반환."""
        outs = [t.detach().cpu() for t in gathered_outputs if t is not None]
        if not outs:
            return None
        Y = torch.cat(outs, dim=0).float().cpu().numpy()
        return Y

    def save_outputs(self, outputs: np.ndarray, out_path: str, meta: dict[str, Any]) -> None:
        """결과 저장: .npy + .json(sidecar)."""
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), outputs.astype(np.float32), allow_pickle=False)
        with open(str(p.with_suffix(".json")), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
