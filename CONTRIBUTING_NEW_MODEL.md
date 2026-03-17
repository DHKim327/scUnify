# Adding a New Model to scUnify — LLM Guide

> This document is a step-by-step guide for **developers using LLMs** (Claude Code, GPT, Gemini, Codex)
> to integrate a new single-cell foundation model into the scUnify framework.
>
> **Copy the relevant sections into your LLM prompt** along with the original model's source code.

---

## 0. Prerequisites

Before starting, ensure you have:

1. **Original model source code** accessible (any location — just need to read the code)
2. **Pretrained weights** available (will be downloaded into `resources/{ModelName}/` via `_download.py`)
3. **Understanding of the model's inference pipeline**: tokenization & dataset → model load → forward → embedding extraction

---

## 1. Architecture Overview

scUnify uses a **3-file pattern** per model. Each model needs:

```
src/scunify/
├── registry/dataset/_mymodel_dataset.py    # Data preprocessing + tokenization
├── registry/models/_mymodel_wrapper.py     # Model loading + forward pass
├── inferencer/_mymodel_inferencer.py       # Glue (3 methods, ~20 lines)
├── config/architecture/mymodel.yaml        # Model hyperparameters
└── config/envs/mymodel_env.yaml            # Conda environment spec
```

Plus **registration** in 6 existing files (detailed in Step 8).

---

## 2. Execution Flow (How scUnify Runs Inference)

```
User Config YAML
    → ScUnifyConfig(adata_dir, config_dir, save_dir)
    → resolve_inferencer(cfg) → MyModelInferencer class
    → infer = MyModelInferencer(cfg)
    → ds = infer.build_dataset(adata)      # Your Dataset class
    → dl = infer.build_dataloader(ds)       # BaseInferencer handles this
    → model = infer.build_model()           # Your Wrapper class
    → for batch in dl:
        emb, cid = infer.forward_step(model, batch)   # Your forward logic
    → save .npy + .json sidecar
```

Key points:
- **Each model runs in its own conda env** (via Ray `runtime_env`)
- **torch is NOT available** in the base scUnify env — only in model-specific envs
- **BaseInferencer** provides `build_dataloader()`, `postprocess()`, `save_outputs()` for free

---

## 3. Step-by-Step: Create the Dataset

**File**: `src/scunify/registry/dataset/_mymodel_dataset.py`

```python
import numpy as np
import torch
from torch.utils.data import Dataset, SequentialSampler

class MyModelDataset(Dataset):
    def __init__(self, adata, config):
        inference_cfg = config.get("inference", {})
        resources = config.get("resources", {})

        # Load tokenizer / vocab / gene mappings from resources
        # ... (adapt from original model's preprocessing code)

        # Store preprocessed data as numpy arrays
        self.data = ...           # (n_cells, ...) preprocessed
        self.n_cells = adata.n_obs

        # REQUIRED: sampler for BaseInferencer.build_dataloader()
        self.sampler = SequentialSampler(self)

        # OPTIONAL: custom collator (if default collation doesn't work)
        # self.collator = self._collate_fn  # or a static method

    def __len__(self):
        return self.n_cells

    def __getitem__(self, idx):
        # MUST return a dict containing at minimum:
        # - Model inputs (tensors)
        # - "cid": cell index (int) for ordering after DDP gather
        return {
            "input_ids": torch.tensor(...),
            "cid": idx,
        }

    @staticmethod
    def collator(batch):
        """Optional: custom collation for variable-length sequences."""
        # If sequences have variable length, implement padding here
        # If fixed-length, you can omit this (PyTorch default collation works)
        ...
```

### Rules for Dataset

| Rule | Why |
|------|-----|
| Return `"cid"` (cell index) in every sample | Used to restore original order after DDP gather |
| Set `self.sampler = SequentialSampler(self)` | BaseInferencer passes it to DataLoader |
| Custom `collator` as static method or `None` | BaseInferencer checks `getattr(ds, "collator", None)` |
| Keep heavy computation in `__init__`, not `__getitem__` | Avoid per-sample overhead in DataLoader workers |

### Common Pitfalls

1. **Gene ID format**: scUnify input uses **HUGO gene symbols** (e.g. `TP53`, `BRCA1`). If the model requires ENSEMBL IDs or other formats, implement conversion logic in the Dataset.
2. **Sparse matrix**: Use `issparse(X)` check — `adata.X` can be sparse or dense

---

## 4. Step-by-Step: Create the Model Wrapper

**File**: `src/scunify/registry/models/_mymodel_wrapper.py`

```python
import torch
import torch.nn as nn

class MyModelWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = load(config)
        # Store inference params from config
        inference_cfg = config.get("inference", {})
        self.some_param = inference_cfg.get("some_param", default_value)

    def forward(self, input_ids, **kwargs):
        """
        Args: batch tensors from Dataset.__getitem__
        Returns: cell embeddings, shape (batch_size, embedding_dim)
        """
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, **kwargs)

        # Extract embeddings (model-specific logic)
        hidden = outputs.last_hidden_state  # or outputs.hidden_states[layer]
        embeddings = hidden[:, 0, :]        # e.g., CLS token
        return embeddings  # (B, D)


def load(config):
    """Load pretrained model from disk."""
    # Use config.get(), not config["key"]
    inference_cfg = config.get("inference", {})
    resources = config.get("resources", {})
    model_path = resources["model_file"]  # resources is a plain dict, subscript OK

    # Load model (adapt from original code)
    model = ...
    model.eval()
    return model
```

### Rules for Wrapper

| Rule | Why |
|------|-----|
| Return shape `(B, D)` | inference_loop expects `emb:(B,D)` for `accelerator.gather_for_metrics()` |
| `torch.no_grad()` inside forward | Redundant safety (loop also has `torch.no_grad()` context) |
| `model.eval()` in `load()` | inference_loop calls `model.eval()` again, but set it early |
| `config.get()` for ScUnifyConfig fields | But `resources["key"]` is OK — `resources` is a plain dict after `.get()` |
| Keep `load()` as a standalone function | Cleaner separation; easier to debug model loading issues |

### What NOT to do

- Do NOT add training logic, optimizer, scheduler
- Do NOT add quantization/PEFT unless explicitly needed
- Do NOT import heavy libraries at module top level — use lazy imports inside `load()`

---

## 5. Step-by-Step: Create the Inferencer

**File**: `src/scunify/inferencer/_mymodel_inferencer.py`

```python
from ..registry.dataset import MyModelDataset
from ..registry.models import MyModelWrapper
from .base._baseinferencer import BaseInferencer


class MyModelInferencer(BaseInferencer):
    def build_dataset(self, adata):
        return MyModelDataset(adata, self.cfg)

    def build_model(self):
        return MyModelWrapper(self.cfg)

    def forward_step(self, model, batch):
        # Unpack batch dict from Dataset.__getitem__ / collator
        input_ids = batch["input_ids"]
        # ... other batch fields

        emb = model(input_ids)  # (B, D)
        cid = batch["cid"]     # (B,) cell indices
        return emb, cid
```

This is typically **~20 lines**. All heavy logic lives in Dataset and Wrapper.

### Rules for Inferencer

| Rule | Why |
|------|-----|
| `forward_step` returns `(emb, cid)` tuple | inference_loop unpacks exactly this |
| `emb` shape: `(B, D)`, `cid` shape: `(B,)` | Required for `accelerator.gather_for_metrics()` |
| Import Dataset/Wrapper via `registry` package | Uses lazy import system — no circular deps |

---

## 6. Step-by-Step: Architecture YAML

**File**: `src/scunify/config/architecture/mymodel.yaml`

```yaml
# Model hyperparameters — loaded via BaseInferencer.__init__()
# Available as: self.cfg.model_param (dict)
default:
  hidden_size: 768
  num_layers: 12
  # ... model-specific params

# If model has variants:
variant_A:
  hidden_size: 512
  num_layers: 6
variant_B:
  hidden_size: 1024
  num_layers: 24
```

This file is **read-only** — pre-packaged with scUnify, never generated at runtime.

---

## 7. Step-by-Step: Conda Environment

**File**: `src/scunify/config/envs/mymodel_env.yaml`

```yaml
name: scunify_mymodel
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pytorch::pytorch>=2.0.1
  - pytorch::pytorch-cuda=12.1
  - pip
  - pip:
    - transformers==4.46  # or whatever the model needs
    - accelerate>=1
    - scunify  # the package itself
```

Each model runs in its own conda env via Ray's `runtime_env`.

---

## 8. Step-by-Step: Register the Model

Update these **6 files** to register the new model:

### 8-1. `inferencer/__init__.py`

```python
# Add to _ALIAS dict:
_ALIAS = {
    ...
    "mymodel": "MyModelInferencer",
}

# Add to __all__:
__all__ = [..., "MyModelInferencer"]

# Add lazy import branch in __getattr__:
elif name == "MyModelInferencer":
    if name not in _IMPORTED:
        from ._mymodel_inferencer import MyModelInferencer
        _IMPORTED[name] = MyModelInferencer
    return _IMPORTED[name]
```

### 8-2. `registry/dataset/__init__.py`

```python
# Add to __all__:
__all__ = [..., "MyModelDataset"]

# Add lazy import branch:
elif name == "MyModelDataset":
    if name not in _IMPORTED:
        from ._mymodel_dataset import MyModelDataset
        _IMPORTED[name] = MyModelDataset
    return _IMPORTED[name]
```

### 8-3. `registry/models/__init__.py`

```python
# Add to __all__:
__all__ = [..., "MyModelWrapper"]

# Add lazy import branch:
elif name == "MyModelWrapper":
    if name not in _IMPORTED:
        from ._mymodel_wrapper import MyModelWrapper
        _IMPORTED[name] = MyModelWrapper
    return _IMPORTED[name]
```

### 8-4. `config/_validators.py`

```python
RESOURCES_LISTS = {
    ...
    "MyModel": [
        "model.pt",           # pretrained weights
        "vocab.json",         # tokenizer vocab
        # list ALL required resource files (relative to Resources/MyModel/)
    ],
}
```

### 8-5. `config/_config_creators.py`

```python
def create_mymodel_config(resource_dir: Path, config_dir: Path):
    """Create MyModel sample configuration file"""
    resource_dir = resource_dir / "MyModel"
    config_data = {
        "model_name": "MyModel",
        "preprocessing": { ... },
        "inference": {
            "seed": 0,
            "batch_size": 64,
            "num_workers": 0,
            # model-specific inference params
        },
        "resources": {
            "model_file": (resource_dir / "model.pt").as_posix(),
            # all resource paths
        },
    }
    output_path = config_dir / "mymodel_config_sample.yaml"
    with open(output_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
    print(f"✅ 'MyModel' Configuration File Created: {output_path}")

# Add to CONFIG_CREATORS:
CONFIG_CREATORS = {
    ...
    "MyModel": create_mymodel_config,
}
```

### 8-6. `config/_download.py`

```python
def download_mymodel(resource_dir: Path) -> None:
    """Download MyModel resources."""
    output_dir = resource_dir / "MyModel"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download all required files (weights, vocab, config, custom code, etc.)
    # Use hf_hub_download(), urllib, or copy from Foundations/
    ...

# Add to downloaders dict:
downloaders = {
    ...
    "MyModel": download_mymodel,
}
```

> **Important**: Download **all** files the model needs at runtime, including custom Python files for HuggingFace `trust_remote_code=True` models. The download list must match `_validators.py`'s `RESOURCES_LISTS`.

---

## 9. Checklist

Use this checklist to verify completeness:

- [ ] `registry/dataset/_mymodel_dataset.py` — Dataset class with `__init__`, `__len__`, `__getitem__`, `sampler`
- [ ] `registry/models/_mymodel_wrapper.py` — Wrapper class with `__init__`, `forward` → `(B, D)`
- [ ] `inferencer/_mymodel_inferencer.py` — Inferencer with `build_dataset`, `build_model`, `forward_step`
- [ ] `config/architecture/mymodel.yaml` — Model hyperparameters
- [ ] `config/envs/mymodel_env.yaml` — Conda env spec with **all** runtime dependencies
- [ ] `inferencer/__init__.py` — Added to `_ALIAS`, `__all__`, `__getattr__`
- [ ] `registry/dataset/__init__.py` — Added to `__all__`, `__getattr__`
- [ ] `registry/models/__init__.py` — Added to `__all__`, `__getattr__`
- [ ] `config/_validators.py` — Added to `RESOURCES_LISTS`
- [ ] `config/_config_creators.py` — Added creator function + `CONFIG_CREATORS`
- [ ] `config/_download.py` — Added download function + `downloaders` dict
- [ ] Algorithm verification: input/output identical to original model code
- [ ] `config.get()` used everywhere (never `config["key"]` on ScUnifyConfig)
- [ ] `forward_step` returns `(emb, cid)` with correct shapes

---

## 10. LLM Prompt Template

Copy this template and fill in `{placeholders}` when prompting your LLM:

```
You are integrating {ModelName} into the scUnify framework.

## scUnify Context
- scUnify is a unified zero-shot inference pipeline for single-cell RNA-seq foundation models.
- Each model has 3 files: Dataset, Wrapper, Inferencer (see CONTRIBUTING_NEW_MODEL.md).
- Config access: use `config.get("key", default)` — ScUnifyConfig has NO __getitem__.
- forward_step must return `(embeddings, cell_ids)` with shapes `(B, D)` and `(B,)`.
- Dataset must set `self.sampler = SequentialSampler(self)`.
- Dataset must return `"cid"` (cell index) in every sample dict.

## Original Model Code
{Paste the original model's key source files here:
 - Tokenization / preprocessing code
 - Model definition / loading code
 - Embedding extraction code}

## Resources Available
{List the pretrained weight files, vocab files, etc. in Resources/{ModelName}/}

## Task
1. Create `registry/dataset/_{modelname}_dataset.py`
   - Adapt the original tokenization into a PyTorch Dataset
   - Keep the algorithm IDENTICAL — only restructure into Dataset format
2. Create `registry/models/_{modelname}_wrapper.py`
   - Wrap the original model loading + forward pass
   - Return cell embeddings (B, D)
3. Create `inferencer/_{modelname}_inferencer.py`
   - ~20 lines, inherits BaseInferencer
4. Create `config/architecture/{modelname}.yaml`
5. Create `config/envs/{modelname}_env.yaml`
6. Update 5 registration files (see CONTRIBUTING_NEW_MODEL.md Step 8)

## Rules
- NEVER modify the algorithm — only restructure code
- Use config.get() for ScUnifyConfig, dict["key"] only for plain dicts
- Handle both sparse and dense adata.X
- Use lazy imports for heavy libraries (torch, transformers) inside functions
```

---

## 11. Known Gotchas

| Issue | How to Avoid |
|-------|--------------|
| `ScUnifyConfig` not subscriptable | Always use `config.get("key", default)`, never `config["key"]` |
| Gene ID mismatch | scUnify input = HUGO gene symbol. 모델이 ENSEMBL 등 다른 format을 요구하면 Dataset에서 변환 로직 구현 |
| Missing conda dependencies | env.yaml에 **런타임에 필요한 모든 패키지** 기재. `import` 시점에야 발견되므로 누락 주의 |
| scanpy in base env | base env에는 torch/scanpy 없음. `_config_creators.py` 등에서 torch import 금지. `anndata`만 사용 |
| Architecture extraction at setup | `torch.load()` 등으로 runtime에 architecture 추출하지 말 것. YAML로 pre-package |
| Download list incomplete | `_download.py`와 `_validators.py`의 파일 목록이 **정확히 일치**해야 함. 누락 시 model load에서 실패 |

---

## 12. File Reference

```
src/scunify/
├── config/
│   ├── _config.py              # ScUnifyConfig — .get() method, no __getitem__
│   ├── _config_creators.py     # CONFIG_CREATORS dict — sample YAML generators
│   ├── _validators.py          # RESOURCES_LISTS — required file validation
│   ├── architecture/*.yaml     # Model hyperparams (read-only, pre-packaged)
│   └── envs/*.yaml             # Conda env specs per model
├── core/
│   └── loops/inference_loop.py # Main inference loop (DO NOT MODIFY)
├── inferencer/
│   ├── __init__.py             # resolve_inferencer() + _ALIAS + lazy imports
│   ├── base/_baseinferencer.py # Abstract base (build_dataloader, postprocess, save)
│   └── _*_inferencer.py        # Per-model inferencer (~20 lines each)
└── registry/
    ├── dataset/
    │   ├── __init__.py         # Lazy imports
    │   └── _*_dataset.py       # Per-model Dataset classes
    └── models/
        ├── __init__.py         # Lazy imports
        ├── _*_wrapper.py       # Per-model nn.Module wrappers
        └── modules/            # Original model source (if not pip-installable)
```
