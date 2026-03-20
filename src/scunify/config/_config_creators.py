"""Model-specific configuration creators.

Structure (2026-03-20):
    config/defaults/models/{model}.yaml     — model-specific (shared inference/training)
    config/defaults/training_param.yaml     — training defaults (scPEFT-based)
    config/defaults/system.yaml             — ray/accelerator defaults

Output:
    config_dir/{model_lower}/inference.yaml  — model yaml as-is
    config_dir/{model_lower}/train.yaml      — model yaml + training defaults merged
"""

import copy
from pathlib import Path

import yaml


_DEFAULTS_DIR = Path(__file__).resolve().parent / "defaults"

# Canonical model key → lowercase directory name
_MODEL_KEY_MAP = {
    "scGPT": "scgpt",
    "scFoundation": "scfoundation",
    "UCE": "uce",
    "Geneformer": "geneformer",
    "Nicheformer": "nicheformer",
}


def _load_yaml(path: Path) -> dict:
    """Load a YAML file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_and_fill(template_path: Path, resource_dir: str) -> dict:
    """Load a YAML template and replace {resource_dir} placeholders."""
    text = template_path.read_text(encoding="utf-8")
    text = text.replace("{resource_dir}", resource_dir)
    return yaml.safe_load(text)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge *override* into *base* (override wins on conflicts).

    Returns a new dict — neither input is mutated.
    """
    result = copy.deepcopy(base)
    for key, val in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(val, dict)
        ):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def _write_yaml(data: dict, output_path: Path):
    """Write dict to YAML file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


def create_config(model_name: str, resource_dir: Path, config_dir: Path):
    """Create inference and training config files for a model.

    Output structure:
        config_dir/{model_lower}/inference.yaml
        config_dir/{model_lower}/train.yaml
    """
    model_key = _MODEL_KEY_MAP.get(model_name)
    if model_key is None:
        raise ValueError(f"Unknown model: {model_name}")

    resource_str = str(resource_dir)
    model_dir = config_dir / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model template ---
    model_template = _DEFAULTS_DIR / "models" / f"{model_key}.yaml"
    if not model_template.exists():
        print(f"  ⚠️  [{model_name}] model template not found: {model_template}")
        return

    model_data = _load_and_fill(model_template, resource_str)

    # --- Inference: model data as-is ---
    inf_path = model_dir / "inference.yaml"
    _write_yaml(model_data, inf_path)
    print(f"  ✅ [{model_name}] inference.yaml → {inf_path}")

    # --- Training: model data + training defaults ---
    defaults_path = _DEFAULTS_DIR / "training_param.yaml"
    if defaults_path.exists():
        training_defaults = _load_yaml(defaults_path)
        train_data = _deep_merge(model_data, training_defaults)
        # Add mode marker
        train_data["mode"] = "training"
    else:
        train_data = copy.deepcopy(model_data)
        train_data["mode"] = "training"
        print(f"  ⚠️  [{model_name}] training defaults not found: {defaults_path}")

    train_path = model_dir / "train.yaml"
    _write_yaml(train_data, train_path)
    print(f"  ✅ [{model_name}] train.yaml → {train_path}")


# Config creator mapping — unified entry point per model
CONFIG_CREATORS = {
    model_name: lambda rd, cd, mn=model_name: create_config(mn, rd, cd)
    for model_name in _MODEL_KEY_MAP
}
