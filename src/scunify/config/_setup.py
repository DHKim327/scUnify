from pathlib import Path

import torch
import yaml

ARCHITECTURE_DIR = Path(__file__).parent / "architecture"
ARCHITECTURE_DIR.mkdir(parents=True, exist_ok=True)


RESOURCES_LISTS: dict[str, list[str]] = {
    "scFoundation": ["models.ckpt", "OS_scRNA_gene_index.19264.tsv"],
    "scGPT": [
        "args.json",
        "best_model.pt",
        "vocab.json",
    ],
    "UCE": [
        "4layer_model.torch",
        "33l_8ep_1024t_1280.torch",
        "all_tokens.torch",
        "species_chrom.csv",
        "species_offsets.pkl",
        "protein_embeddings/Danio_rerio.GRCz11.gene_symbol_to_embedding_ESM2.pt",
        "protein_embeddings/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt",
        "protein_embeddings/Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt",
        "protein_embeddings/Macaca_fascicularis.Macaca_fascicularis_6.0.gene_symbol_to_embedding_ESM2.pt",
        "protein_embeddings/Microcebus_murinus.Mmur_3.0.gene_symbol_to_embedding_ESM2.pt",
        "protein_embeddings/Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt",
        "protein_embeddings/Sus_scrofa.Sscrofa11.1.gene_symbol_to_embedding_ESM2.pt",
        "protein_embeddings/Xenopus_tropicalis.Xenopus_tropicalis_v9.1.gene_symbol_to_embedding_ESM2.pt",
    ],
}


def _create_scfoundation_config(resource_dir: Path, config_dir: Path):
    model_path = resource_dir / "scFoundation" / "models.ckpt"

    config_data = {
        "model_name": "scFoundation",
        "preprocessing": {"option" : 'F',"normalize_total": 10000.0, "log1p": True},
        "inference": {
            "version": "cell",
            "pool_type": "all",
            "tgthighres": "t4",
            "batch_size": 16,
            "num_workers": 4,
        },
        "resources": {
            "model_file": str(model_path),
            "gene_list": str(resource_dir / "scFoundation" / "OS_scRNA_gene_index.19264.tsv"),
        },
    }

    params = torch.load(model_path, map_location="cpu")

    model_params = {}
    model_params["cell"] = params["cell"]["config"]["model_config"]
    model_params["rde"] = params["rde"]["config"]["model_config"]

    output_path = config_dir / "scfoundation_config_sample.yaml"
    with open(output_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
    print(f"✅ 'scFoundation' Configuration File Created: {output_path}")

    with open(ARCHITECTURE_DIR / "scfoundation.yaml", "w") as f:
        yaml.dump(model_params, f, default_flow_style=False, sort_keys=False)
    print("✅ 'scFoundation' Parameters Saved Completed")


def _create_scgpt_config(resource_dir: Path, config_dir: Path):
    import json

    resource_dir = resource_dir / "scGPT"

    config_data = {
        "model_name": "scGPT",
        "preprocessing": {
            "filter_gene_by_counts": 400,
            "filter_cell_by_counts": 100,
            "normalize_total": 1e4,
            "log1p": True,
            "subset_hvg": False,
        },
        "inference": {
            "seed": 0,
            "batch_size": 128,
            "num_workers": 8,
        },
        "resources": {
            "model_file": (resource_dir / "best_model.pt").as_posix(),
            "vocab_file": (resource_dir / "vocab.json").as_posix(),
        },
    }

    output_path = config_dir / "scgpt_config_sample.yaml"
    with open(output_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
    print(f"✅ 'scGPT' Configuration File Created: {output_path}")

    # Get Model Params
    with open(resource_dir / "args.json", encoding="utf-8") as f:
        model_args = json.load(f)

    model_params = {
        "d_model": model_args["embsize"],
        "nhead": model_args["nheads"],
        "d_hid": model_args["d_hid"],
        "nlayers": model_args["nlayers"],
        "nlayers_cls": model_args["n_layers_cls"],
        "n_cls": 1,
        "dropout": model_args["dropout"],
        "pad_token": model_args["pad_token"],
        "pad_value": model_args["pad_value"],
        "do_mvc": model_args["MVC"],
        "do_dab": False,
        "use_batch_labels": False,
        "domain_spec_batchnorm": False,
        "explicit_zero_prob": False,
        "fast_transformer_backend": "flash",
        "pre_norm": False,
    }

    with open(ARCHITECTURE_DIR / "scgpt.yaml", "w") as f:
        yaml.dump(model_params, f, default_flow_style=False, sort_keys=False)
    print("✅ 'scGPT' Parameters Saved Completed")


def _create_uce_config(resource_dir: Path, config_dir: Path):
    resource_dir = resource_dir / "UCE"
    config_data = {
        "model_name": "UCE",
        "preprocessing": {
            "species": "human",
            "filter": False,
            "filter_genes_min_cells": 10,
            "filter_cells_min_genes": 25,
            "hv_genes": None,
        },
        "inference": {
            "seed" : 0,
            "nlayers": 4,  # or 33
            "batch_size": 64,
            "num_workers": 4,
        },
        "resources": {
            "spec_chrom_csv_path": (resource_dir / "species_chrom.csv").as_posix(),
            "token_file": (resource_dir / "all_tokens.torch").as_posix(),
            "offset_pkl_path": (resource_dir / "species_offsets.pkl").as_posix(),
            "4_layer_model": (resource_dir / "4layer_model.torch").as_posix(),
            "33_layer_model": (resource_dir / "33l_8ep_1024t_1280.torch").as_posix(),
            "protein_embeddings": {
                "human": (
                    resource_dir / "protein_embeddings" / "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt"
                ).as_posix(),
                "mouse": (
                    resource_dir / "protein_embeddings" / "Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt"
                ).as_posix(),
                "frog": (
                    resource_dir
                    / "protein_embeddings"
                    / "Xenopus_tropicalis.Xenopus_tropicalis_v9.1.gene_symbol_to_embedding_ESM2.pt"
                ).as_posix(),
                "zebrafish": (
                    resource_dir / "protein_embeddings" / "Danio_rerio.GRCz11.gene_symbol_to_embedding_ESM2.pt"
                ).as_posix(),
                "mouse_lemur": (
                    resource_dir / "protein_embeddings" / "Microcebus_murinus.Mmur_3.0.gene_symbol_to_embedding_ESM2.pt"
                ).as_posix(),
                "pig": (
                    resource_dir / "protein_embeddings" / "Sus_scrofa.Sscrofa11.1.gene_symbol_to_embedding_ESM2.pt"
                ).as_posix(),
                "macaca_fascicularis": (
                    resource_dir
                    / "protein_embeddings"
                    / "Macaca_fascicularis.Macaca_fascicularis_6.0.gene_symbol_to_embedding_ESM2.pt"
                ).as_posix(),
                "macaca_mulatta": (
                    resource_dir / "protein_embeddings" / "Macaca_mulatta.Mmul_10.gene_symbol_to_embedding_ESM2.pt"
                ).as_posix(),
            },
        },
    }

    output_path = config_dir / "uce_config_sample.yaml"
    with open(output_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
    print(f"✅ 'UCE' Configuration File Created: {output_path}")

    model_params = {}
    model_param = {}
    model_param["pad_length"] = 1536
    model_param["pad_token_idx"] = 0
    model_param["chrom_token_left_idx"] = 1
    model_param["chrom_token_right_idx"] = 2
    model_param["CHROM_TOKEN_OFFSET"] = 143574
    model_param["sample_size"] = 1024
    model_param["CXG"] = True
    model_param["output_dim"] = 1280
    model_param["d_hid"] = 5120
    model_param["token_dim"] = 5120
    model_param["cls_token_idx"] = 3
    model_params[4] = model_param.copy()
    model_params[33] = model_param.copy()
    with open(ARCHITECTURE_DIR / "uce.yaml", "w") as f:
        yaml.dump(model_params, f, default_flow_style=False, sort_keys=False)
    print("✅ 'UCE' Parameters Saved Completed")


def _validate_resources(resource_dir: Path, model_name: str) -> None:
    print(f"[{model_name}] Resource File Validation Started...")

    required_files = RESOURCES_LISTS.get(model_name)
    if not required_files:
        raise ValueError(f"'{model_name}' Resource List is not defined.")

    model_resource_path = resource_dir / model_name
    if not model_resource_path.is_dir():
        raise FileNotFoundError(
            f"'{model_resource_path}' Directory not found. "
            f"'{model_name}' Model File is not downloaded in the corresponding path."
        )

    missing_files = []
    for file_rel_path in required_files:
        file_abs_path = model_resource_path / file_rel_path
        if not file_abs_path.exists():
            missing_files.append(file_rel_path)

    if missing_files:
        error_msg = (
            f"'{model_name}' Model File is missing:\n"
            + "\n".join([f" - {f}" for f in missing_files])
            + f"\n\n'{model_resource_path}' Directory is not found."
        )
        raise FileNotFoundError(error_msg)

    print(f"[{model_name}] All Resource Files are validated.")


def setup(resource_dir: str, config_dir: str, auto_download: bool = False):
    """
    Setup configuration files for scUnify models
    
    This function validates model resources and creates configuration files.
    It can optionally download missing models automatically.
    
    Args:
        resource_dir: Directory containing model weights and resources
        config_dir: Directory to save generated configuration files
        auto_download: If True, automatically download missing models (default: False)
    
    Examples:
        >>> # Basic setup (validation only)
        >>> setup("./resources", "./configs")
        
        >>> # Setup with auto-download
        >>> setup("./resources", "./configs", auto_download=True)
    
    Raises:
        NotADirectoryError: If resource_dir is not a valid directory
    """
    resource_path = Path(resource_dir).expanduser().resolve()
    config_dir = Path(config_dir).expanduser().resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    
    if not resource_path.is_dir():
        resource_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created resource directory: {resource_path}")

    print(f"\n{'='*60}")
    print(f"🚀 scUnify Setup")
    print(f"{'='*60}")
    print(f"Resource Directory: {resource_path}")
    print(f"Config Directory: {config_dir}")
    print(f"Auto Download: {auto_download}")
    print(f"{'='*60}\n")

    success_count = 0
    failed_models = []

    for model_name in RESOURCES_LISTS.keys():
        try:
            print(f"\n[{model_name}] Starting validation...")
            _validate_resources(resource_path, model_name)
            
            print(f"[{model_name}] Creating configuration files...")
            globals()[f"_create_{model_name.lower()}_config"](resource_path, config_dir=config_dir)
            success_count += 1

        except (FileNotFoundError, ValueError) as e:
            if auto_download:
                print(f"\n⚠️  [{model_name}] Resources missing. Attempting auto-download...")
                try:
                    from ._download import download_model
                    download_model(model_name, resource_path)
                    
                    # Re-validate after download
                    print(f"\n[{model_name}] Re-validating after download...")
                    _validate_resources(resource_path, model_name)
                    
                    print(f"[{model_name}] Creating configuration files...")
                    globals()[f"_create_{model_name.lower()}_config"](resource_path, config_dir=config_dir)
                    success_count += 1
                    
                except Exception as download_error:
                    print(f"❌ [{model_name}] Auto-download failed: {download_error}")
                    failed_models.append((model_name, str(download_error)))
                    continue
            else:
                print(f"\n⚠️  [{model_name}] Configuration not created: {e}")
                print(f"     To download this model, run:")
                print(f"     >>> from scunify.config import download_model")
                print(f"     >>> download_model('{model_name}', '{resource_dir}')")
                failed_models.append((model_name, str(e)))
                continue

    print(f"\n{'='*60}")
    print(f"✅ Setup Completed!")
    print(f"{'='*60}")
    print(f"Successfully configured: {success_count}/{len(RESOURCES_LISTS)} models")
    
    if failed_models:
        print(f"\n⚠️  Failed models: {len(failed_models)}")
        for model_name, error in failed_models:
            print(f"   - {model_name}: {error[:80]}...")
    else:
        print("🎉 All models configured successfully!")
    print()
