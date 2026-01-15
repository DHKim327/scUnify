"""Model-specific configuration creators"""

from pathlib import Path
import torch
import yaml

ARCHITECTURE_DIR = Path(__file__).parent / "architecture"
ARCHITECTURE_DIR.mkdir(parents=True, exist_ok=True)


def create_scfoundation_config(resource_dir: Path, config_dir: Path):
    """Create scFoundation configuration files"""
    model_path = resource_dir / "scFoundation" / "models.ckpt"

    config_data = {
        "model_name": "scFoundation",
        "preprocessing": {"option": 'F', "normalize_total": 10000.0, "log1p": True},
        "inference": {
            "version": "cell",
            "pool_type": "all",
            "tgthighres": "t4",
            "batch_size": 1,
            "num_workers": 0,
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


def create_scgpt_config(resource_dir: Path, config_dir: Path):
    """Create scGPT configuration files"""
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
            "batch_size": 64,
            "num_workers": 0,
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


def create_uce_config(resource_dir: Path, config_dir: Path):
    """Create UCE configuration files"""
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
            "seed": 0,
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


# Config creator mapping
CONFIG_CREATORS = {
    "scGPT": create_scgpt_config,
    "scFoundation": create_scfoundation_config,
    "UCE": create_uce_config,
}
