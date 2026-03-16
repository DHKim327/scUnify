"""Model-specific configuration creators

Architecture parameters are pre-defined in config/architecture/*.yaml
and shipped with the package. These functions only generate sample config files.
"""

from pathlib import Path
import yaml


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

    output_path = config_dir / "scfoundation_config_sample.yaml"
    with open(output_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
    print(f"✅ 'scFoundation' Configuration File Created: {output_path}")


def create_scgpt_config(resource_dir: Path, config_dir: Path):
    """Create scGPT configuration files"""
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
            "batch_size": 25,
            "num_workers": 0,
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


def create_geneformer_config(resource_dir: Path, config_dir: Path):
    """Create Geneformer configuration files"""
    resource_dir = resource_dir / "Geneformer"

    config_data = {
        "model_name": "Geneformer",
        "preprocessing": {
            "use_raw_counts": True,
        },
        "inference": {
            "seed": 0,
            "batch_size": 64,
            "num_workers": 0,
            "emb_layer": -1,
            "emb_mode": "cls",
            "model_variant": "V2-104M",
        },
        "resources": {
            "model_dirs": {
                "V1-10M": (resource_dir / "Geneformer-V1-10M").as_posix(),
                "V2-104M": (resource_dir / "Geneformer-V2-104M").as_posix(),
                "V2-104M_CLcancer": (resource_dir / "Geneformer-V2-104M_CLcancer").as_posix(),
                "V2-316M": (resource_dir / "Geneformer-V2-316M").as_posix(),
            },
            "gene_dicts": {
                "104M": {
                    "gene_median_file": (resource_dir / "gene_median_dictionary_gc104M.pkl").as_posix(),
                    "token_dict_file": (resource_dir / "token_dictionary_gc104M.pkl").as_posix(),
                    "gene_name_id_file": (resource_dir / "gene_name_id_dict_gc104M.pkl").as_posix(),
                    "ensembl_mapping_file": (resource_dir / "ensembl_mapping_dict_gc104M.pkl").as_posix(),
                },
                "30M": {
                    "gene_median_file": (resource_dir / "gene_dictionaries_30m" / "gene_median_dictionary_gc30M.pkl").as_posix(),
                    "token_dict_file": (resource_dir / "gene_dictionaries_30m" / "token_dictionary_gc30M.pkl").as_posix(),
                    "gene_name_id_file": (resource_dir / "gene_dictionaries_30m" / "gene_name_id_dict_gc30M.pkl").as_posix(),
                    "ensembl_mapping_file": (resource_dir / "gene_dictionaries_30m" / "ensembl_mapping_dict_gc30M.pkl").as_posix(),
                },
            },
        },
        "model_input_size": 4096,
    }

    output_path = config_dir / "geneformer_config_sample.yaml"
    with open(output_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
    print(f"✅ 'Geneformer' Configuration File Created: {output_path}")


# Config creator mapping
CONFIG_CREATORS = {
    "scGPT": create_scgpt_config,
    "scFoundation": create_scfoundation_config,
    "UCE": create_uce_config,
    "Geneformer": create_geneformer_config,
}
