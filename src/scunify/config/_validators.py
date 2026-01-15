"""Resource validation utilities"""

from pathlib import Path

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


def validate_resources(resource_dir: Path, model_name: str) -> None:
    """
    Validate that all required resource files exist for a given model
    
    Args:
        resource_dir: Base directory containing model resources
        model_name: Name of the model to validate
    
    Raises:
        ValueError: If model_name is not defined in RESOURCES_LISTS
        FileNotFoundError: If model directory or required files are missing
    """
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
