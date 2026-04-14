"""Model weight download utilities for scUnify"""

import os
import tarfile
from pathlib import Path
from typing import Literal

import requests
from tqdm import tqdm


def _figshare_download(url: str, save_path: Path) -> None:
    """
    Figshare download helper with progress bar

    Args:
        url: The URL of the dataset
        save_path: The path to save the dataset
    """
    if save_path.exists():
        print(f"✅ File already exists: {save_path}")
        return

    # Create directory if not exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading {save_path.name} from Figshare...")
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024

    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(save_path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    # Extract if tar.gz
    if save_path.suffix == ".gz" and save_path.stem.endswith(".tar"):
        print(f"📦 Extracting {save_path.name}...")
        with tarfile.open(save_path) as tar:
            tar.extractall(path=save_path.parent)
        print("✅ Extraction completed!")


def download_scgpt(resource_dir: Path) -> None:
    """
    Download scGPT weights from Google Drive

    Args:
        resource_dir: Base resource directory (e.g., ./resources)
    """
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required for downloading scGPT. "
            "Install it with: pip install gdown"
        )

    output_dir = resource_dir / "scGPT"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Google Drive folder URL
    url = "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y?usp=sharing"

    print("📥 Downloading scGPT from Google Drive...")
    print(f"   Output: {output_dir}")

    gdown.download_folder(url, output=str(output_dir), quiet=False, use_cookies=False)
    print("✅ scGPT download completed!")


def download_uce(resource_dir: Path) -> None:
    """
    Download UCE weights and embeddings from Figshare

    Downloads individual files via ndownloader.figshare.com direct links,
    then extracts protein_embeddings.tar.gz and removes the archive.

    Source: https://figshare.com/articles/dataset/Universal_Cell_Embedding_Model_Files/24320806

    Args:
        resource_dir: Base resource directory (e.g., ./resources)
    """
    output_dir = resource_dir / "UCE"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("📥 Downloading UCE from Figshare...")
    print(f"   Output: {output_dir}")

    # Figshare ndownloader direct links (from UCE/download.sh)
    uce_files = {
        "species_offsets.pkl": "https://ndownloader.figshare.com/files/42706555",
        "species_chrom.csv": "https://ndownloader.figshare.com/files/42706558",
        "4layer_model.torch": "https://ndownloader.figshare.com/files/42706576",
        "all_tokens.torch": "https://ndownloader.figshare.com/files/42706585",
        "protein_embeddings.tar.gz": "https://ndownloader.figshare.com/files/42715213",
        "33l_8ep_1024t_1280.torch": "https://ndownloader.figshare.com/files/43423236",
    }

    # Download each file
    for fname, url in uce_files.items():
        save_path = output_dir / fname
        _figshare_download(url, save_path)

    # Extract protein_embeddings.tar.gz if not already extracted
    pe_dir = output_dir / "protein_embeddings"
    pe_tar = output_dir / "protein_embeddings.tar.gz"
    if not pe_dir.is_dir() and pe_tar.exists():
        print("📦 Extracting protein_embeddings.tar.gz...")
        with tarfile.open(pe_tar) as tar:
            tar.extractall(path=output_dir)
        print("✅ Extraction completed!")

    # Clean up tar.gz after successful extraction
    if pe_dir.is_dir() and pe_tar.exists():
        pe_tar.unlink()
        print("🗑️  Removed protein_embeddings.tar.gz (extracted)")

    print("\n✅ UCE download completed!")


def download_scfoundation(resource_dir: Path) -> None:
    """
    Download scFoundation weights from HuggingFace and GitHub

    Args:
        resource_dir: Base resource directory (e.g., ./resources)
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for downloading scFoundation. "
            "Install it with: pip install huggingface_hub"
        )

    output_dir = resource_dir / "scFoundation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("📥 Downloading scFoundation...")
    print(f"   Output: {output_dir}")

    # 1. Download model checkpoint from HuggingFace
    print("\n[1/2] Downloading models.ckpt from HuggingFace (genbio-ai/scFoundation)...")
    try:
        hf_hub_download(
            repo_id="genbio-ai/scFoundation",
            filename="model.ckpt",
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        # Rename to models.ckpt if downloaded as model.ckpt
        downloaded_ckpt = output_dir / "model.ckpt"
        target_ckpt = output_dir / "models.ckpt"
        if downloaded_ckpt.exists() and not target_ckpt.exists():
            downloaded_ckpt.rename(target_ckpt)
            print("   ✓ models.ckpt downloaded")
    except Exception as e:
        print(f"   ⚠️  HuggingFace download failed, trying alternative...")
        # Fallback: try 'models.ckpt' directly
        try:
            hf_hub_download(
                repo_id="genbio-ai/scFoundation",
                filename="models.ckpt",
                local_dir=str(output_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            print("   ✓ models.ckpt downloaded")
        except Exception as e2:
            print(f"   ❌ Failed to download models.ckpt: {e2}")

    # 2. Download TSV file from GitHub
    print("\n[2/2] Downloading OS_scRNA_gene_index.19264.tsv from GitHub...")
    github_url = "https://raw.githubusercontent.com/biomap-research/scFoundation/main/OS_scRNA_gene_index.19264.tsv"
    tsv_path = output_dir / "OS_scRNA_gene_index.19264.tsv"

    if tsv_path.exists():
        print(f"   ✓ File already exists: {tsv_path.name}")
    else:
        try:
            response = requests.get(github_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc="   TSV file")
            with open(tsv_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    f.write(data)
            progress_bar.close()
            
            print("   ✓ OS_scRNA_gene_index.19264.tsv downloaded")
        except Exception as e:
            print(f"   ❌ Failed to download TSV file: {e}")
            print(f"   You can manually download from: {github_url}")

    print("\n✅ scFoundation download completed!")


def _find_foundations_geneformer(resource_dir: Path) -> Path | None:
    """Try to locate Foundations/Geneformer relative to resource_dir"""
    # resource_dir is typically <repo>/resources
    # Foundations/Geneformer is at <repo>/Foundations/Geneformer
    repo_root = resource_dir.parent
    candidates = [
        repo_root / "Foundations" / "Geneformer",
        repo_root.parent / "Foundations" / "Geneformer",
    ]
    for p in candidates:
        if (p / "geneformer").is_dir() and (p / "Geneformer-V2-104M").is_dir():
            return p
    return None


def download_geneformer(resource_dir: Path) -> None:
    """
    Download Geneformer model weights and gene dictionaries.

    Strategy:
      1. If Foundations/Geneformer exists locally, copy resource files from there.
      2. Otherwise, fall back to HuggingFace individual file downloads.

    Copies all 4 model variants + 104M gene dicts + 30M gene dicts.

    Args:
        resource_dir: Base resource directory (e.g., ./resources)
    """
    import shutil

    output_dir = resource_dir / "Geneformer"
    output_dir.mkdir(parents=True, exist_ok=True)

    foundations_dir = _find_foundations_geneformer(resource_dir)

    if foundations_dir is not None:
        _copy_geneformer_from_foundations(foundations_dir, output_dir)
    else:
        _download_geneformer_from_hf(output_dir)

    print("\n✅ Geneformer download completed!")


def _copy_geneformer_from_foundations(src: Path, dst: Path) -> None:
    """Copy Geneformer resource files from Foundations/ directory"""
    import shutil

    print(f"📥 Copying Geneformer from local Foundations: {src}")
    print(f"   Output: {dst}")

    # 1. Copy model variants
    variants = [
        "Geneformer-V1-10M",
        "Geneformer-V2-104M",
        "Geneformer-V2-104M_CLcancer",
        "Geneformer-V2-316M",
    ]
    for variant in variants:
        src_dir = src / variant
        dst_dir = dst / variant
        if dst_dir.exists():
            print(f"   ✓ Already exists: {variant}/")
            continue
        if src_dir.is_dir():
            shutil.copytree(src_dir, dst_dir)
            print(f"   ✓ Copied {variant}/")
        else:
            print(f"   ⚠️  Not found: {src_dir}")

    # 2. Copy 104M gene dictionaries (from geneformer/ subdir to root)
    print("\n[Gene Dicts] Copying 104M gene dictionaries...")
    gene_dict_104m = [
        "gene_median_dictionary_gc104M.pkl",
        "token_dictionary_gc104M.pkl",
        "gene_name_id_dict_gc104M.pkl",
        "ensembl_mapping_dict_gc104M.pkl",
    ]
    for fname in gene_dict_104m:
        src_file = src / "geneformer" / fname
        dst_file = dst / fname
        if dst_file.exists():
            print(f"   ✓ Already exists: {fname}")
            continue
        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            print(f"   ✓ Copied {fname}")
        else:
            print(f"   ⚠️  Not found: {src_file}")

    # 3. Copy 30M gene dictionaries
    print("\n[Gene Dicts] Copying 30M gene dictionaries...")
    src_30m = src / "geneformer" / "gene_dictionaries_30m"
    dst_30m = dst / "gene_dictionaries_30m"
    if dst_30m.exists():
        print("   ✓ Already exists: gene_dictionaries_30m/")
    elif src_30m.is_dir():
        shutil.copytree(src_30m, dst_30m)
        print("   ✓ Copied gene_dictionaries_30m/")
    else:
        print(f"   ⚠️  Not found: {src_30m}")


def _download_geneformer_from_hf(output_dir: Path) -> None:
    """Fallback: download from HuggingFace when Foundations/ is not available"""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for downloading Geneformer. "
            "Install it with: pip install huggingface_hub"
        )

    repo_id = "ctheodoris/Geneformer"

    print("📥 Downloading Geneformer from HuggingFace (ctheodoris/Geneformer)...")
    print(f"   Output: {output_dir}")
    print("   ⚠️  Foundations/Geneformer not found locally, using HF (large download).")

    # 1. Download model variants
    variants = [
        "Geneformer-V1-10M",
        "Geneformer-V2-104M",
        "Geneformer-V2-104M_CLcancer",
        "Geneformer-V2-316M",
    ]
    model_files = ["config.json", "model.safetensors"]

    for variant in variants:
        variant_dir = output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[Model] Downloading {variant}...")

        for fname in model_files:
            target = variant_dir / fname
            if target.exists():
                print(f"   ✓ Already exists: {variant}/{fname}")
                continue
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=f"{variant}/{fname}",
                    local_dir=str(output_dir),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                print(f"   ✓ {variant}/{fname} downloaded")
            except Exception as e:
                print(f"   ❌ Failed to download {variant}/{fname}: {e}")

    # 2. Download 104M gene dictionaries
    print("\n[Gene Dicts] Downloading gene dictionaries (gc104M)...")
    gene_dict_files = [
        "geneformer/gene_median_dictionary_gc104M.pkl",
        "geneformer/token_dictionary_gc104M.pkl",
        "geneformer/gene_name_id_dict_gc104M.pkl",
        "geneformer/ensembl_mapping_dict_gc104M.pkl",
    ]

    for hf_path in gene_dict_files:
        fname = Path(hf_path).name
        target = output_dir / fname
        if target.exists():
            print(f"   ✓ Already exists: {fname}")
            continue
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=hf_path,
                local_dir=str(output_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            # Move from geneformer/ subdirectory to output_dir root
            downloaded = output_dir / hf_path
            if downloaded.exists() and not target.exists():
                downloaded.rename(target)
            print(f"   ✓ {fname} downloaded")
        except Exception as e:
            print(f"   ❌ Failed to download {fname}: {e}")

    # 3. Download 30M gene dictionaries
    print("\n[Gene Dicts] Downloading gene dictionaries (gc30M)...")
    gene_dict_30m_files = [
        "geneformer/gene_dictionaries_30m/gene_median_dictionary_gc30M.pkl",
        "geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl",
        "geneformer/gene_dictionaries_30m/gene_name_id_dict_gc30M.pkl",
        "geneformer/gene_dictionaries_30m/ensembl_mapping_dict_gc30M.pkl",
    ]

    dst_30m = output_dir / "gene_dictionaries_30m"
    dst_30m.mkdir(parents=True, exist_ok=True)

    for hf_path in gene_dict_30m_files:
        fname = Path(hf_path).name
        target = dst_30m / fname
        if target.exists():
            print(f"   ✓ Already exists: {fname}")
            continue
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=hf_path,
                local_dir=str(output_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            downloaded = output_dir / hf_path
            if downloaded.exists() and not target.exists():
                downloaded.rename(target)
            print(f"   ✓ {fname} downloaded")
        except Exception as e:
            print(f"   ❌ Failed to download {fname}: {e}")


def download_nicheformer(resource_dir: Path) -> None:
    """
    Download Nicheformer model from HuggingFace and technology means from GitHub

    Args:
        resource_dir: Base resource directory (e.g., ./resources)
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for downloading Nicheformer. "
            "Install it with: pip install huggingface_hub"
        )

    output_dir = resource_dir / "Nicheformer"
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download essential model files from HuggingFace
    repo_id = "theislab/nicheformer"
    essential_files = [
        "model.h5ad",
        "model.safetensors",
        "config.json",
        "configuration_nicheformer.py",
        "modeling_nicheformer.py",
        "tokenization_nicheformer.py",
    ]

    print("📥 Downloading Nicheformer from HuggingFace (theislab/nicheformer)...")
    print(f"   Output: {model_dir}")

    for fname in essential_files:
        target = model_dir / fname
        if target.exists():
            print(f"   ✓ Already exists: {fname}")
            continue
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            print(f"   ✓ {fname} downloaded")
        except Exception as e:
            print(f"   ❌ Failed to download {fname}: {e}")
            raise

    # 2. Download technology mean files from GitHub
    print("\n📥 Downloading technology mean files from GitHub...")
    means_dir = output_dir / "model_means"
    means_dir.mkdir(parents=True, exist_ok=True)

    github_base = (
        "https://raw.githubusercontent.com/theislab/nicheformer/main/data/model_means/"
    )
    mean_files = [
        "dissociated_mean_script.npy",
        "merfish_mean_script.npy",
        "cosmx_mean_script.npy",
        "xenium_mean_script.npy",
        "iss_mean_script.npy",
    ]

    for fname in mean_files:
        fpath = means_dir / fname
        if fpath.exists():
            print(f"   ✓ Already exists: {fname}")
            continue
        try:
            response = requests.get(github_base + fname, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            progress_bar = tqdm(
                total=total_size, unit="iB", unit_scale=True, desc=f"   {fname}"
            )
            with open(fpath, "wb") as f:
                for data in response.iter_content(1024):
                    progress_bar.update(len(data))
                    f.write(data)
            progress_bar.close()
            print(f"   ✓ {fname} downloaded")
        except Exception as e:
            print(f"   ⚠️  Failed to download {fname}: {e}")
            print(f"      You can manually place it in: {means_dir}")

    print("\n✅ Nicheformer download completed!")


def download_model(
    model_name: Literal["scGPT", "UCE", "scFoundation", "Geneformer", "Nicheformer", "all"],
    resource_dir: str | Path = "./resources",
) -> None:
    """
    Download model weights and necessary files

    Args:
        model_name: Model to download. Options: 'scGPT', 'UCE', 'scFoundation', 'Geneformer', 'Nicheformer', 'all'
        resource_dir: Directory to save downloaded files (default: './resources')

    Examples:
        >>> from scunify.config import download_model
        >>> download_model("scGPT", "./resources")
        >>> download_model("all", "./resources")  # Download all models

    Raises:
        ValueError: If invalid model_name is provided
        ImportError: If required dependencies are not installed
    """
    resource_path = Path(resource_dir).expanduser().resolve()
    resource_path.mkdir(parents=True, exist_ok=True)

    downloaders = {
        "scGPT": download_scgpt,
        "UCE": download_uce,
        "scFoundation": download_scfoundation,
        "Geneformer": download_geneformer,
        "Nicheformer": download_nicheformer,
    }

    if model_name == "all":
        print("📦 Downloading all models...\n")
        for name, func in downloaders.items():
            try:
                print(f"\n{'='*60}")
                print(f"📥 Downloading {name}...")
                print(f"{'='*60}")
                func(resource_path)
            except Exception as e:
                print(f"❌ Failed to download {name}: {e}")
                print(f"   Skipping {name}...\n")
                continue
        print("\n✅ All downloads completed!")

    elif model_name in downloaders:
        downloaders[model_name](resource_path)
    else:
        raise ValueError(
            f"Invalid model_name: {model_name}. "
            f"Choose from: {list(downloaders.keys())} or 'all'"
        )


if __name__ == "__main__":
    # CLI interface for testing
    import sys

    if len(sys.argv) < 2:
        print("Usage: python _download.py <model_name> [resource_dir]")
        print("  model_name: scGPT, UCE, scFoundation, Geneformer, Nicheformer, or all")
        print("  resource_dir: optional, default='./resources'")
        sys.exit(1)

    model = sys.argv[1]
    resource = sys.argv[2] if len(sys.argv) > 2 else "./resources"

    download_model(model, resource)
