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
    
    ⚠️  NOTE: Figshare has blocked automated downloads. Please download manually:
    1. Visit: https://figshare.com/articles/dataset/Universal_Cell_Embedding_Model_Files/24320806
    2. Click "Download all" to download the files
    3. Extract to the UCE directory

    Args:
        resource_dir: Base resource directory (e.g., ./resources)
    """
    output_dir = resource_dir / "UCE"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("⚠️  UCE Download Notice")
    print("="*70)
    print("Figshare has blocked automated script downloads (403 Forbidden).")
    print("Please download UCE files manually:\n")
    print("1. Visit: https://figshare.com/articles/dataset/24320806")
    print(f"2. Download all files (Total: ~14 GB)")
    print(f"3. Place files in: {output_dir}\n")
    print("Required files:")
    required_files = [
        'species_offsets.pkl',
        'species_chrom.csv',
        '4layer_model.torch',
        'all_tokens.torch',
        'protein_embeddings.tar.gz',
        '33l_8ep_1024t_1280.torch'
    ]
    for f in required_files:
        print(f"   - {f}")
    print("\nAfter downloading, extract protein_embeddings.tar.gz if needed.")
    print("="*70)
    
    raise RuntimeError(
        "UCE automatic download is currently blocked by Figshare. "
        "Please download manually from: "
        "https://figshare.com/articles/dataset/24320806"
    )


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


def download_model(
    model_name: Literal["scGPT", "UCE", "scFoundation", "all"],
    resource_dir: str | Path = "./resources",
) -> None:
    """
    Download model weights and necessary files

    Args:
        model_name: Model to download. Options: 'scGPT', 'UCE', 'scFoundation', 'all'
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
        print("  model_name: scGPT, UCE, scFoundation, or all")
        print("  resource_dir: optional, default='./resources'")
        sys.exit(1)

    model = sys.argv[1]
    resource = sys.argv[2] if len(sys.argv) > 2 else "./resources"

    download_model(model, resource)
