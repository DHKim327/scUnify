"""Setup utilities for scUnify models"""

import subprocess
import warnings
from pathlib import Path

from ._validators import RESOURCES_LISTS, validate_resources
from ._config_creators import CONFIG_CREATORS


# Path to conda environment YAML files  
_ENV_DIR = Path(__file__).resolve().parent / "envs"


def setup(
    resource_dir: str,
    config_dir: str,
    auto_download: bool = False,
    create_conda_envs: bool = False,
    perturb: bool = False,
):
    """
    Setup configuration files for scUnify models

    This function validates model resources, creates configuration files,
    and optionally creates conda environments for each model.

    Args:
        resource_dir: Directory containing model weights and resources
        config_dir: Directory to save generated configuration files
        auto_download: If True, automatically download missing models (default: False)
        create_conda_envs: If True, create system conda environments (default: False)
                           This will create: scunify_scgpt, scunify_scfoundation, scunify_uce
                           Takes about 45 minutes on first run.
        perturb: If True, additionally create the ``scunify_perturb`` conda env
                 used by perturbation tasks (torch_geometric + GEARS deps shared
                 across scGPT and scFoundation perturbation mixins). Independent
                 of ``create_conda_envs`` — pass ``perturb=True`` even if model
                 envs are already created.

    Examples:
        >>> # Basic setup (validation only)
        >>> setup("./resources", "./configs")

        >>> # Setup with auto-download
        >>> setup("./resources", "./configs", auto_download=True)

        >>> # Setup + pre-cache Ray environments (recommended!)
        >>> setup("./resources", "./configs", create_conda_envs=True)

        >>> # Add the perturbation env on top of existing model envs
        >>> setup("./resources", "./configs", perturb=True)

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
    print(f"Create Conda Envs: {create_conda_envs}")
    print(f"{'='*60}\n")

    success_count = 0
    failed_models = []
    configured_models = []

    for model_name in RESOURCES_LISTS.keys():
        try:
            print(f"\n[{model_name}] Starting validation...")
            validate_resources(resource_path, model_name)
            
            print(f"[{model_name}] Creating configuration files...")
            config_creator = CONFIG_CREATORS[model_name]
            config_creator(resource_path, config_dir)
            success_count += 1
            configured_models.append(model_name)

        except (FileNotFoundError, ValueError) as e:
            if auto_download:
                print(f"\n⚠️  [{model_name}] Resources missing. Attempting auto-download...")
                try:
                    from ._download import download_model
                    download_model(model_name, resource_path)
                    
                    # Re-validate after download
                    print(f"\n[{model_name}] Re-validating after download...")
                    validate_resources(resource_path, model_name)
                    
                    print(f"[{model_name}] Creating configuration files...")
                    config_creator = CONFIG_CREATORS[model_name]
                    config_creator(resource_path, config_dir)
                    success_count += 1
                    configured_models.append(model_name)
                    
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

    # ========== Create conda environments ==========
    if create_conda_envs and configured_models:
        _create_system_conda_envs(configured_models)

    if perturb:
        _create_perturb_conda_env()

    # ========== Final summary ==========
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
    
    if create_conda_envs:
        print(f"\n💡 Conda environments created!")
        print(f"   ScUnifyRunner will use these environments automatically.")
        print(f"   Verify with: conda env list")
    else:
        print(f"\n💡 To create conda environments for faster startup:")
        print(f"   >>> from scunify.config import setup")
        print(f"   >>> setup('{resource_dir}', '{config_dir}', create_conda_envs=True)")

    if perturb:
        print(f"\n💡 GEARS perturbation deps added to existing model envs")
        print(f"   Default mapping continues to work (model_name → scunify_<model>);")
        print(f"   no yaml ``env:`` override needed for perturbation tasks.")

    print()


# Lightweight extras pinned for ABI safety. ``--no-deps`` on torch_geometric
# is critical: PyPI's torch_geometric (no torch version pin) otherwise pulls
# the latest torch and breaks the conda-installed torch ABI.
_PERTURB_GEARS_PKGS = ["dcor", "networkx>=2.8", "omegaconf"]
_PERTURB_PYG_PKG = "torch_geometric==2.5.0"
_PERTURB_BACKBONES = ("scgpt", "scfoundation")


def _create_perturb_conda_env():
    """Add GEARS perturbation deps (torch_geometric + dcor + networkx) to
    each existing model env (``scunify_scgpt``, ``scunify_scfoundation``).

    Why add to existing envs instead of creating dedicated perturb envs:
    scGPT (torch 2.3 + torchtext 0.18) and scFoundation (torch 2.11 + cu128 +
    local-attention) have mutually incompatible ABIs — a single shared env is
    impossible. The cheapest path is to layer the small GEARS deps onto each
    paper-faithful backbone env. No yaml ``env:`` override is needed.

    ``torch_geometric`` is installed with ``--no-deps`` so its unpinned ``torch``
    requirement does not silently upgrade the conda-installed torch.
    """
    print(f"\n{'='*60}")
    print(f"🔧 Adding GEARS perturbation deps to model envs")
    print(f"{'='*60}")

    for backbone in _PERTURB_BACKBONES:
        env_name = f"scunify_{backbone}"

        # Skip if env doesn't exist (model not configured for this user)
        try:
            result = subprocess.run(
                ["conda", "env", "list"],
                capture_output=True, text=True, check=False,
            )
            if env_name not in result.stdout:
                print(f"⚠️  {env_name} not found — skip. Run setup(create_conda_envs=True) first.")
                continue
        except Exception as e:
            print(f"⚠️  Could not check env list: {e}")
            continue

        print(f"\n📦 [{env_name}] installing GEARS deps...")
        try:
            # 1) torch_geometric WITH --no-deps so it cannot upgrade torch.
            subprocess.run([
                "conda", "run", "-n", env_name,
                "pip", "install", "--no-deps", _PERTURB_PYG_PKG,
            ], check=True)
            # 2) Pure-python helpers (torch-agnostic).
            subprocess.run([
                "conda", "run", "-n", env_name,
                "pip", "install", *_PERTURB_GEARS_PKGS,
            ], check=True)
            print(f"    ✅ {env_name} ready for perturbation tasks")
        except subprocess.CalledProcessError as e:
            print(f"    ❌ failed: {e}")
        except Exception as e:
            print(f"    ❌ unexpected: {e}")


def _create_system_conda_envs(models: list[str]):
    """
    Create per-model conda environments via system conda.
    
    Args:
        models: List of models to create envs for (e.g., ["scGPT", "UCE", "scFoundation"])
    """
    print(f"\n{'='*60}")
    print(f"🔧 Creating System Conda Environments")
    print(f"{'='*60}")
    print(f"⏱️  This will take about 45 minutes (first time only)")
    print(f"   Models: {', '.join(models)}")
    print()
    
    for i, model_name in enumerate(models, 1):
        yaml_file = _ENV_DIR / f"{model_name.lower()}_env.yaml"
        env_name = f"scunify_{model_name.lower()}"
        
        if not yaml_file.exists():
            print(f"[{i}/{len(models)}] ⚠️  YAML not found: {yaml_file}")
            continue
        
        print(f"[{i}/{len(models)}] Creating {env_name}...")
        
        # Check if the env already exists via conda env list
        try:
            result = subprocess.run(
                ["conda", "env", "list"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if env_name in result.stdout:
                print(f"    ⚠️  {env_name} already exists. Skipping.")
                continue
        except Exception as e:
            print(f"    ⚠️  Could not check existing envs: {e}")
        
        # Create the environment
        try:
            print(f"    📦 Installing packages (this may take 10-15 minutes)...")
            subprocess.run([
                "conda", "env", "create",
                "-f", str(yaml_file),
                "-n", env_name,
            ], check=True)
            
            print(f"    ✅ {env_name} created successfully!")
            
            # Install scunify[core] in editable mode (local, not from YAML)
            print(f"    🔗 Installing scunify[core] in editable mode...")
            project_root = _ENV_DIR.parent.parent.parent.parent  # Github/scUnify
            subprocess.run([
                "conda", "run", "-n", env_name,
                "pip", "install", "-e", f"{project_root}[core]"
            ], check=True)
            
            print(f"    ✅ scunify[core] installed in editable mode!")

        except subprocess.CalledProcessError as e:
            print(f"    ❌ Failed to create {env_name}: {e}")
        except Exception as e:
            print(f"    ❌ Unexpected error: {e}")
    
    print(f"\n{'='*60}")
    print(f"✅ Conda Environment Setup Completed!")
    print(f"{'='*60}")
    print(f"   Created environments:")
    for model_name in models:
        print(f"   - scunify_{model_name.lower()}")
    print(f"\n   📋 Verify with: conda env list")
    print()
