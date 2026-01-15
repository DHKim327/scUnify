"""Setup utilities for scUnify models"""

import subprocess
import warnings
from pathlib import Path

from ._validators import RESOURCES_LISTS, validate_resources
from ._config_creators import CONFIG_CREATORS


# Conda 환경 YAML 파일 경로  
_ENV_DIR = Path(__file__).resolve().parent / "envs"


def setup(
    resource_dir: str, 
    config_dir: str, 
    auto_download: bool = False,
    create_conda_envs: bool = False,
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
    
    Examples:
        >>> # Basic setup (validation only)
        >>> setup("./resources", "./configs")
        
        >>> # Setup with auto-download
        >>> setup("./resources", "./configs", auto_download=True)
        
        >>> # Setup + pre-cache Ray environments (recommended!)
    
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

    # ========== Conda 환경 생성 ==========
    if create_conda_envs and configured_models:
        _create_system_conda_envs(configured_models)

    # ========== 최종 요약 ==========
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
    
    print()


def _create_system_conda_envs(models: list[str]):
    """
    시스템 conda에 모델별 환경 생성
    
    Args:
        models: 환경을 생성할 모델 리스트 (예: ["scGPT", "UCE", "scFoundation"])
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
        
        # conda env list로 이미 존재하는지 확인
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
        
        # 환경 생성
        try:
            print(f"    📦 Installing packages (this may take 10-15 minutes)...")
            subprocess.run([
                "conda", "env", "create",
                "-f", str(yaml_file),
                "-n", env_name,
            ], check=True)
            
            print(f"    ✅ {env_name} created successfully!")
            
            # scunify[core] editable 설치 (YAML이 아닌 로컬에서!)
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
