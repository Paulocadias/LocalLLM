"""
Automatic LoRA Model Deployment
Watches for new models and deploys them automatically
"""

import os
import time
import subprocess
import json
from pathlib import Path
from datetime import datetime

# Configuration
MODELS_DOWNLOAD_DIR = Path("C:/BOT/localLLM/lora_models_download")
DEPLOYED_MODELS_FILE = Path("C:/BOT/localLLM/deployed_models.json")
OLLAMA_BASE_MODEL = "qwen2.5-coder:7b"

PROFILES = [
    "backend",
    "frontend",
    "android_mobile",
    "bug_fixing",
    "refactoring",
    "documentation"
]

def load_deployed_models():
    """Load record of deployed models"""
    if DEPLOYED_MODELS_FILE.exists():
        with open(DEPLOYED_MODELS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_deployed_models(deployed):
    """Save record of deployed models"""
    with open(DEPLOYED_MODELS_FILE, 'w') as f:
        json.dump(deployed, indent=2, fp=f)

def get_model_hash(model_dir):
    """Get hash of model directory to detect changes"""
    import hashlib

    hash_obj = hashlib.md5()

    # Hash all files in directory
    for file_path in sorted(model_dir.rglob("*")):
        if file_path.is_file():
            hash_obj.update(str(file_path.stat().st_mtime).encode())
            hash_obj.update(str(file_path.stat().st_size).encode())

    return hash_obj.hexdigest()

def create_modelfile(profile, lora_adapter_path):
    """Create Ollama Modelfile with LoRA adapter"""

    modelfile_content = f"""FROM {OLLAMA_BASE_MODEL}

# LoRA adapter for {profile}
ADAPTER {lora_adapter_path}

# System prompt
SYSTEM You are a specialized AI assistant for {profile} tasks.

# Parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
"""

    modelfile_path = Path(f"Modelfile_{profile}")
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)

    return modelfile_path

def deploy_model(profile, model_dir):
    """Deploy a LoRA model to Ollama"""

    print(f"\n{'='*80}")
    print(f"DEPLOYING MODEL: {profile}")
    print(f"{'='*80}\n")

    # Find adapter file
    adapter_file = None
    for ext in ["*.safetensors", "*.bin", "adapter_model.bin"]:
        matches = list(model_dir.glob(ext))
        if matches:
            adapter_file = matches[0]
            break

    if not adapter_file:
        print(f"ERROR: No adapter file found in {model_dir}")
        return False

    print(f"Adapter file: {adapter_file}")

    # Create Modelfile
    modelfile = create_modelfile(profile, str(adapter_file))
    print(f"Modelfile created: {modelfile}")

    # Create Ollama model
    model_name = f"qwen-coder-{profile}-lora"

    print(f"\nCreating Ollama model: {model_name}")
    print("This may take a few minutes...")

    try:
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile)],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            print(f"\n✓ SUCCESS! Model deployed: {model_name}")
            print(f"\nYou can now use it with:")
            print(f"  ollama run {model_name}")

            # Clean up Modelfile
            modelfile.unlink()

            return True
        else:
            print(f"\nERROR deploying model:")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print("\nERROR: Deployment timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"\nERROR: {e}")
        return False

def check_for_new_models():
    """Check for new or updated models"""

    print(f"\n{'='*80}")
    print(f"CHECKING FOR NEW MODELS")
    print(f"{'='*80}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Download directory: {MODELS_DOWNLOAD_DIR}")

    if not MODELS_DOWNLOAD_DIR.exists():
        print(f"\nDownload directory not found. Creating...")
        MODELS_DOWNLOAD_DIR.mkdir(parents=True)
        return

    deployed = load_deployed_models()
    newly_deployed = []

    # Check each profile
    for profile in PROFILES:
        # Find latest model directory for this profile
        pattern = f"{profile}_*"
        model_dirs = sorted(MODELS_DOWNLOAD_DIR.glob(pattern))

        if not model_dirs:
            print(f"\n[{profile}] No model found yet")
            continue

        # Get most recent model
        latest_model = model_dirs[-1]
        model_hash = get_model_hash(latest_model)

        # Check if already deployed
        if profile in deployed:
            if deployed[profile].get("hash") == model_hash:
                print(f"\n[{profile}] Already deployed (up to date)")
                continue

        # New or updated model found!
        print(f"\n[{profile}] NEW MODEL DETECTED!")
        print(f"  Path: {latest_model}")

        # Deploy it
        if deploy_model(profile, latest_model):
            deployed[profile] = {
                "path": str(latest_model),
                "hash": model_hash,
                "deployed_at": datetime.now().isoformat()
            }
            newly_deployed.append(profile)

    # Save deployment record
    if newly_deployed:
        save_deployed_models(deployed)

        print(f"\n{'='*80}")
        print(f"DEPLOYMENT COMPLETE")
        print(f"{'='*80}")
        print(f"\nNewly deployed models: {len(newly_deployed)}")
        for profile in newly_deployed:
            print(f"  ✓ {profile}")

        print(f"\nAll models updated and ready to use!")
    else:
        print(f"\nNo new models to deploy.")

def main():
    """Main monitoring loop"""

    print("="*80)
    print("AUTOMATIC LORA MODEL DEPLOYMENT")
    print("="*80)
    print(f"\nMonitoring directory: {MODELS_DOWNLOAD_DIR}")
    print(f"Checking every 30 minutes for new models...\n")

    while True:
        try:
            check_for_new_models()
        except Exception as e:
            print(f"\nERROR: {e}")

        # Wait 30 minutes before checking again
        print(f"\nNext check in 30 minutes...")
        time.sleep(1800)  # 30 minutes

if __name__ == "__main__":
    main()
