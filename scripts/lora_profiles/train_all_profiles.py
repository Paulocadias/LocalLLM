"""
Master Training Script - Train All LoRA Profiles
Automates dataset collection and LoRA training for all 6 profiles

Usage:
    python train_all_profiles.py --github-token YOUR_TOKEN --skip-collection  # Use existing datasets
    python train_all_profiles.py --github-token YOUR_TOKEN  # Collect + train
"""

import argparse
import subprocess
import sys
from pathlib import Path
import logging
from typing import List, Dict
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Profile configurations
PROFILES = {
    "android": {
        "dataset": "datasets/lora_profiles/android_mobile.jsonl",
        "output": "lora_adapters/android_mobile",
        "examples": 800,
        "epochs": 3
    },
    "backend": {
        "dataset": "datasets/lora_profiles/backend.jsonl",
        "output": "lora_adapters/backend",
        "examples": 800,
        "epochs": 3
    },
    "frontend": {
        "dataset": "datasets/lora_profiles/frontend.jsonl",
        "output": "lora_adapters/frontend",
        "examples": 800,
        "epochs": 3
    },
    "bug_fixing": {
        "dataset": "datasets/lora_profiles/bug_fixing.jsonl",
        "output": "lora_adapters/bug_fixing",
        "examples": 800,
        "epochs": 3
    },
    "refactoring": {
        "dataset": "datasets/lora_profiles/refactoring.jsonl",
        "output": "lora_adapters/refactoring",
        "examples": 800,
        "epochs": 3
    },
    "documentation": {
        "dataset": "datasets/lora_profiles/documentation.jsonl",
        "output": "lora_adapters/documentation",
        "examples": 800,
        "epochs": 3
    }
}


def run_command(cmd: List[str], description: str) -> bool:
    """Run command and log output"""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"✓ {description} completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed:")
        logger.error(f"  Error: {e.stderr}")
        return False


def collect_dataset(profile: str, github_token: str, count: int) -> bool:
    """Collect dataset for a profile"""
    config = PROFILES[profile]

    # Collect from GitHub
    cmd_github = [
        sys.executable,
        "scripts/lora_profiles/collect_github_datasets.py",
        "--profile", profile,
        "--output", f"datasets/lora_profiles/{profile}_github.jsonl",
        "--count", str(count // 2),
        "--token", github_token
    ]

    if not run_command(cmd_github, f"Collecting GitHub data for {profile}"):
        return False

    # Collect from Stack Overflow
    cmd_so = [
        sys.executable,
        "scripts/lora_profiles/collect_stackoverflow_datasets.py",
        "--profile", profile,
        "--output", f"datasets/lora_profiles/{profile}_stackoverflow.jsonl",
        "--count", str(count // 2)
    ]

    if not run_command(cmd_so, f"Collecting Stack Overflow data for {profile}"):
        logger.warning(f"Stack Overflow collection failed for {profile}, continuing...")

    # Merge datasets
    merge_datasets(profile, config['dataset'])

    return True


def merge_datasets(profile: str, output_path: str):
    """Merge GitHub and Stack Overflow datasets"""
    logger.info(f"Merging datasets for {profile}...")

    all_examples = []

    # Load GitHub dataset
    github_path = f"datasets/lora_profiles/{profile}_github.jsonl"
    if Path(github_path).exists():
        with open(github_path, 'r', encoding='utf-8') as f:
            for line in f:
                all_examples.append(json.loads(line))
        logger.info(f"  Loaded {len(all_examples)} GitHub examples")

    # Load Stack Overflow dataset
    so_path = f"datasets/lora_profiles/{profile}_stackoverflow.jsonl"
    if Path(so_path).exists():
        with open(so_path, 'r', encoding='utf-8') as f:
            for line in f:
                all_examples.append(json.loads(line))
        logger.info(f"  Total: {len(all_examples)} examples")

    # Save merged dataset
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    logger.info(f"✓ Merged dataset saved to {output_path}")


def train_profile(profile: str, model: str, batch_size: int) -> bool:
    """Train LoRA adapter for a profile"""
    config = PROFILES[profile]

    cmd = [
        sys.executable,
        "scripts/lora_profiles/fine_tune_profile_lora.py",
        "--profile", profile,
        "--dataset", config['dataset'],
        "--output", config['output'],
        "--model", model,
        "--epochs", str(config['epochs']),
        "--batch-size", str(batch_size)
    ]

    return run_command(cmd, f"Training LoRA for {profile}")


def create_ollama_modelfile(profile: str):
    """Create Ollama Modelfile for profile"""
    config = PROFILES[profile]
    modelfile_path = Path(config['output']) / "Modelfile"

    # System prompts per profile
    system_prompts = {
        "android": "You are an expert Android developer specializing in Kotlin and modern Android development with Jetpack Compose.",
        "backend": "You are an expert backend developer specializing in API design, database optimization, and server architecture.",
        "frontend": "You are an expert frontend developer specializing in React, TypeScript, and modern web development.",
        "bug_fixing": "You are an expert at debugging code and identifying root causes of errors with clear fix solutions.",
        "refactoring": "You are an expert at code refactoring and optimization while maintaining functionality.",
        "documentation": "You are an expert at writing clear, comprehensive code documentation."
    }

    modelfile_content = f"""FROM qwen3-coder:latest
ADAPTER ./{profile}.ggml
PARAMETER temperature 0.7
PARAMETER top_p 0.9
SYSTEM \"\"\"{system_prompts[profile]}\"\"\"
"""

    modelfile_path.parent.mkdir(parents=True, exist_ok=True)
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)

    logger.info(f"✓ Created Modelfile for {profile}")


def main():
    parser = argparse.ArgumentParser(description="Train all LoRA profiles")
    parser.add_argument("--github-token", type=str,
                        help="GitHub access token for dataset collection")
    parser.add_argument("--skip-collection", action="store_true",
                        help="Skip dataset collection, use existing datasets")
    parser.add_argument("--profiles", type=str, nargs='+',
                        choices=list(PROFILES.keys()),
                        default=list(PROFILES.keys()),
                        help="Profiles to train (default: all)")
    parser.add_argument("--model", type=str,
                        default="Qwen/Qwen2.5-Coder-32B-Instruct",
                        help="Base model name")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("LoRA Profile Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"Profiles to train: {', '.join(args.profiles)}")
    logger.info(f"Base model: {args.model}")
    logger.info("")

    # Track results
    results = {}

    for profile in args.profiles:
        logger.info("-" * 60)
        logger.info(f"Processing profile: {profile.upper()}")
        logger.info("-" * 60)

        config = PROFILES[profile]

        # Step 1: Dataset Collection
        if not args.skip_collection:
            if not args.github_token:
                logger.error("GitHub token required for collection. Use --github-token or --skip-collection")
                results[profile] = "FAILED - No GitHub token"
                continue

            logger.info(f"[1/4] Collecting dataset ({config['examples']} examples)...")
            if not collect_dataset(profile, args.github_token, config['examples']):
                results[profile] = "FAILED - Dataset collection"
                continue
        else:
            logger.info(f"[1/4] Skipping dataset collection (using existing)")
            if not Path(config['dataset']).exists():
                logger.error(f"Dataset not found: {config['dataset']}")
                results[profile] = "FAILED - Dataset not found"
                continue

        # Step 2: Fine-tune LoRA
        logger.info(f"[2/4] Fine-tuning LoRA adapter...")
        if not train_profile(profile, args.model, args.batch_size):
            results[profile] = "FAILED - Training"
            continue

        # Step 3: Create Ollama Modelfile
        logger.info(f"[3/4] Creating Ollama Modelfile...")
        create_ollama_modelfile(profile)

        # Step 4: Note conversion step (manual)
        logger.info(f"[4/4] LoRA adapter ready!")
        logger.info(f"  Next steps:")
        logger.info(f"    1. Convert LoRA to GGML format")
        logger.info(f"    2. Run: ollama create qwen-{profile} -f {config['output']}/Modelfile")
        logger.info("")

        results[profile] = "SUCCESS"

    # Print summary
    logger.info("=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    for profile, status in results.items():
        status_icon = "✓" if status == "SUCCESS" else "✗"
        logger.info(f"  {status_icon} {profile}: {status}")

    logger.info("")
    logger.info("Training pipeline complete!")

    # Return success if all profiles succeeded
    return all(status == "SUCCESS" for status in results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
