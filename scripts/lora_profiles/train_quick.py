"""Quick LoRA training script with qwen2.5-coder-7b"""
import subprocess
import sys

MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"  # 7B - matches qwen3:latest, faster training
EPOCHS = 2
BATCH_SIZE = 1  # Small batch for quick training

profiles = [
    ("android", "../../datasets/lora_profiles/android_mobile_starter.jsonl", "../../lora_adapters/android-mobile"),
    ("backend", "../../datasets/lora_profiles/backend_starter.jsonl", "../../lora_adapters/backend"),
    ("frontend", "../../datasets/lora_profiles/frontend_starter.jsonl", "../../lora_adapters/frontend"),
]

for profile, dataset, output in profiles:
    print(f"\n{'='*60}")
    print(f"Training {profile} profile...")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "fine_tune_profile_lora.py",
        "--profile", profile,
        "--dataset", dataset,
        "--output", output,
        "--model", MODEL,
        "--epochs", str(EPOCHS),
        "--batch-size", str(BATCH_SIZE)
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ {profile} training complete!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {profile} training failed: {e}")
        continue

print("\n" + "="*60)
print("All training complete!")
print("="*60)
