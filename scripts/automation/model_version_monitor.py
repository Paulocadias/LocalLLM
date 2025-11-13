"""
Model Version Monitor
Automatically detects new Qwen model releases and triggers fine-tuning
Runs every 6 hours via Task Scheduler
"""

import requests
import json
import os
from datetime import datetime
from pathlib import Path

# Configuration
HUGGINGFACE_API = "https://huggingface.co/api/models"
OLLAMA_API = "http://localhost:11434/api/tags"
TRACKED_MODELS = [
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen2.5-Coder-14B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
]

VERSION_FILE = Path("model_versions.json")
TRIGGER_FILE = Path("trigger_training.flag")

def load_known_versions():
    """Load known model versions"""
    if VERSION_FILE.exists():
        with open(VERSION_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_known_versions(versions):
    """Save model versions"""
    with open(VERSION_FILE, 'w') as f:
        json.dump(versions, indent=2, fp=f)

def check_huggingface_version(model_name):
    """Check Hugging Face for model updates"""
    try:
        url = f"{HUGGINGFACE_API}/{model_name}"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            # Get last modified date as version indicator
            last_modified = data.get('lastModified', '')
            sha = data.get('sha', '')
            return {
                'last_modified': last_modified,
                'sha': sha,
                'source': 'huggingface'
            }
    except Exception as e:
        print(f"  Error checking HuggingFace: {e}")

    return None

def check_ollama_version(model_name):
    """Check Ollama for model updates"""
    try:
        response = requests.get(OLLAMA_API, timeout=5)

        if response.status_code == 200:
            models = response.json().get('models', [])

            # Match model name (e.g., "qwen2.5-coder:7b")
            ollama_name = model_name.split('/')[-1].lower().replace('-instruct', '')

            for model in models:
                if ollama_name in model['name'].lower():
                    return {
                        'digest': model.get('digest', ''),
                        'modified_at': model.get('modified_at', ''),
                        'source': 'ollama'
                    }
    except Exception as e:
        print(f"  Error checking Ollama: {e}")

    return None

def detect_new_version(model_name, known_versions):
    """Check if a new version is available"""
    print(f"\nChecking: {model_name}")

    # Check HuggingFace
    hf_version = check_huggingface_version(model_name)

    # Check Ollama
    ollama_version = check_ollama_version(model_name)

    # Compare with known versions
    known = known_versions.get(model_name, {})

    new_version_detected = False

    if hf_version:
        known_sha = known.get('huggingface', {}).get('sha', '')
        if hf_version['sha'] != known_sha:
            print(f"  ✓ NEW VERSION on HuggingFace!")
            print(f"    Old SHA: {known_sha[:12]}...")
            print(f"    New SHA: {hf_version['sha'][:12]}...")
            new_version_detected = True
        else:
            print(f"  No change on HuggingFace")

    if ollama_version:
        known_digest = known.get('ollama', {}).get('digest', '')
        if ollama_version['digest'] != known_digest:
            print(f"  ✓ NEW VERSION on Ollama!")
            print(f"    Old: {known_digest[:12]}...")
            print(f"    New: {ollama_version['digest'][:12]}...")
            new_version_detected = True
        else:
            print(f"  No change on Ollama")

    return new_version_detected, hf_version, ollama_version

def trigger_training():
    """Create trigger file to start fine-tuning"""
    trigger_data = {
        'timestamp': datetime.now().isoformat(),
        'reason': 'new_model_version_detected',
        'message': 'New model version detected! Fine-tuning should start with ALL accumulated data.'
    }

    with open(TRIGGER_FILE, 'w') as f:
        json.dump(trigger_data, indent=2, fp=f)

    print(f"\n{'='*80}")
    print("TRAINING TRIGGER CREATED")
    print(f"{'='*80}")
    print(f"File: {TRIGGER_FILE}")
    print("Next: Run monthly_unsloth_automation.py to start training")

def send_notification(model_name):
    """Send notification about new version (placeholder)"""
    notification = f"""
{'='*80}
NEW MODEL VERSION DETECTED
{'='*80}

Model: {model_name}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Action Required:
1. New model version available
2. Training trigger created
3. Fine-tune with ALL accumulated history
4. Deploy and test new model

The system will use all reflexion data, conversation feedback,
and distillation data accumulated so far!
"""

    print(notification)

    # Save notification
    notifications_dir = Path("notifications")
    notifications_dir.mkdir(exist_ok=True)

    notification_file = notifications_dir / f"new_version_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(notification_file, 'w') as f:
        f.write(notification)

    print(f"Notification saved to: {notification_file}")

def main():
    """Main monitoring workflow"""
    print("="*80)
    print("MODEL VERSION MONITOR")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tracked models: {len(TRACKED_MODELS)}")
    print()

    # Load known versions
    known_versions = load_known_versions()

    # Check each model
    updates_found = False
    updated_models = []

    for model_name in TRACKED_MODELS:
        new_version, hf_version, ollama_version = detect_new_version(model_name, known_versions)

        if new_version:
            updates_found = True
            updated_models.append(model_name)

            # Update known versions
            if model_name not in known_versions:
                known_versions[model_name] = {}

            if hf_version:
                known_versions[model_name]['huggingface'] = hf_version

            if ollama_version:
                known_versions[model_name]['ollama'] = ollama_version

            known_versions[model_name]['last_checked'] = datetime.now().isoformat()

    # Save updated versions
    save_known_versions(known_versions)

    # Handle updates
    if updates_found:
        print(f"\n{'='*80}")
        print(f"FOUND {len(updated_models)} MODEL UPDATE(S)")
        print(f"{'='*80}")

        for model in updated_models:
            print(f"  - {model}")

        # Trigger training
        trigger_training()

        # Send notifications
        for model in updated_models:
            send_notification(model)

        print("\n✓ Training trigger created!")
        print("✓ Notifications sent!")

    else:
        print(f"\n{'='*80}")
        print("NO UPDATES FOUND")
        print(f"{'='*80}")
        print("All models are up to date.")
        print(f"Next check in 6 hours...")

    print(f"\nVersion data saved to: {VERSION_FILE}")

if __name__ == "__main__":
    main()
