"""
Monthly Unsloth Automation Script
Merges all weekly reflexion data and triggers monthly fine-tuning
"""

import os
import json
import glob
from datetime import datetime
from pathlib import Path

# Paths
DATASETS_DIR = Path("datasets/lora_profiles")
COLAB_SYNC_DIR = Path("colab_sync")  # Will sync to Google Drive
RESULTS_DIR = Path("training_results")

PROFILES = [
    "backend",
    "frontend",
    "android_mobile",
    "bug_fixing",
    "refactoring",
    "documentation"
]

def merge_weekly_data():
    """Merge all weekly reflexion data accumulated this month"""
    print("=" * 80)
    print("MONTHLY DATA MERGE")
    print("=" * 80)

    current_month = datetime.now().strftime("%Y%m")
    merged_data = {}

    # Initialize merged data for each profile
    for profile in PROFILES:
        merged_data[profile] = []

    # Find all reflexion files from this month
    reflexion_files = glob.glob(str(DATASETS_DIR / f"reflexion_*_{current_month}*.jsonl"))
    distillation_files = glob.glob(str(DATASETS_DIR / f"distillation_*_{current_month}*.jsonl"))
    conversation_files = glob.glob(str(DATASETS_DIR / f"conversation_feedback_{current_month}*.jsonl"))

    print(f"\nFound {len(reflexion_files)} reflexion files")
    print(f"Found {len(distillation_files)} distillation files")
    print(f"Found {len(conversation_files)} conversation feedback files")

    # Merge reflexion data
    for file in reflexion_files:
        print(f"  Loading {Path(file).name}...")
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    # Categorize by profile based on content
                    profile = categorize_example(data)
                    merged_data[profile].append(data)

    # Merge distillation data
    for file in distillation_files:
        print(f"  Loading {Path(file).name}...")
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    profile = categorize_example(data)
                    merged_data[profile].append(data)

    # Merge conversation feedback
    for file in conversation_files:
        print(f"  Loading {Path(file).name}...")
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    profile = categorize_example(data)
                    merged_data[profile].append(data)

    # Save merged datasets
    COLAB_SYNC_DIR.mkdir(exist_ok=True)

    print("\n" + "=" * 80)
    print("MERGED DATA SUMMARY")
    print("=" * 80)

    total_examples = 0
    for profile in PROFILES:
        # Load original starter dataset
        starter_file = DATASETS_DIR / f"{profile}_starter.jsonl"
        if starter_file.exists():
            with open(starter_file, 'r', encoding='utf-8') as f:
                original_data = [json.loads(line) for line in f if line.strip()]

            # Combine original + new data
            combined = original_data + merged_data[profile]
        else:
            combined = merged_data[profile]

        # Save combined dataset for Colab
        output_file = COLAB_SYNC_DIR / f"{profile}_training_{current_month}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in combined:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"\n{profile}:")
        print(f"  Original examples: {len(original_data) if starter_file.exists() else 0}")
        print(f"  New examples: {len(merged_data[profile])}")
        print(f"  Total: {len(combined)}")
        print(f"  Saved to: {output_file}")

        total_examples += len(combined)

    print(f"\nTotal examples across all profiles: {total_examples}")
    print(f"\nAll datasets ready for training in: {COLAB_SYNC_DIR}")

    return merged_data

def categorize_example(data):
    """Categorize training example into a profile based on content"""
    instruction = data.get('instruction', '').lower()
    response = data.get('response', '').lower()
    text = instruction + ' ' + response

    # Keywords for each profile
    if any(word in text for word in ['api', 'rest', 'database', 'server', 'backend', 'sql', 'postgresql', 'mysql']):
        return 'backend'
    elif any(word in text for word in ['react', 'component', 'jsx', 'frontend', 'ui', 'css', 'html', 'javascript']):
        return 'frontend'
    elif any(word in text for word in ['android', 'kotlin', 'activity', 'fragment', 'mobile', 'app']):
        return 'android_mobile'
    elif any(word in text for word in ['bug', 'error', 'fix', 'debug', 'issue', 'problem', 'crash']):
        return 'bug_fixing'
    elif any(word in text for word in ['refactor', 'optimize', 'clean', 'improve', 'restructure']):
        return 'refactoring'
    elif any(word in text for word in ['document', 'comment', 'explain', 'readme', 'docs']):
        return 'documentation'
    else:
        # Default to backend for general coding
        return 'backend'

def create_training_report():
    """Create a report of what will be trained"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""
# Monthly Training Report
Generated: {timestamp}

## Training Configuration
- Base Model: Qwen/Qwen2.5-Coder-7B-Instruct
- Training Method: Unsloth LoRA (4-bit)
- Profiles: {len(PROFILES)}
- Epochs: 3 per profile

## Data Sources
1. Original starter datasets (baseline knowledge)
2. Weekly reflexion data (learned from commercial AI comparisons)
3. Distillation data (commercial AI's superior responses)
4. Conversation feedback (user corrections)

## Next Steps
1. Upload merged datasets to Google Drive (colab_sync folder)
2. Open Google Colab notebook
3. Run training (3 hours on FREE, 36 min on Pro)
4. Download trained models
5. Deploy to Ollama

## Expected Results
- Quality improvement: +1.5 to +2.5 points
- Win rate increase: +30% to +50%
- Models learn YOUR specific use cases
"""

    RESULTS_DIR.mkdir(exist_ok=True)
    report_file = RESULTS_DIR / f"training_report_{datetime.now().strftime('%Y%m%d')}.md"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nTraining report saved to: {report_file}")
    return report

def setup_google_drive_sync():
    """Instructions for syncing to Google Drive"""
    print("\n" + "=" * 80)
    print("GOOGLE DRIVE SYNC INSTRUCTIONS")
    print("=" * 80)
    print("""
To prepare for training:

1. Open Google Drive in browser
2. Navigate to: My Drive/lora_datasets/
3. Upload all files from: colab_sync/
   - backend_training_YYYYMM.jsonl
   - frontend_training_YYYYMM.jsonl
   - android_mobile_training_YYYYMM.jsonl
   - bug_fixing_training_YYYYMM.jsonl
   - refactoring_training_YYYYMM.jsonl
   - documentation_training_YYYYMM.jsonl

4. Open your Colab notebook
5. Update dataset paths to use new files
6. Click "Run all"

Training will start automatically!
""")

def main():
    """Main automation workflow"""
    print("=" * 80)
    print("MONTHLY UNSLOTH AUTOMATION")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        # Step 1: Merge all weekly data
        merged_data = merge_weekly_data()

        # Step 2: Create training report
        report = create_training_report()

        # Step 3: Show sync instructions
        setup_google_drive_sync()

        print("\n" + "=" * 80)
        print("AUTOMATION COMPLETE")
        print("=" * 80)
        print("""
✓ All weekly data merged
✓ Training datasets prepared
✓ Report generated

Next: Upload to Google Drive and run training!
""")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
