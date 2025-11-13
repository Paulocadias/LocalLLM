# Unsloth Training Script for LocalLLM LoRA Profiles
# Run this in Google Colab with Unsloth installed
# Trains all 6 coding profiles: backend, frontend, android, bug_fixing, refactoring, documentation

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

# Profile configurations
PROFILES = {
    "backend": {
        "dataset": "backend_starter.jsonl",
        "description": "REST APIs, databases, microservices"
    },
    "frontend": {
        "dataset": "frontend_starter.jsonl",
        "description": "React, JavaScript, TypeScript"
    },
    "android": {
        "dataset": "android_mobile_starter.jsonl",
        "description": "Kotlin, Java, Android development"
    },
    "bug_fixing": {
        "dataset": "bug_fixing_starter.jsonl",
        "description": "Debugging and error fixing"
    },
    "refactoring": {
        "dataset": "refactoring_starter.jsonl",
        "description": "Code optimization and cleanup"
    },
    "documentation": {
        "dataset": "documentation_starter.jsonl",
        "description": "Writing docs and comments"
    }
}

BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
MAX_SEQ_LENGTH = 2048
EPOCHS = 3

def format_prompts(examples):
    """Format dataset for Qwen chat template"""
    texts = []
    instruction_key = "instruction" if "instruction" in examples else "prompt"

    for instruction, response in zip(examples[instruction_key], examples["response"]):
        text = f"""<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>"""
        texts.append(text)
    return {"text": texts}

def train_profile(profile_name, config):
    """Train a single LoRA profile"""
    print(f"\n{'='*80}")
    print(f"Training: {profile_name}")
    print(f"Description: {config['description']}")
    print(f"Dataset: {config['dataset']}")
    print(f"{'='*80}\n")

    # Load model
    print("[1/5] Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = BASE_MODEL,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = True,
    )

    # Configure LoRA
    print("[2/5] Configuring LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = True,
        random_state = 3407,
    )

    # Load dataset
    print("[3/5] Loading dataset...")
    dataset_path = f"/content/drive/MyDrive/lora_datasets/{config['dataset']}"
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.map(format_prompts, batched=True, remove_columns=dataset.column_names)

    print(f"Dataset size: {len(dataset)} examples")

    # Training arguments
    print("[4/5] Configuring training...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = EPOCHS,
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = f"outputs/{profile_name}",
            save_strategy = "epoch",
        ),
    )

    # Train!
    print(f"[5/5] Training {profile_name}...")
    print(f"Epochs: {EPOCHS}, Batch size: 2, Gradient accumulation: 4")
    trainer.train()

    # Save LoRA adapter
    output_dir = f"lora_adapters/{profile_name}"
    print(f"\nSaving LoRA adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save merged model (16-bit)
    merged_dir = f"merged_models/qwen-{profile_name}"
    print(f"Saving merged model to {merged_dir}...")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    print(f"\nSUCCESS: {profile_name} training complete!")
    print(f"  LoRA adapter: {output_dir}")
    print(f"  Merged model: {merged_dir}")

    # Clean up to free memory
    del model
    del trainer
    torch.cuda.empty_cache()

    return True

def main():
    """Train all profiles sequentially"""
    print("="*80)
    print("Unsloth Training - LocalLLM LoRA Profiles")
    print("="*80)
    print(f"\nBase Model: {BASE_MODEL}")
    print(f"Profiles to train: {len(PROFILES)}")
    print(f"Epochs per profile: {EPOCHS}")
    print(f"\nEstimated time: {len(PROFILES) * 30} minutes (FREE on Colab!)")
    print("\n" + "="*80 + "\n")

    results = {}

    for profile_name, config in PROFILES.items():
        try:
            success = train_profile(profile_name, config)
            results[profile_name] = "SUCCESS" if success else "FAILED"
        except Exception as e:
            print(f"\nERROR training {profile_name}: {e}")
            results[profile_name] = f"FAILED: {e}"
            continue

    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    for profile, status in results.items():
        print(f"  {profile:20s}: {status}")

    successful = sum(1 for s in results.values() if s == "SUCCESS")
    print(f"\nTotal: {successful}/{len(PROFILES)} profiles trained successfully")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Download merged models from Colab to your local machine")
    print("2. Convert to GGUF: python convert_hf_to_gguf.py merged_models/qwen-backend")
    print("3. Deploy to Ollama: ollama create qwen-backend -f Modelfile.backend")
    print("")
    print("Cost: FREE (Google Colab T4)")
    print("Time: ~3 hours total for all 6 profiles")
    print("Quality: 9.0-9.5/10 expected")

if __name__ == "__main__":
    main()
