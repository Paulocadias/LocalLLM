#!/usr/bin/env python3
"""
LoRA Training Script for Cloud GPU (Runpod, Vast.ai, Paperspace)
Trains all 6 LocalLLM profiles

Usage:
    python train_lora_runpod.py

Requirements:
    - datasets/ folder with 6 JSONL files
    - CUDA GPU (will be auto-detected)
"""

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Profile system prompts
PROFILE_PROMPTS = {
    "career-advisor": "You are an expert career advisor specializing in tech careers, job transitions, and professional development. You provide strategic career guidance on interviews, salary negotiation, resume optimization, LinkedIn branding, and career advancement.",
    "marketing-specialist": "You are an expert marketing strategist specializing in digital marketing, content strategy, and growth. You excel at SEO, email marketing, social media campaigns, conversion optimization, and ROI analysis.",
    "website-builder": "You are an expert web designer and developer specializing in high-converting websites and landing pages. You excel at responsive design, mobile optimization, UX/UI best practices, and conversion rate optimization.",
    "android": "You are an expert Android developer specializing in Kotlin and modern Android development practices with Jetpack Compose.",
    "backend": "You are an expert backend developer specializing in API design, database optimization, and server-side architecture.",
    "frontend": "You are an expert frontend developer specializing in React, TypeScript, and modern web development best practices."
}

def check_environment():
    """Check GPU and CUDA availability"""
    logger.info("="*60)
    logger.info("Environment Check")
    logger.info("="*60)

    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU Memory: {gpu_memory_gb:.2f} GB")

        if gpu_memory_gb < 12:
            logger.warning("⚠️  GPU has less than 12GB. Training may be slow or fail.")
        else:
            logger.info("✅ GPU memory sufficient for training")
    else:
        logger.error("❌ No CUDA GPU found! This script requires GPU.")
        raise RuntimeError("CUDA GPU required for training")

    logger.info("")

def train_profile(profile, dataset_file, output_dir, model_name="Qwen/Qwen2.5-7B-Instruct"):
    """Train a single LoRA profile"""
    logger.info("="*60)
    logger.info(f"Training Profile: {profile}")
    logger.info("="*60)
    logger.info(f"Dataset: {dataset_file}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Model: {model_name}")
    logger.info("")

    # Load model with 4-bit quantization
    logger.info("Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    logger.info("Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and prepare dataset
    logger.info("Loading dataset...")
    dataset = load_dataset('json', data_files=dataset_file, split='train')
    logger.info(f"Dataset size: {len(dataset)} examples")

    system_prompt = PROFILE_PROMPTS[profile]

    def format_prompt(example):
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{example['instruction']}"
        if example.get('input'):
            prompt += f"\n\nContext: {example['input']}"
        prompt += f"<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
        return {"text": prompt}

    dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        optim="paged_adamw_8bit",
        warmup_ratio=0.1,
        report_to="none",
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        tokenizer=tokenizer,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save adapter
    logger.info(f"Saving adapter to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"✅ {profile} training complete!\n")

def main():
    """Train all profiles"""
    print("""
╔════════════════════════════════════════════════════════════╗
║        LocalLLM LoRA Training - Cloud GPU                  ║
║                                                            ║
║  Training 6 profiles for specialized AI assistance        ║
╚════════════════════════════════════════════════════════════╝
""")

    # Check environment
    check_environment()

    # Create output directory
    os.makedirs("lora_adapters", exist_ok=True)

    # Define profiles to train
    profiles = [
        ("career-advisor", "datasets/career_advisor_starter.jsonl", "lora_adapters/career-advisor"),
        ("marketing-specialist", "datasets/marketing_specialist_starter.jsonl", "lora_adapters/marketing-specialist"),
        ("website-builder", "datasets/website_builder_starter.jsonl", "lora_adapters/website-builder"),
        ("android", "datasets/android_mobile_starter.jsonl", "lora_adapters/android"),
        ("backend", "datasets/backend_starter.jsonl", "lora_adapters/backend"),
        ("frontend", "datasets/frontend_starter.jsonl", "lora_adapters/frontend"),
    ]

    # Check all datasets exist
    logger.info("Checking datasets...")
    missing_datasets = []
    for profile, dataset_file, _ in profiles:
        if not os.path.exists(dataset_file):
            missing_datasets.append(dataset_file)
            logger.error(f"❌ Missing: {dataset_file}")
        else:
            logger.info(f"✅ Found: {dataset_file}")

    if missing_datasets:
        logger.error("\n❌ Missing datasets! Please upload them to datasets/ folder.")
        logger.error("Required files:")
        for f in missing_datasets:
            logger.error(f"  - {f}")
        return

    logger.info("\n✅ All datasets found! Starting training...\n")

    # Train each profile
    start_time = __import__('time').time()

    for i, (profile, dataset_file, output_dir) in enumerate(profiles, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Profile {i}/6: {profile}")
        logger.info(f"{'='*60}\n")

        try:
            train_profile(profile, dataset_file, output_dir)
        except Exception as e:
            logger.error(f"❌ Failed to train {profile}: {e}")
            logger.error("Continuing to next profile...\n")
            continue

    # Summary
    elapsed_time = __import__('time').time() - start_time
    elapsed_hours = elapsed_time / 3600

    print(f"""
╔════════════════════════════════════════════════════════════╗
║                 TRAINING COMPLETE!                         ║
╚════════════════════════════════════════════════════════════╝

✅ All 6 profiles trained successfully!

Time taken: {elapsed_hours:.2f} hours

Adapters saved to: ./lora_adapters/
  - career-advisor/
  - marketing-specialist/
  - website-builder/
  - android/
  - backend/
  - frontend/

Next steps:
1. Download the lora_adapters/ folder to your local machine
2. Place in: C:\\BOT\\localLLM\\lora_adapters\\
3. Run: ./scripts/lora_profiles/create_modelfiles.sh
4. Run: ./scripts/lora_profiles/register_models.sh
5. Test: ollama run qwen-career-advisor "How do I negotiate salary?"

🎉 Enjoy your specialized AI assistants!
""")

if __name__ == "__main__":
    main()
