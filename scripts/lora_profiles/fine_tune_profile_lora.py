"""
LoRA Fine-Tuning Script for Coding Profiles
Trains profile-specific LoRA adapters for qwen3-coder-32b

Usage:
    python fine_tune_profile_lora.py --profile android --dataset datasets/android_mobile.jsonl --output lora_adapters/android_mobile
"""

import argparse
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Profile-specific system prompts
PROFILE_PROMPTS = {
    "android": "You are an expert Android developer specializing in Kotlin and modern Android development practices with Jetpack Compose.",
    "backend": "You are an expert backend developer specializing in API design, database optimization, and server-side architecture.",
    "frontend": "You are an expert frontend developer specializing in React, TypeScript, and modern web development best practices.",
    "bug_fixing": "You are an expert at debugging code, identifying root causes of errors, and providing clear fix solutions with explanations.",
    "refactoring": "You are an expert at code refactoring, optimization, and improving code quality while maintaining functionality.",
    "documentation": "You are an expert at writing clear, comprehensive code documentation and explaining technical concepts."
}


def load_model_and_tokenizer(model_name: str, use_quantization: bool = True):
    """Load base model with optional 4-bit quantization"""
    logger.info(f"Loading model: {model_name}")

    # Configure 4-bit quantization for memory efficiency
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offload for memory
        )
        logger.info("Using 4-bit quantization with CPU offload")
    else:
        bnb_config = None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Determine max memory allocation (leave headroom for system)
    import psutil
    available_ram_gb = psutil.virtual_memory().available / (1024**3)
    max_memory = {
        "cpu": f"{int(available_ram_gb * 0.6)}GB",  # Use 60% of available RAM (conservative)
    }
    # Only add GPU if CUDA is available
    if torch.cuda.is_available():
        max_memory[0] = "12GB"
    logger.info(f"Memory allocation: {max_memory}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    # Load model with memory constraints
    device_map = "cpu" if not torch.cuda.is_available() else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        max_memory=max_memory,
        low_cpu_mem_usage=True  # Optimize CPU memory usage
    )
    logger.info(f"Model loaded on device: {device_map}")

    if use_quantization:
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def create_lora_config():
    """Configure LoRA parameters"""
    return LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA scaling factor
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )


def format_training_data(example, profile: str):
    """Format dataset example with profile-specific system prompt"""
    system_prompt = PROFILE_PROMPTS.get(profile, "You are a helpful coding assistant.")

    formatted = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{example['instruction']}<|im_end|>
<|im_start|>assistant
{example['response']}<|im_end|>"""

    return {"text": formatted}


def fine_tune_lora(
    profile: str,
    dataset_path: str,
    output_dir: str,
    model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048
):
    """Main fine-tuning function"""

    logger.info(f"Starting LoRA fine-tuning for profile: {profile}")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Output: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Create LoRA configuration
    lora_config = create_lora_config()

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # Format dataset
    logger.info("Formatting dataset...")
    formatted_dataset = dataset.map(
        lambda x: format_training_data(x, profile),
        remove_columns=dataset.column_names
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        report_to="tensorboard"
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=lora_config,
        max_seq_length=max_seq_length,
        dataset_text_field="text"
    )

    # Start training
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LoRA adapter for coding profile")
    parser.add_argument("--profile", type=str, required=True, choices=PROFILE_PROMPTS.keys(),
                        help="Coding profile to train")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to JSONL dataset file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for LoRA adapter")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-32B-Instruct",
                        help="Base model name")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Learning rate")

    args = parser.parse_args()

    fine_tune_lora(
        profile=args.profile,
        dataset_path=args.dataset,
        output_dir=args.output,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
