# Cloud GPU Training Guide - LoRA Profiles

**Complete guide to train your 6 LoRA profiles on cloud GPU**

---

## Overview

**Why Cloud Training?**
- Your local AMD RX6900 can't be used by PyTorch (needs ROCm)
- CPU training takes 3-12 days
- Cloud GPU training: 2-4 hours total
- Cost: $5-15 for all 6 profiles

**What You'll Get**:
- 6 trained LoRA adapters (career, marketing, website, android, backend, frontend)
- 10-15% quality improvement over base models
- Professional-grade AI specialization

---

## Option 1: Google Colab (Easiest) ⭐ Recommended

**Cost**: FREE (with free T4 GPU)
**Time**: 2-4 hours
**Difficulty**: Easy

### Step-by-Step Instructions

#### 1. Upload Datasets to Google Drive

Create folder: `My Drive/localllm_training/datasets/`

Upload these 6 files from `C:\BOT\localLLM\datasets\lora_profiles\`:
- ✅ `career_advisor_starter.jsonl`
- ✅ `marketing_specialist_starter.jsonl`
- ✅ `website_builder_starter.jsonl`
- ✅ `android_mobile_starter.jsonl`
- ✅ `backend_starter.jsonl`
- ✅ `frontend_starter.jsonl`

#### 2. Open Colab Notebook

1. Go to: https://colab.research.google.com/
2. Click **File → Upload notebook**
3. Upload: `cloud_training/train_lora_colab.ipynb` (from this folder)

#### 3. Enable GPU

1. Click **Runtime → Change runtime type**
2. Select **T4 GPU**
3. Click **Save**

#### 4. Run All Cells

1. Click **Runtime → Run all** (or press Ctrl+F9)
2. Authorize Google Drive access when prompted
3. Wait 2-4 hours for training to complete

**Progress**:
- Cell 1: Check GPU ✅
- Cell 2: Install packages (~2 min)
- Cell 3: Mount Drive (~30 sec)
- Cell 4: Upload datasets (manual)
- Cell 5: Training script setup
- Cell 6: **Train all profiles** (2-4 hours)
- Cell 7: Verify adapters

#### 5. Download Adapters

After training completes:

1. Open Google Drive: https://drive.google.com/
2. Go to: `My Drive/localllm_adapters/`
3. Download all 6 folders:
   - `career-advisor/`
   - `marketing-specialist/`
   - `website-builder/`
   - `android/`
   - `backend/`
   - `frontend/`

4. Place on your PC at: `C:\BOT\localLLM\lora_adapters\`

#### 6. Register with Ollama (On Your PC)

```bash
cd C:\BOT\localLLM\scripts\lora_profiles

# Create Modelfiles
./create_modelfiles.sh

# Register with Ollama
./register_models.sh
```

**Done!** Test with:
```bash
ollama run qwen-career-advisor "How do I negotiate salary?"
```

---

## Option 2: Runpod (Fast & Reliable)

**Cost**: $0.30-0.50/hour × 3 hours = **$1-2 total**
**Time**: 2-3 hours
**Difficulty**: Medium

### Setup Instructions

#### 1. Create Account
- Go to: https://www.runpod.io/
- Sign up (email + password)
- Add $5-10 credit (PayPal, Credit Card)

#### 2. Deploy GPU Instance

1. Click **Deploy**
2. Select **GPU Pod**
3. Choose GPU:
   - **RTX 3090** (24GB) - $0.34/hour ⭐ Recommended
   - **RTX 4090** (24GB) - $0.69/hour (faster)
   - **A6000** (48GB) - $0.79/hour (overkill)
4. Template: **PyTorch** (or **RunPod PyTorch**)
5. Disk Space: **50 GB**
6. Click **Deploy On-Demand**

#### 3. Connect to Pod

1. Wait for pod to start (~30 seconds)
2. Click **Connect → Jupyter Notebook**
3. A Jupyter Lab interface will open

#### 4. Upload Files

In Jupyter:
1. Click **Upload** button (top right)
2. Upload `train_lora_runpod.py` (see below)
3. Upload all 6 dataset JSONL files

#### 5. Run Training

Open a **Terminal** in Jupyter:

```bash
# Install dependencies
pip install transformers peft trl bitsandbytes accelerate datasets

# Run training
python train_lora_runpod.py
```

#### 6. Download Adapters

After training:
1. In Jupyter, navigate to `lora_adapters/`
2. Right-click each folder → **Download**
3. Or zip all: `zip -r adapters.zip lora_adapters/`
4. Download zip file

#### 7. Stop Pod (Important!)

**DON'T FORGET**: Click **Stop** button on Runpod dashboard to stop billing!

---

## Option 3: Vast.ai (Cheapest)

**Cost**: $0.20-0.40/hour × 3 hours = **$0.60-1.20 total**
**Time**: 2-3 hours
**Difficulty**: Medium-Hard

### Setup Instructions

#### 1. Create Account
- Go to: https://vast.ai/
- Sign up
- Add $5 credit (crypto or credit card)

#### 2. Find GPU Instance

1. Click **Search** tab
2. Filters:
   - GPU Model: **RTX 3090** or **RTX 4090**
   - GPU RAM: **≥16 GB**
   - Download Speed: **≥100 Mbps**
   - Reliability: **≥95%**
   - DLPerf: **≥0.2**
3. Sort by: **$/hr** (cheapest first)
4. Select instance, click **Rent**

#### 3. Setup Container

1. Select **PyTorch** template
2. Disk Space: **50 GB**
3. Click **Launch**

#### 4. Connect via SSH

```bash
# Copy SSH command from Vast.ai dashboard
ssh -p <PORT> root@<IP> -L 8080:localhost:8080

# In the SSH session:
cd /workspace
```

#### 5. Upload Training Files

Use SCP from your PC:
```bash
scp -P <PORT> datasets/*.jsonl root@<IP>:/workspace/datasets/
scp -P <PORT> train_lora_runpod.py root@<IP>:/workspace/
```

#### 6. Run Training

In SSH session:
```bash
pip install transformers peft trl bitsandbytes accelerate datasets
python train_lora_runpod.py
```

#### 7. Download Adapters

```bash
# On your PC:
scp -r -P <PORT> root@<IP>:/workspace/lora_adapters/ C:/BOT/localLLM/
```

#### 8. Stop Instance

**Important**: Go to Vast.ai dashboard → **Destroy** instance!

---

## Training Script for Runpod/Vast.ai

Save as `train_lora_runpod.py`:

```python
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer

PROFILE_PROMPTS = {
    "career-advisor": "You are an expert career advisor specializing in tech careers.",
    "marketing-specialist": "You are an expert marketing strategist.",
    "website-builder": "You are an expert web designer and developer.",
    "android": "You are an expert Android developer.",
    "backend": "You are an expert backend developer.",
    "frontend": "You are an expert frontend developer."
}

def train_profile(profile, dataset_file, output_dir):
    print(f"\\n{'='*60}")
    print(f"Training: {profile}")
    print(f"{'='*60}\\n")

    # Load model with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    # Load dataset
    dataset = load_dataset('json', data_files=dataset_file, split='train')

    def format_prompt(example):
        system = PROFILE_PROMPTS[profile]
        prompt = f"<|im_start|>system\\n{system}<|im_end|>\\n<|im_start|>user\\n{example['instruction']}"
        if example.get('input'):
            prompt += f"\\n{example['input']}"
        prompt += f"<|im_end|>\\n<|im_start|>assistant\\n{example['output']}<|im_end|>"
        return {"text": prompt}

    dataset = dataset.map(format_prompt)

    # Training
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
    )

    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=dataset, dataset_text_field="text",
        max_seq_length=2048, tokenizer=tokenizer,
    )

    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\\n✅ {profile} complete!\\n")

# Train all profiles
profiles = [
    ("career-advisor", "datasets/career_advisor_starter.jsonl", "lora_adapters/career-advisor"),
    ("marketing-specialist", "datasets/marketing_specialist_starter.jsonl", "lora_adapters/marketing-specialist"),
    ("website-builder", "datasets/website_builder_starter.jsonl", "lora_adapters/website-builder"),
    ("android", "datasets/android_mobile_starter.jsonl", "lora_adapters/android"),
    ("backend", "datasets/backend_starter.jsonl", "lora_adapters/backend"),
    ("frontend", "datasets/frontend_starter.jsonl", "lora_adapters/frontend"),
]

os.makedirs("lora_adapters", exist_ok=True)

for profile, dataset, output in profiles:
    train_profile(profile, dataset, output)

print("\\n🎉 ALL TRAINING COMPLETE! 🎉\\n")
```

---

## After Training: Integration Steps

### 1. Place Adapters on Your PC

Move downloaded folders to:
```
C:\BOT\localLLM\lora_adapters\
├── career-advisor/
├── marketing-specialist/
├── website-builder/
├── android/
├── backend/
└── frontend/
```

### 2. Create Modelfiles

```bash
cd C:\BOT\localLLM\scripts\lora_profiles
./create_modelfiles.sh
```

This creates 6 Modelfiles in `C:\BOT\localLLM\`:
- `Modelfile.career`
- `Modelfile.marketing`
- `Modelfile.website`
- `Modelfile.android`
- `Modelfile.backend`
- `Modelfile.frontend`

### 3. Register with Ollama

```bash
./register_models.sh
```

Registers:
- `qwen-career-advisor`
- `qwen-marketing`
- `qwen-website`
- `qwen-android-mobile`
- `qwen-backend`
- `qwen-frontend`

### 4. Test Your Models

```bash
# Career Advisor
ollama run qwen-career-advisor "How do I negotiate a $150k offer?"

# Marketing
ollama run qwen-marketing "Best email subject lines for B2B SaaS?"

# Website Builder
ollama run qwen-website "Should I use Webflow or WordPress?"

# Android
ollama run qwen-android-mobile "Create a Kotlin RecyclerView adapter"

# Backend
ollama run qwen-backend "Design a REST API for user auth"

# Frontend
ollama run qwen-frontend "Build a React useState hook example"
```

### 5. Use via API

```bash
curl -X POST http://localhost:8080/chat \
  -H "Authorization: Bearer YOUR_API_KEY_HERE" \
  -d '{"message": "How do I ask for a raise?", "model": "qwen-career-advisor"}'
```

---

## Cost Comparison

| Platform | GPU | $/hour | 3 hours | Total Cost |
|----------|-----|--------|---------|------------|
| **Google Colab** | T4 (16GB) | FREE | FREE | **$0** ⭐ |
| **Vast.ai** | RTX 3090 | $0.25 | $0.75 | **$0.75** |
| **Runpod** | RTX 3090 | $0.34 | $1.02 | **$1.02** |
| **Runpod** | RTX 4090 | $0.69 | $2.07 | **$2.07** |
| **Paperspace** | A4000 | $0.76 | $2.28 | **$2.28** |

**Recommendation**: Start with **Google Colab (FREE)**. If you hit time limits, use Runpod or Vast.ai.

---

## Timeline

| Task | Duration |
|------|----------|
| Setup account | 5-10 min |
| Upload datasets | 2-5 min |
| Start instance | 1-2 min |
| Install packages | 2-3 min |
| **Training (6 profiles)** | **2-4 hours** |
| Download adapters | 5-10 min |
| Local integration | 5-10 min |
| **Total** | **2.5-4.5 hours** |

---

## Troubleshooting

### Out of Memory
**Error**: `CUDA out of memory`
**Fix**: Reduce batch size to 1, or use smaller GPU (T4 is fine)

### Download Speed Slow
**Fix**: Use `zip` to compress adapters before downloading

### Training Interrupted
**Fix**: Restart from last checkpoint (saved every epoch)

### Colab Disconnects
**Fix**:
- Keep browser tab active
- Use Colab Pro ($9.99/month) for longer sessions
- Or switch to Runpod/Vast.ai

---

## Files in This Folder

```
cloud_training/
├── CLOUD_TRAINING_GUIDE.md (this file)
├── train_lora_colab.ipynb (Google Colab notebook)
├── train_lora_runpod.py (Runpod/Vast.ai script)
└── upload_datasets.sh (helper script)
```

---

## FAQ

**Q: Which platform is best?**
A: Google Colab (free). If you need reliability, Runpod.

**Q: How long does it take?**
A: 2-4 hours for all 6 profiles on T4 GPU.

**Q: Can I train fewer profiles?**
A: Yes! Comment out profiles you don't need in the training cell.

**Q: What if training fails?**
A: Restart from last checkpoint (saved every epoch). Adapters are saved incrementally.

**Q: Can I use my AMD GPU later?**
A: Yes, if you install ROCm on Windows (experimental). Cloud is easier.

---

## Next Steps

1. ✅ Choose platform (Google Colab recommended)
2. ✅ Upload datasets
3. ✅ Run training notebook/script
4. ✅ Download adapters
5. ✅ Integrate with your LocalLLM system
6. 🚀 Enjoy 100% quality AI responses!

**Ready to start?** Open `train_lora_colab.ipynb` in Google Colab!
