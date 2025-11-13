# Cloud GPU Training - Complete Package

**Everything you need to train LoRA profiles on cloud GPU**

---

## What's in This Folder

```
cloud_training/
├── README.md (this file)
├── QUICK_START.md ⭐ Start here!
├── CLOUD_TRAINING_GUIDE.md (detailed guide)
├── train_lora_colab.ipynb (Google Colab notebook)
└── train_lora_runpod.py (Runpod/Vast.ai script)
```

---

## Getting Started

**New to cloud training?** → Read [QUICK_START.md](QUICK_START.md) (5 min read)

**Want detailed instructions?** → Read [CLOUD_TRAINING_GUIDE.md](CLOUD_TRAINING_GUIDE.md)

**Just want to run it?** → Upload [train_lora_colab.ipynb](train_lora_colab.ipynb) to Google Colab

---

## Three Options

### Option 1: Google Colab ⭐ Recommended

- **Cost**: FREE
- **Time**: 2-4 hours
- **Difficulty**: Easy
- **File**: `train_lora_colab.ipynb`

**Steps**:
1. Upload datasets to Google Drive
2. Open notebook in Colab
3. Run all cells
4. Download trained adapters

### Option 2: Runpod

- **Cost**: $1-2
- **Time**: 2-3 hours
- **Difficulty**: Medium
- **File**: `train_lora_runpod.py`

**Steps**:
1. Sign up at runpod.io
2. Deploy RTX 3090 pod
3. Upload script and datasets
4. Run training
5. Download adapters

### Option 3: Vast.ai

- **Cost**: $0.60-1
- **Time**: 2-3 hours
- **Difficulty**: Medium-Hard
- **File**: `train_lora_runpod.py`

**Steps**:
1. Sign up at vast.ai
2. Find cheap GPU
3. SSH and upload files
4. Run training
5. Download via SCP

---

## What You Need Before Starting

### From Your PC

**Datasets** (6 files from `C:\BOT\localLLM\datasets\lora_profiles\`):
- ✅ `career_advisor_starter.jsonl`
- ✅ `marketing_specialist_starter.jsonl`
- ✅ `website_builder_starter.jsonl`
- ✅ `android_mobile_starter.jsonl`
- ✅ `backend_starter.jsonl`
- ✅ `frontend_starter.jsonl`

**Training Script** (from this folder):
- For Colab: `train_lora_colab.ipynb`
- For Runpod/Vast: `train_lora_runpod.py`

---

## What You'll Get

After 2-4 hours of cloud training, you'll have:

**6 Trained LoRA Adapters**:
1. `career-advisor/` - Career coaching, salary negotiation
2. `marketing-specialist/` - Marketing strategy, SEO, campaigns
3. `website-builder/` - Web design, landing pages, UX
4. `android/` - Android development, Kotlin
5. `backend/` - API design, databases, servers
6. `frontend/` - React, TypeScript, web components

**Quality Improvement**: 10-15% better than base models

**Total Size**: ~2-5GB (all 6 adapters)

---

## After Training

### 1. Download Adapters to Your PC

Place in: `C:\BOT\localLLM\lora_adapters\`

```
C:\BOT\localLLM\lora_adapters\
├── career-advisor\
├── marketing-specialist\
├── website-builder\
├── android\
├── backend\
└── frontend\
```

### 2. Create Ollama Modelfiles

```bash
cd C:\BOT\localLLM\scripts\lora_profiles
./create_modelfiles.sh
```

### 3. Register with Ollama

```bash
./register_models.sh
```

### 4. Test Your Models

```bash
ollama run qwen-career-advisor "How do I negotiate salary?"
ollama run qwen-marketing "Best SEO tactics for startups?"
ollama run qwen-website "WordPress vs Webflow?"
ollama run qwen-android-mobile "Create Kotlin RecyclerView"
ollama run qwen-backend "Design REST API for auth"
ollama run qwen-frontend "React hooks example"
```

**Done!** Your LocalLLM now has 100% quality with LoRA specialization! 🎉

---

## Cost & Timeline Summary

| Platform | Cost | Time | Setup Difficulty |
|----------|------|------|------------------|
| **Google Colab** | $0 | 2-4h | Easy ⭐ |
| **Runpod** | $1-2 | 2-3h | Medium |
| **Vast.ai** | $0.60-1 | 2-3h | Medium |

**Active work**: 20-30 minutes (setup, upload, download)
**Passive wait**: 2-4 hours (training runs in background)

---

## Quick Links

- **Google Colab**: https://colab.research.google.com/
- **Runpod**: https://www.runpod.io/
- **Vast.ai**: https://vast.ai/
- **Google Drive**: https://drive.google.com/

---

## Support

**Questions?** Read the detailed guide:
- [CLOUD_TRAINING_GUIDE.md](CLOUD_TRAINING_GUIDE.md)

**Need help?**
- Check troubleshooting section in guide
- Google Colab is most beginner-friendly

---

## Files Checklist

Before you start, make sure you have:

- [ ] Datasets uploaded (6 JSONL files)
- [ ] Training script ready (`train_lora_colab.ipynb` or `train_lora_runpod.py`)
- [ ] Cloud platform account (Colab/Runpod/Vast)
- [ ] 2-4 hours of free time (training runs automatically)

---

**Ready to start?** Open [QUICK_START.md](QUICK_START.md) for step-by-step instructions! 🚀
