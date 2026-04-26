---
title: Enterprise Email Triage Simulator
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: streamlit
pinned: false
license: mit
app_port: 7860
---

# 📧 Enterprise Email Triage Simulator
### Meta PyTorch OpenEnv AI Hackathon Submission

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/Proteinrequired/enterprise-email-triage)

This project introduces an autonomous agentic system for corporate email triage, built on the **OpenEnv** framework. It automates high-volume decision-making—routing VIP issues to humans while auto-responding to routine tasks—using a fine-tuned **Llama-3.2-3B** model.

---

## 📽️ Documentation & Links
* **Technical Blog Post:** [Read the full write-up here](./BLOG.md)
* **Training Workbench:** [View the Training Space here](https://huggingface.co/spaces/Proteinrequired/email-agent-training)

## 💡 Motivation
In large enterprises, communication bottlenecks lead to delayed IT support and missed VIP opportunities. This environment was built to solve the **"Triage Fatigue"** problem by training an agent that understands urgency, sender priority, and corporate context, allowing human employees to focus on complex problem-solving.

---

## 🏗️ Project Architecture: Two-Space Workflow
To ensure a clean production environment, this project utilized a separated architecture:
1. **The Training Factory:** Model fine-tuning, Unsloth compilation, and Behavioral Cloning were executed in an isolated, GPU-heavy environment to prevent memory leaks and keep the production codebase clean.
2. **The Production Showroom (This Repo):** The resulting LoRA adapters, evidence plots, and OpenEnv logic were exported and deployed here for inference, UI interaction, and judge evaluation.

---

## 📊 Training Evidence & Results
The agent underwent Behavioral Cloning (BC) using **Unsloth** and **Hugging Face TRL**. We collected expert-weighted rollouts where the agent was rewarded for accuracy and penalized for misrouting.

| **Training Loss (Convergence)** | **Reward Distribution (Performance)** |
| :---: | :---: |
| ![Loss](./training_loss.png) | ![Rewards](./reward_curve.png) |

> **Analysis:** The loss curve shows successful optimization of the LoRA adapters over 60 steps. The reward histogram proves the agent successfully shifted its behavior toward high-reward actions.

---

## 🧠 Environment Logic (OpenEnv)
This project extends the **OpenEnv** framework to handle high-dimensional text-based corporate state spaces.

### LLM-Friendly State Space
The environment returns observations as flat dictionaries optimized for LLM prompt injection:

```python
{
    "current_email": {
        "email_id": "email_001",
        "sender": "user@company.com",
        "subject": "Password Reset Request",
        "body": "Email content...",
        "is_vip": False,
        "suggested_department": "IT"
    },
    "available_tools": ["route_to_human", "auto_reply", "ask_for_clarification"]
}
```

### Enhanced Reward Structure
* **+10.0**: Route VIP outage/HR issues to correct departments or successful auto-reply to routine tasks.
* **+5.0 to +8.5**: Route to suboptimal but acceptable departments.
* **-1.0 to -2.75**: Unnecessary clarification requests.
* **-5.0**: Incorrect routing (e.g., auto-replying to an angry client).

---

## 📂 File Manifest
* `env.py`: OpenEnv-compliant environment definition.
* `dataset.json`: Synthetic corporate dataset (100+ email scenarios).
* `reward_system.py`: Dynamic reward logic for agent optimization.
* `training_script.ipynb`: Fully documented training script with logs.
* `inference.py`: Standalone script to test the model's "Before and After" behavior locally.
* `app.py`: Streamlit-based UI for the live showcase.
* `openenv.yaml`: Configuration file for environment validation.
* **`email-triage-lora-final.tar.gz/`**: The exported fine-tuned model artifacts containing:
    * `adapter_model.safetensors`: The trained Behavioral Cloning weights.
    * `adapter_config.json`: The LoRA configuration used.
    * `tokenizer.json` & `special_tokens_map.json`: Tokenizer settings enforcing JSON tool-calling.

---

## 🧪 How to Reproduce the Training
If you would like to run the training script (`train.ipynb`) locally or in a Colab environment, you will need the dataset and environment files from this repository.

**Option 1: Clone the Repository (Recommended)**
Clone this repository directly to get all files, including the pre-trained adapters:
```bash
git clone [https://huggingface.co/spaces/Proteinrequired/enterprise-email-triage](https://huggingface.co/spaces/Proteinrequired/enterprise-email-triage)
cd enterprise-email-triage
```

**Option 2: Using Hugging Face CLI**
```bash
huggingface-cli download spaces/Proteinrequired/enterprise-email-triage --local-dir ./email-triage
```

**Option 3: Manual Download**
Navigate to the **Files** tab at the top of this Space and manually download `training_script.ipynb`, `dataset.json`, `env.py`, `reward_system.py`.

---

## Hackathon Requirements Met
- [x] OpenEnv-compliant environment
- [x] LLM tool call action format
- [x] Working training script (Unsloth/TRL) provided via `training_script.ipynb`
- [x] Evidence of training (Loss and Reward plots embedded)
- [x] Pushed to Hugging Face Space for discoverability
- [x] Comprehensive documentation and blog links

### Acknowledgments
* **Meta PyTorch Team** for the OpenEnv framework.
* **National Institute of Technology Delhi (NITD)** for institutional support.
* **Unsloth AI** for high-performance training kernels.

**Author:** [Vaishali](https://huggingface.co/Proteinrequired)
```


Once you paste this in, double-check that your `email-triage-lora-final` folder is successfully uploaded to the Files tab, and you are 100% ready to submit!
