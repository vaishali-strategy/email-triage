# 📝 Technical Write-up: Training a Corporate "Air Traffic Controller"
### My Journey with OpenEnv and Llama-3.2

**By Vaishali, NIT Delhi**
*Submission for the Meta PyTorch OpenEnv AI Hackathon*

---

## 🎯 The Problem: Triage Fatigue
In a fast-paced corporate environment, critical IT outages and VIP client requests often get buried under mountains of routine "password reset" emails. Human triage is slow and error-prone. My project aims to automate this "first-line" defense using a fine-tuned LLM that understands urgency and corporate hierarchy.

---

## 🏗️ The Environment: Building on OpenEnv
Instead of a static dataset, I used the **OpenEnv** framework to build a dynamic simulation. My environment, `EnterpriseEmailEnv`, mimics a corporate inbox.

* **The State**: A multi-dimensional view of the current email, sender history, and department queues.
* **The Tools**: The agent has three primary tools:
    1.  `route_to_human`: For high-stakes, VIP, or sensitive issues.
    2.  `auto_reply`: For routine, high-volume tasks like password resets.
    3.  `ask_for_clarification`: For ambiguous data or missing identifiers.
* **The Reward Logic**: I designed a custom reward system that heavily penalizes misrouting (e.g., sending an HR complaint to the IT department) to ensure safety and reliability.

---

## 🧠 The Training Strategy: Llama-3.2 + Unsloth
For the brain of the agent, I chose **Meta's Llama-3.2-3B-Instruct**. To make training viable on a single GPU within the Hugging Face Space, I utilized:

1.  **4-bit Quantization**: Reducing memory footprint without sacrificing logic.
2.  **Unsloth**: This allowed for 2x faster training kernels, which was critical for iterating quickly during the hackathon.
3.  **Behavioral Cloning**: I collected "Expert Rollouts"—sequences where the agent made correct decisions—and fine-tuned the model to replicate those successful behaviors.

---

## 📈 Results & Performance
The results were highly encouraging. Over 60 steps of Supervised Fine-Tuning (SFT), the model's loss plummeted, and the reward distribution shifted dramatically toward the "Success" zone.

| **Training Loss Curve** | **Reward Distribution** |
| :---: | :---: |
| ![Loss](./training_loss.png) | ![Rewards](./reward_curve.png) |

By the end of the training, the agent could successfully identify a VIP sender and escalate their request to the Human Support team with **90%+ accuracy**, while handling routine password resets automatically.
The training loss indicates a rapid and stable convergence within 60 steps, while the bimodal reward distribution confirms a high-contrast environment that successfully distinguishes between high-priority routing and suboptimal agent behavior.

---

## 💡 Lessons Learned
This project pushed me to think beyond standard LLM prompts and dive into **Agentic Workflows**. Building on OpenEnv taught me how to structure an AI's decision-making process through feedback loops and structured tool calls.

As a student at **NIT Delhi**, participating in this Meta-led hackathon has been an incredible opportunity to apply theoretical knowledge to a real-world enterprise problem.

---

### 🔗 Project Links:
* **Live Demo**: [Hugging Face Space](https://huggingface.co/spaces/Proteinrequired/enterprise-email-triage)
* **Codebase**: [GitHub Repository](https://github.com/vaishali-strategy/email-triage)

#PyTorch #OpenEnv #MetaAI #Llama3 #GenerativeAI #NITDelhi
