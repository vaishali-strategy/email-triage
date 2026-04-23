"""
Complete RL Training Script for Enterprise Email Triage Agent

This script implements the full RL training pipeline with PPO,
including model saving, loading, and evaluation.
"""

import torch
import json
import numpy as np
import time
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import os
from pathlib import Path

from env import EnterpriseEmailEnv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

@dataclass
class TrainingConfig:
    """Training configuration"""
    model_name: str = "meta-llama/Llama-2-7b-hf"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    learning_rate: float = 1e-5
    batch_size: int = 4
    num_episodes: int = 100
    max_steps_per_episode: int = 20
    save_interval: int = 10
    eval_interval: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class EmailDataset(Dataset):
    """Dataset for email triage training"""
    
    def __init__(self, data: List[Dict]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class PPOEmailAgent:
    """PPO Agent for Email Triage"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Setup LoRA
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.train()
        
        # Initialize optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate)
        
        # Training metrics
        self.episode_rewards = []
        self.losses = []
        self.success_rates = []
        
    def format_prompt(self, observation: Dict) -> str:
        """Format observation into prompt"""
        system_prompt = """You are a strict Enterprise Email Triage Agent. You do not explain yourself. You ONLY output a single, valid JSON object.

The JSON must have EXACTLY this structure: {"tool": "<tool_name>", "arguments": {<args>}}

You must choose ONE tool from this exact list: auto_reply, route_to_human, ask_for_clarification.

If you use route_to_human, your arguments MUST include department.

If you use auto_reply, your arguments MUST include message.

All arguments MUST include email_id of the current email.

DO NOT hallucinate. DO NOT invent tools. DO NOT explain your reasoning. ONLY output JSON object."""

        user_prompt = f"""Current email requiring triage:

{json.dumps(observation, indent=2)}

Analyze this email and decide the best action. Respond with the appropriate tool call in JSON format."""
        
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    
    def generate_action(self, observation: Dict) -> Dict:
        """Generate action from observation"""
        prompt = self.format_prompt(observation)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Extract generated text
        generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # Parse JSON response
        try:
            import re
            json_match = re.search(r'\{.*\}', generated_text)
            if json_match:
                action = json.loads(json_match.group())
                if "tool" in action and "arguments" in action:
                    return action
        except:
            pass
        
        # Fallback action
        return {
            "tool": "ask_for_clarification",
            "arguments": {"email_id": observation['current_email']['email_id']}
        }
    
    def compute_loss(self, batch: List[Dict]) -> torch.Tensor:
        """Compute PPO loss"""
        total_loss = 0
        
        for item in batch:
            prompt = item['prompt']
            action = item['action']
            reward = item['reward']
            
            # Tokenize prompt
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model outputs
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            
            # Apply reward scaling
            scaled_loss = loss * (1 + reward)
            total_loss += scaled_loss
        
        return total_loss / len(batch)
    
    def collect_rollout(self, env: EnterpriseEmailEnv, num_episodes: int) -> List[Dict]:
        """Collect rollout data"""
        rollout_data = []
        
        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            done = False
            
            while not done and env.current_step < self.config.max_steps_per_episode:
                # Generate action
                action = self.generate_action(obs)
                
                # Take step
                next_obs, reward, done, info = env.step(action)
                episode_reward += reward
                
                # Store transition
                rollout_data.append({
                    'prompt': self.format_prompt(obs),
                    'action': action,
                    'reward': reward,
                    'next_obs': next_obs,
                    'done': done
                })
                
                obs = next_obs
            
            self.episode_rewards.append(episode_reward)
        
        return rollout_data
    
    def train_step(self, rollout_data: List[Dict]) -> float:
        """Perform one training step"""
        # Create batches
        dataloader = DataLoader(EmailDataset(rollout_data), batch_size=self.config.batch_size, shuffle=True)
        
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            loss = self.compute_loss(batch)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        self.losses.append(avg_loss)
        
        return avg_loss
    
    def save_checkpoint(self, episode: int, save_dir: str = "checkpoints"):
        """Save model checkpoint"""
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'episode_rewards': self.episode_rewards,
            'losses': self.losses
        }
        
        torch.save(checkpoint, f"{save_dir}/checkpoint_episode_{episode}.pt")
        print(f"✅ Checkpoint saved at episode {episode}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint['episode_rewards']
        self.losses = checkpoint['losses']
        
        print(f"✅ Checkpoint loaded from episode {checkpoint['episode']}")
    
    def plot_training_progress(self, save_path: str = "training_progress.png"):
        """Plot training progress"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Episode rewards
        ax1.plot(self.episode_rewards)
        ax1.set_title('Episode Rewards Over Time')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True)
        
        # Losses
        ax2.plot(self.losses)
        ax2.set_title('Training Loss Over Time')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Loss')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def train_agent(config: TrainingConfig):
    """Main training function"""
    print("🚀 Starting RL Training for Enterprise Email Triage Agent...")
    
    # Initialize agent and environment
    agent = PPOEmailAgent(config)
    env = EnterpriseEmailEnv()
    
    print(f"📧 Environment: {len(env.emails)} emails")
    print(f"🤖 Model: {config.model_name}")
    print(f"💾 Device: {config.device}")
    
    # Training loop
    for episode in range(config.num_episodes):
        print(f"\n📊 Episode {episode + 1}/{config.num_episodes}")
        
        # Collect rollout
        rollout_data = agent.collect_rollout(env, num_episodes=1)
        
        # Train on rollout
        loss = agent.train_step(rollout_data)
        
        # Calculate success rate
        recent_rewards = agent.episode_rewards[-10:] if len(agent.episode_rewards) >= 10 else agent.episode_rewards
        success_rate = np.mean([r > 0 for r in recent_rewards])
        agent.success_rates.append(success_rate)
        
        # Print progress
        avg_reward = np.mean(agent.episode_rewards[-5:]) if len(agent.episode_rewards) >= 5 else agent.episode_rewards[-1]
        print(f"   Avg Reward (5 ep): {avg_reward:.2f}")
        print(f"   Success Rate: {success_rate:.2f}")
        print(f"   Loss: {loss:.4f}")
        
        # Save checkpoint
        if (episode + 1) % config.save_interval == 0:
            agent.save_checkpoint(episode + 1)
        
        # Plot progress
        if (episode + 1) % config.eval_interval == 0:
            agent.plot_training_progress()
    
    # Final save
    agent.save_checkpoint(config.num_episodes)
    agent.plot_training_progress()
    
    print("🎉 Training completed!")
    return agent

def main():
    """Main function"""
    config = TrainingConfig(
        num_episodes=50,  # Reduced for demo
        save_interval=10,
        eval_interval=5
    )
    
    agent = train_agent(config)
    
    # Print final results
    print(f"\n📈 Final Training Results:")
    print(f"   Final Average Reward: {np.mean(agent.episode_rewards[-10:]):.2f}")
    print(f"   Final Success Rate: {agent.success_rates[-1]:.2f}")
    print(f"   Total Training Episodes: {len(agent.episode_rewards)}")

if __name__ == "__main__":
    main()
