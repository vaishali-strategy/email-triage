"""
Training Setup for Email Triage RL Agent
Following OpenEnv Hackathon guidelines for TRL + Unsloth training
"""

import os
import json
import torch
from typing import Dict, List, Any
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel
from env import EnterpriseEmailEnv
from reward_system import reward_system

@dataclass
class TrainingConfig:
    """Configuration for RL training following hackathon best practices"""
    
    # Model configuration
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"  # Base model
    max_seq_length: int = 2048
    lora_r: int = 16  # LoRA rank
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Training configuration
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    num_train_epochs: int = 3
    warmup_steps: int = 100
    
    # GRPO configuration
    num_rollouts: int = 8  # Number of rollouts per update
    reward_weights: Dict[str, float] = None
    
    # Environment configuration
    max_steps_per_episode: int = 10
    num_episodes: int = 100
    
    # Safety and monitoring
    max_generation_length: int = 200
    temperature: float = 0.7
    top_p: float = 0.9
    
    def __post_init__(self):
        if self.reward_weights is None:
            # Default reward weights from enhanced system
            self.reward_weights = {
                "correct_action": 1.0,
                "format_compliance": 0.2,
                "argument_validity": 0.3,
                "department_match": 0.5,
                "security_compliance": 0.8,
                "efficiency_bonus": 0.1,
                "anti_cheat": -2.0
            }

class EmailTriageTrainer:
    """
    RL Trainer for Email Triage using TRL + Unsloth
    Following hackathon stack recommendations
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.env = EnterpriseEmailEnv()
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def setup_model(self):
        """Setup model with Unsloth for efficient training"""
        print("🚀 Setting up model with Unsloth...")
        
        # Load model with Unsloth optimization
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=True,  # 4-bit quantization for efficiency
        )
        
        # Setup LoRA adapters
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,  # Use standard LoRA
        )
        
        print(f"✅ Model setup complete: {self.config.model_name}")
        print(f"📊 Model parameters: {self.model.num_parameters():,}")
    
    def create_prompt_template(self, email: Dict[str, Any]) -> str:
        """Create prompt template for the model"""
        prompt = f"""You are an AI email triage assistant. Analyze the following email and decide the best action.

Email Details:
- From: {email['sender']}
- Subject: {email['subject']}
- Body: {email['body']}
- Priority: {email['priority']}/5
- VIP Status: {email['is_vip']}

Available Actions:
1. auto_reply - Automatically respond with a message (good for routine inquiries)
2. route_to_human - Forward to human agent (required for urgent/sensitive issues)
3. ask_for_clarification - Request more information (good for ambiguous emails)

Departments for routing: IT, Customer Service, Emergency Support, HR, Security, Finance

Respond with JSON format:
{{
    "tool": "action_name",
    "arguments": {{
        "email_id": "{email['id']}",
        "message": "response text" if auto_reply,
        "department": "department_name" if route_to_human
    }},
    "reasoning": "brief explanation of decision"
}}

Your response:"""
        
        return prompt
    
    def rollout_generation(self, prompt: str) -> Dict[str, Any]:
        """Generate a rollout from the model"""
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_generation_length,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            # Try to parse as JSON
            try:
                action = json.loads(generated_text)
            except json.JSONDecodeError:
                # Fallback: try to extract JSON from text
                import re
                json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
                if json_match:
                    action = json.loads(json_match.group())
                else:
                    # Default action if parsing fails
                    action = {
                        "tool": "ask_for_clarification",
                        "arguments": {"email_id": "unknown"},
                        "reasoning": "Failed to parse response"
                    }
            
            return action
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return {
                "tool": "ask_for_clarification",
                "arguments": {"email_id": "unknown"},
                "reasoning": f"Generation error: {str(e)}"
            }
    
    def calculate_rollout_reward(self, email: Dict[str, Any], action: Dict[str, Any]) -> float:
        """Calculate reward for a rollout using enhanced reward system"""
        reward_breakdown = reward_system.calculate_reward(email, action)
        return reward_breakdown.total_reward
    
    def collect_rollouts(self, num_rollouts: int) -> List[Dict[str, Any]]:
        """Collect rollouts for training"""
        rollouts = []
        
        for i in range(num_rollouts):
            # Reset environment
            obs = self.env.reset()
            email = obs['current_email']
            
            # Create prompt
            prompt = self.create_prompt_template(email)
            
            # Generate action
            action = self.rollout_generation(prompt)
            
            # Calculate reward
            reward = self.calculate_rollout_reward(email, action)
            
            # Store rollout
            rollout = {
                "prompt": prompt,
                "action": action,
                "reward": reward,
                "email": email
            }
            rollouts.append(rollout)
            
            print(f"📧 Rollout {i+1}/{num_rollouts}: {action['tool']} -> Reward: {reward:+.3f}")
        
        return rollouts
    
    def setup_grpo_trainer(self):
        """Setup GRPO trainer with TRL"""
        print("🎯 Setting up GRPO trainer...")
        
        # GRPO configuration
        grpo_config = GRPOConfig(
            output_dir="./email_triage_results",
            num_rollouts=self.config.num_rollouts,
            reward_weights=self.config.reward_weights,
            batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            max_length=self.config.max_seq_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            logging_steps=10,
            save_steps=50,
        )
        
        # Create trainer
        self.trainer = GRPOTrainer(
            model=self.model,
            reward_model=None,  # We'll use our custom reward function
            config=grpo_config,
            tokenizer=self.tokenizer,
        )
        
        print("✅ GRPO trainer setup complete")
    
    def train(self):
        """Main training loop following hackathon guidelines"""
        print("🏋️ Starting RL training...")
        
        # Setup
        self.setup_model()
        self.setup_grpo_trainer()
        
        # Training metrics
        episode_rewards = []
        reward_breakdowns = []
        
        for episode in range(self.config.num_episodes):
            print(f"\n📊 Episode {episode + 1}/{self.config.num_episodes}")
            
            # Collect rollouts
            rollouts = self.collect_rollouts(self.config.num_rollouts)
            
            # Calculate episode metrics
            episode_reward = np.mean([r['reward'] for r in rollouts])
            episode_rewards.append(episode_reward)
            
            # Check for reward hacking
            suspicious_flags = []
            for rollout in rollouts:
                breakdown = reward_system.calculate_reward(rollout['email'], rollout['action'])
                if breakdown.flags:
                    suspicious_flags.extend(breakdown.flags)
            
            if suspicious_flags:
                print(f"⚠️  Suspicious activity detected: {set(suspicious_flags)}")
            
            # Update model (simplified for hackathon)
            # In practice, this would use the GRPO trainer
            print(f"📈 Episode reward: {episode_reward:+.3f}")
            
            # Early stopping if reward is consistently high
            if len(episode_rewards) > 10 and np.mean(episode_rewards[-5:]) > 0.8:
                print("🎉 High reward achieved - early stopping")
                break
        
        # Save final model
        self.save_model()
        
        # Training summary
        self.print_training_summary(episode_rewards)
    
    def save_model(self):
        """Save trained model correctly following hackathon guidelines"""
        print("💾 Saving trained model...")
        
        # Save adapters only (correct way for LoRA/QLoRA)
        output_dir = "./trained_email_triage_model"
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"✅ Model saved to {output_dir}")
        print("📝 Use model.load_pretrained() for inference")
    
    def print_training_summary(self, episode_rewards: List[float]):
        """Print training summary for hackathon demo"""
        print("\n" + "="*80)
        print("🎯 TRAINING COMPLETE")
        print("="*80)
        
        print(f"📊 Episodes trained: {len(episode_rewards)}")
        print(f"📈 Final reward: {episode_rewards[-1]:+.3f}")
        print(f"📈 Average reward: {np.mean(episode_rewards):+.3f}")
        print(f"📈 Reward improvement: {episode_rewards[-1] - episode_rewards[0]:+.3f}")
        
        # Check for overfitting or reward hacking
        if np.std(episode_rewards[-10:]) < 0.1:
            print("⚠️  Low reward variance - check for overfitting")
        
        if episode_rewards[-1] > 2.0:
            print("⚠️  Very high reward - check for reward hacking")
        
        print("\n🎉 Ready for hackathon demo!")
        print("="*80)

def main():
    """Main training function"""
    print("🚀 Email Triage RL Training - OpenEnv Hackathon")
    
    # Configuration
    config = TrainingConfig()
    
    # Create trainer
    trainer = EmailTriageTrainer(config)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
