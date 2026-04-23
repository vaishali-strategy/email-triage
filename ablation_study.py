"""
Ablation Study for Enterprise Email Triage Agent

This script performs ablation studies to understand the impact of different
components on agent performance.
"""

import numpy as np
import json
import time
from typing import Dict, List, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

from env import EnterpriseEmailEnv
from evaluate import evaluate_agent, BaselineAgent, EvaluationResults

@dataclass
class AblationConfig:
    """Configuration for ablation study"""
    name: str
    reward_function: str
    prompt_strategy: str
    model_size: str
    temperature: float
    max_tokens: int

class AblationAgent:
    """Agent with configurable components for ablation study"""
    
    def __init__(self, config: AblationConfig):
        self.config = config
        self.reward_function = config.reward_function
        self.prompt_strategy = config.prompt_strategy
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
    
    def select_action(self, observation: Dict) -> Dict:
        """Select action based on configuration"""
        email = observation['current_email']
        
        # Different prompt strategies
        if self.prompt_strategy == "minimal":
            return self._minimal_prompt_action(email)
        elif self.prompt_strategy == "detailed":
            return self._detailed_prompt_action(email)
        elif self.prompt_strategy == "no_context":
            return self._no_context_action(email)
        else:
            return self._standard_action(email)
    
    def _minimal_prompt_action(self, email) -> Dict:
        """Minimal prompt strategy"""
        intent = email['intent']
        
        if intent in ["vip_server_outage", "angry_client_refund"]:
            return {
                "tool": "route_to_human",
                "arguments": {
                    "email_id": email['email_id'],
                    "department": "Emergency Support" if intent == "vip_server_outage" else "Customer Service"
                }
            }
        elif intent == "routine_password_reset":
            return {
                "tool": "auto_reply",
                "arguments": {
                    "email_id": email['email_id'],
                    "message": "Password reset link sent."
                }
            }
        else:
            return {
                "tool": "ask_for_clarification",
                "arguments": {"email_id": email['email_id']}
            }
    
    def _detailed_prompt_action(self, email) -> Dict:
        """Detailed prompt strategy with more context"""
        intent = email['intent']
        priority = email['priority']
        is_vip = email['is_vip']
        
        # Enhanced decision logic
        if intent == "vip_server_outage":
            return {
                "tool": "route_to_human",
                "arguments": {
                    "email_id": email['email_id'],
                    "department": "Emergency Support"
                }
            }
        elif intent == "angry_client_refund":
            return {
                "tool": "route_to_human",
                "arguments": {
                    "email_id": email['email_id'],
                    "department": "Customer Service"
                }
            }
        elif intent == "hr_sensitive":
            return {
                "tool": "route_to_human",
                "arguments": {
                    "email_id": email['email_id'],
                    "department": "HR"
                }
            }
        elif intent == "spear_phishing":
            return {
                "tool": "route_to_human",
                "arguments": {
                    "email_id": email['email_id'],
                    "department": "Security"
                }
            }
        elif intent == "routine_password_reset":
            return {
                "tool": "auto_reply",
                "arguments": {
                    "email_id": email['email_id'],
                    "message": "I'll help you reset your password. Please check your email for the reset link."
                }
            }
        elif intent == "feature_request":
            return {
                "tool": "auto_reply",
                "arguments": {
                    "email_id": email['email_id'],
                    "message": "Thank you for your feature request! We'll forward this to our product team."
                }
            }
        elif intent == "mixed_churn":
            return {
                "tool": "ask_for_clarification",
                "arguments": {"email_id": email['email_id']}
            }
        else:
            return {
                "tool": "route_to_human",
                "arguments": {
                    "email_id": email['email_id'],
                    "department": "General"
                }
            }
    
    def _no_context_action(self, email) -> Dict:
        """No context strategy - ignores email content"""
        tools = ["auto_reply", "route_to_human", "ask_for_clarification"]
        tool = np.random.choice(tools)
        
        if tool == "auto_reply":
            return {
                "tool": tool,
                "arguments": {
                    "email_id": email['email_id'],
                    "message": "Thank you for your message."
                }
            }
        elif tool == "route_to_human":
            departments = ["IT", "Customer Service", "Security", "HR", "Finance"]
            return {
                "tool": tool,
                "arguments": {
                    "email_id": email['email_id'],
                    "department": np.random.choice(departments)
                }
            }
        else:
            return {
                "tool": tool,
                "arguments": {"email_id": email['email_id']}
            }
    
    def _standard_action(self, email) -> Dict:
        """Standard action (baseline)"""
        return BaselineAgent("rule_based").select_action({"current_email": email})

class ModifiedEnvironment(EnterpriseEmailEnv):
    """Environment with modified reward functions for ablation study"""
    
    def __init__(self, reward_function: str = "standard", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_function = reward_function
    
    def _calculate_llm_reward(self, action_type, arguments):
        """Modified reward calculation"""
        if self.reward_function == "binary":
            # Binary reward: +1 for correct action, -1 for incorrect
            return self._binary_reward(action_type, arguments)
        elif self.reward_function == "sparse":
            # Sparse reward: only reward at end of episode
            return self._sparse_reward(action_type, arguments)
        elif self.reward_function == "dense":
            # Dense reward: more granular feedback
            return self._dense_reward(action_type, arguments)
        else:
            # Standard reward (original)
            return super()._calculate_llm_reward(action_type, arguments)
    
    def _binary_reward(self, action_type, arguments):
        """Binary reward function"""
        if self.current_email is None:
            return -1
        
        intent = self.current_email.intent
        
        # Correct actions
        correct_actions = {
            "vip_server_outage": "route_to_human",
            "angry_client_refund": "route_to_human",
            "hr_sensitive": "route_to_human",
            "spear_phishing": "route_to_human",
            "routine_password_reset": "auto_reply",
            "feature_request": "auto_reply",
            "mixed_churn": "ask_for_clarification"
        }
        
        if intent in correct_actions and action_type.value == correct_actions[intent]:
            return 1
        else:
            return -1
    
    def _sparse_reward(self, action_type, arguments):
        """Sparse reward function"""
        # Only give reward at episode end based on overall performance
        return 0  # Will be calculated at episode end
    
    def _dense_reward(self, action_type, arguments):
        """Dense reward function with more granular feedback"""
        if self.current_email is None:
            return -0.5
        
        intent = self.current_email.intent
        base_reward = super()._calculate_llm_reward(action_type, arguments)
        
        # Add additional granular feedback
        if intent == "vip_server_outage" and action_type.value == "route_to_human":
            department = arguments.get("department", "").lower()
            if "emergency" in department:
                return base_reward + 0.5  # Bonus for optimal department
            elif "it" in department or "support" in department:
                return base_reward + 0.2  # Small bonus for good department
        
        return base_reward

def run_ablation_study():
    """Run comprehensive ablation study"""
    print("🔬 Starting Ablation Study...")
    
    # Define ablation configurations
    configs = [
        # Reward function ablations
        AblationConfig("standard_reward", "standard", "standard", "base", 0.1, 100),
        AblationConfig("binary_reward", "binary", "standard", "base", 0.1, 100),
        AblationConfig("sparse_reward", "sparse", "standard", "base", 0.1, 100),
        AblationConfig("dense_reward", "dense", "standard", "base", 0.1, 100),
        
        # Prompt strategy ablations
        AblationConfig("minimal_prompt", "standard", "minimal", "base", 0.1, 100),
        AblationConfig("detailed_prompt", "standard", "detailed", "base", 0.1, 100),
        AblationConfig("no_context_prompt", "standard", "no_context", "base", 0.1, 100),
        
        # Temperature ablations
        AblationConfig("low_temp", "standard", "standard", "base", 0.01, 100),
        AblationConfig("high_temp", "standard", "standard", "base", 0.5, 100),
        AblationConfig("very_high_temp", "standard", "standard", "base", 1.0, 100),
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n🧪 Testing: {config.name}")
        
        # Create environment with modified reward function if needed
        if config.reward_function != "standard":
            env = ModifiedEnvironment(reward_function=config.reward_function)
        else:
            env = EnterpriseEmailEnv()
        
        # Create agent
        agent = AblationAgent(config)
        
        # Evaluate
        eval_results = evaluate_agent(agent, env, num_episodes=20)
        results[config.name] = eval_results
        
        print(f"   Total Reward: {eval_results.total_reward:.2f}")
        print(f"   Success Rate: {eval_results.success_rate:.2f}")
    
    return results

def analyze_ablation_results(results: Dict[str, EvaluationResults]):
    """Analyze and visualize ablation study results"""
    print("\n📊 Ablation Study Analysis:")
    
    # Group results by category
    reward_results = {k: v for k, v in results.items() if "reward" in k}
    prompt_results = {k: v for k, v in results.items() if "prompt" in k}
    temp_results = {k: v for k, v in results.items() if "temp" in k}
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Reward function comparison
    if reward_results:
        names = list(reward_results.keys())
        rewards = [reward_results[name].total_reward for name in names]
        axes[0, 0].bar(names, rewards)
        axes[0, 0].set_title('Reward Function Ablation')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Prompt strategy comparison
    if prompt_results:
        names = list(prompt_results.keys())
        rewards = [prompt_results[name].total_reward for name in names]
        axes[0, 1].bar(names, rewards)
        axes[0, 1].set_title('Prompt Strategy Ablation')
        axes[0, 1].set_ylabel('Total Reward')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Temperature comparison
    if temp_results:
        names = list(temp_results.keys())
        rewards = [temp_results[name].total_reward for name in names]
        axes[1, 0].bar(names, rewards)
        axes[1, 0].set_title('Temperature Ablation')
        axes[1, 0].set_ylabel('Total Reward')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Overall comparison
    all_names = list(results.keys())
    all_rewards = [results[name].total_reward for name in all_names]
    axes[1, 1].bar(range(len(all_names)), all_rewards)
    axes[1, 1].set_title('All Configurations')
    axes[1, 1].set_ylabel('Total Reward')
    axes[1, 1].set_xticks(range(len(all_names)))
    axes[1, 1].set_xticklabels(all_names, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('ablation_study.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results
    with open('ablation_results.json', 'w') as f:
        json.dump({name: {
            'total_reward': result.total_reward,
            'success_rate': result.success_rate,
            'avg_reward_per_step': result.avg_reward_per_step,
            'tool_usage_stats': result.tool_usage_stats
        } for name, result in results.items()}, f, indent=2)
    
    # Print summary
    print("\n🎯 Key Findings:")
    
    # Best reward function
    if reward_results:
        best_reward = max(reward_results.items(), key=lambda x: x[1].total_reward)
        print(f"   Best Reward Function: {best_reward[0]} ({best_reward[1].total_reward:.2f})")
    
    # Best prompt strategy
    if prompt_results:
        best_prompt = max(prompt_results.items(), key=lambda x: x[1].total_reward)
        print(f"   Best Prompt Strategy: {best_prompt[0]} ({best_prompt[1].total_reward:.2f})")
    
    # Best temperature
    if temp_results:
        best_temp = max(temp_results.items(), key=lambda x: x[1].total_reward)
        print(f"   Best Temperature: {best_temp[0]} ({best_temp[1].total_reward:.2f})")
    
    # Overall best
    best_overall = max(results.items(), key=lambda x: x[1].total_reward)
    print(f"   Overall Best: {best_overall[0]} ({best_overall[1].total_reward:.2f})")

def main():
    """Main function"""
    results = run_ablation_study()
    analyze_ablation_results(results)
    print("\n🎉 Ablation study completed!")
    print("📊 Results saved to: ablation_results.json")
    print("📈 Plots saved to: ablation_study.png")

if __name__ == "__main__":
    main()
