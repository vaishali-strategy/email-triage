"""
Evaluation script for Enterprise Email Triage Agent

This script evaluates the trained agent against baselines and provides
comprehensive metrics for the hackathon submission.
"""

import numpy as np
import json
import time
from typing import Dict, List, Tuple
from env import EnterpriseEmailEnv
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import pandas as pd

@dataclass
class EvaluationResults:
    """Results from agent evaluation"""
    total_reward: float
    success_rate: float
    avg_reward_per_step: float
    tool_usage_stats: Dict[str, int]
    episode_rewards: List[float]
    success_by_intent: Dict[str, float]
    total_episodes: int
    total_steps: int

class BaselineAgent:
    """Baseline agents for comparison"""
    
    def __init__(self, strategy: str = "random"):
        self.strategy = strategy
    
    def select_action(self, observation: Dict) -> Dict:
        """Select action based on strategy"""
        email = observation['current_email']
        
        if self.strategy == "random":
            return self._random_action(email)
        elif self.strategy == "rule_based":
            return self._rule_based_action(email)
        elif self.strategy == "always_route":
            return self._always_route_action(email)
        
        return self._random_action(email)
    
    def _random_action(self, email) -> Dict:
        """Random tool selection"""
        tools = ["auto_reply", "route_to_human", "ask_for_clarification"]
        tool = np.random.choice(tools)
        
        if tool == "auto_reply":
            return {
                "tool": tool,
                "arguments": {
                    "email_id": email['email_id'],
                    "message": "This is an automated response."
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
    
    def _rule_based_action(self, email) -> Dict:
        """Simple rule-based agent"""
        intent = email['intent']
        priority = email['priority']
        
        if intent in ["vip_server_outage", "angry_client_refund", "hr_sensitive", "spear_phishing"]:
            # Route urgent issues to human
            department_map = {
                "vip_server_outage": "Emergency Support",
                "angry_client_refund": "Customer Service", 
                "hr_sensitive": "HR",
                "spear_phishing": "Security"
            }
            return {
                "tool": "route_to_human",
                "arguments": {
                    "email_id": email['email_id'],
                    "department": department_map.get(intent, "General")
                }
            }
        elif intent in ["routine_password_reset", "feature_request"]:
            # Auto-reply routine requests
            return {
                "tool": "auto_reply",
                "arguments": {
                    "email_id": email['email_id'],
                    "message": "Thank you for your message. We'll process this request."
                }
            }
        else:
            # Ask for clarification on others
            return {
                "tool": "ask_for_clarification",
                "arguments": {"email_id": email['email_id']}
            }
    
    def _always_route_action(self, email) -> Dict:
        """Always route to human"""
        departments = ["IT", "Customer Service", "Security", "HR", "Finance"]
        return {
            "tool": "route_to_human",
            "arguments": {
                "email_id": email['email_id'],
                "department": np.random.choice(departments)
            }
        }

def evaluate_agent(agent, env: EnterpriseEmailEnv, num_episodes: int = 10) -> EvaluationResults:
    """Evaluate an agent over multiple episodes"""
    episode_rewards = []
    tool_usage = {"auto_reply": 0, "route_to_human": 0, "ask_for_clarification": 0}
    intent_success = {}
    total_steps = 0
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action
            if hasattr(agent, 'select_action'):
                action = agent.select_action(obs)
            else:
                # Assume it's a trained agent with step method
                action = agent.select_action(obs)
            
            # Take step
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            total_steps += 1
            
            # Track tool usage
            tool = action.get("tool", "unknown")
            if tool in tool_usage:
                tool_usage[tool] += 1
            
            # Track intent success
            if "email_intent" in info:
                intent = info["email_intent"]
                if intent not in intent_success:
                    intent_success[intent] = []
                intent_success[intent].append(reward)
    
        episode_rewards.append(episode_reward)
    
    # Calculate success metrics
    success_by_intent = {}
    for intent, rewards in intent_success.items():
        success_by_intent[intent] = np.mean(rewards) if rewards else 0.0
    
    return EvaluationResults(
        total_reward=np.sum(episode_rewards),
        success_rate=np.mean([r > 0 for r in episode_rewards]),
        avg_reward_per_step=np.sum(episode_rewards) / total_steps if total_steps > 0 else 0,
        tool_usage_stats=tool_usage,
        episode_rewards=episode_rewards,
        success_by_intent=success_by_intent,
        total_episodes=num_episodes,
        total_steps=total_steps
    )

def compare_agents(env: EnterpriseEmailEnv, num_episodes: int = 20) -> Dict[str, EvaluationResults]:
    """Compare different agents"""
    results = {}
    
    # Test baseline agents
    baseline_strategies = ["random", "rule_based", "always_route"]
    
    for strategy in baseline_strategies:
        print(f"Evaluating {strategy} baseline...")
        agent = BaselineAgent(strategy)
        results[strategy] = evaluate_agent(agent, env, num_episodes)
    
    return results

def generate_evaluation_report(results: Dict[str, EvaluationResults], save_path: str = "evaluation_report.json"):
    """Generate comprehensive evaluation report"""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "agents": {}
    }
    
    for agent_name, result in results.items():
        report["agents"][agent_name] = {
            "total_reward": result.total_reward,
            "success_rate": result.success_rate,
            "avg_reward_per_step": result.avg_reward_per_step,
            "tool_usage_stats": result.tool_usage_stats,
            "episode_rewards": result.episode_rewards,
            "success_by_intent": result.success_by_intent,
            "total_episodes": result.total_episodes,
            "total_steps": result.total_steps
        }
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

def plot_results(results: Dict[str, EvaluationResults], save_path: str = "evaluation_plots.png"):
    """Create visualization of evaluation results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Total Rewards Comparison
    agents = list(results.keys())
    rewards = [results[agent].total_reward for agent in agents]
    axes[0, 0].bar(agents, rewards)
    axes[0, 0].set_title('Total Rewards Comparison')
    axes[0, 0].set_ylabel('Total Reward')
    
    # 2. Success Rate Comparison
    success_rates = [results[agent].success_rate for agent in agents]
    axes[0, 1].bar(agents, success_rates)
    axes[0, 1].set_title('Success Rate Comparison')
    axes[0, 1].set_ylabel('Success Rate')
    
    # 3. Episode Rewards Distribution
    for agent in agents:
        axes[1, 0].plot(results[agent].episode_rewards, label=agent, alpha=0.7)
    axes[1, 0].set_title('Episode Rewards Over Time')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].legend()
    
    # 4. Tool Usage Comparison
    tools = ["auto_reply", "route_to_human", "ask_for_clarification"]
    x = np.arange(len(tools))
    width = 0.25
    
    for i, agent in enumerate(agents):
        usage = [results[agent].tool_usage_stats.get(tool, 0) for tool in tools]
        axes[1, 1].bar(x + i*width, usage, width, label=agent)
    
    axes[1, 1].set_title('Tool Usage Comparison')
    axes[1, 1].set_xlabel('Tool')
    axes[1, 1].set_ylabel('Usage Count')
    axes[1, 1].set_xticks(x + width)
    axes[1, 1].set_xticklabels(tools)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main evaluation function"""
    print("🔍 Starting Enterprise Email Triage Evaluation...")
    
    # Initialize environment
    env = EnterpriseEmailEnv()
    print(f"📧 Environment loaded with {len(env.emails)} emails")
    
    # Compare agents
    results = compare_agents(env, num_episodes=20)
    
    # Generate report
    report = generate_evaluation_report(results)
    print("📊 Evaluation report generated: evaluation_report.json")
    
    # Create plots
    plot_results(results)
    print("📈 Evaluation plots saved: evaluation_plots.png")
    
    # Print summary
    print("\n🎯 Evaluation Summary:")
    for agent_name, result in results.items():
        print(f"{agent_name:15}: Reward={result.total_reward:7.2f}, Success={result.success_rate:5.2f}")
    
    return results

if __name__ == "__main__":
    results = main()
