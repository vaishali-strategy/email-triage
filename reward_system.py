"""
Enhanced Reward System for Enterprise Email Triage
Following OpenEnv Hackathon guidelines for robust reward design
"""

from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import time

class RewardComponent(Enum):
    """Individual reward components for independent verification"""
    CORRECT_ACTION = "correct_action"
    FORMAT_COMPLIANCE = "format_compliance"
    ARGUMENT_VALIDITY = "argument_validity"
    DEPARTMENT_MATCH = "department_match"
    SECURITY_COMPLIANCE = "security_compliance"
    EFFICIENCY_BONUS = "efficiency_bonus"
    ANTI_CHEAT = "anti_cheat"

@dataclass
class RewardBreakdown:
    """Detailed reward breakdown for transparency and debugging"""
    total_reward: float
    components: Dict[RewardComponent, float]
    reasoning: str
    flags: List[str]  # Potential reward hacking attempts
    
class EnhancedRewardSystem:
    """
    Enhanced reward system with multiple independent reward functions
    Following hackathon best practices for robust RL training
    """
    
    def __init__(self):
        # Reward weights for different components
        self.reward_weights = {
            RewardComponent.CORRECT_ACTION: 1.0,
            RewardComponent.FORMAT_COMPLIANCE: 0.2,
            RewardComponent.ARGUMENT_VALIDITY: 0.3,
            RewardComponent.DEPARTMENT_MATCH: 0.5,
            RewardComponent.SECURITY_COMPLIANCE: 0.8,
            RewardComponent.EFFICIENCY_BONUS: 0.1,
            RewardComponent.ANTI_CHEAT: -2.0  # Penalty for cheating
        }
        
        # Track potential cheating patterns
        self.suspicious_patterns = [
            r'reset.*time',  # Time manipulation
            r'global\.',     # Global variable access
            r'__.*__',       # Magic methods
            r'cache',        # Caching attempts
            r'loop.*inf',    # Infinite loops
        ]
        
        # Valid departments for routing
        self.valid_departments = {
            "IT", "Customer Service", "Emergency Support", 
            "HR", "Security", "Finance", "Legal"
        }
        
        # Email intent to optimal action mapping
        self.optimal_actions = {
            "routine_password_reset": "auto_reply",
            "angry_client_refund": "route_to_human",
            "vip_server_outage": "route_to_human",
            "general_inquiry": "auto_reply",
            "spam": "ask_for_clarification",
            "invoice_discrepancy": "route_to_human",
            "hr_sensitive": "route_to_human",
            "spear_phishing": "route_to_human",
            "feature_request": "auto_reply",
            "mixed_churn": "ask_for_clarification"
        }
        
        # Email intent to optimal department mapping
        self.optimal_departments = {
            "routine_password_reset": "IT",
            "angry_client_refund": "Customer Service",
            "vip_server_outage": "Emergency Support",
            "general_inquiry": "Customer Service",
            "spam": "Security",
            "invoice_discrepancy": "Finance",
            "hr_sensitive": "HR",
            "spear_phishing": "Security",
            "feature_request": "IT",
            "mixed_churn": "Customer Service"
        }
    
    def calculate_reward(self, email: Dict[str, Any], action: Dict[str, Any]) -> RewardBreakdown:
        """
        Calculate comprehensive reward with multiple independent components
        """
        components = {}
        flags = []
        reasoning_parts = []
        
        # 1. Correct Action Reward
        correct_action_reward, correct_reasoning = self._reward_correct_action(email, action)
        components[RewardComponent.CORRECT_ACTION] = correct_action_reward
        reasoning_parts.append(correct_reasoning)
        
        # 2. Format Compliance Reward
        format_reward, format_reasoning = self._reward_format_compliance(action)
        components[RewardComponent.FORMAT_COMPLIANCE] = format_reward
        reasoning_parts.append(format_reasoning)
        
        # 3. Argument Validity Reward
        arg_reward, arg_reasoning = self._reward_argument_validity(action)
        components[RewardComponent.ARGUMENT_VALIDITY] = arg_reward
        reasoning_parts.append(arg_reasoning)
        
        # 4. Department Match Reward (if routing)
        dept_reward, dept_reasoning = self._reward_department_match(email, action)
        components[RewardComponent.DEPARTMENT_MATCH] = dept_reward
        reasoning_parts.append(dept_reasoning)
        
        # 5. Security Compliance Reward
        security_reward, security_reasoning = self._reward_security_compliance(email, action)
        components[RewardComponent.SECURITY_COMPLIANCE] = security_reward
        reasoning_parts.append(security_reasoning)
        
        # 6. Efficiency Bonus
        efficiency_reward, efficiency_reasoning = self._reward_efficiency(email, action)
        components[RewardComponent.EFFICIENCY_BONUS] = efficiency_reward
        reasoning_parts.append(efficiency_reasoning)
        
        # 7. Anti-Cheat Detection
        anti_cheat_reward, cheat_flags = self._detect_cheating_attempts(action)
        components[RewardComponent.ANTI_CHEAT] = anti_cheat_reward
        flags.extend(cheat_flags)
        
        # Calculate total weighted reward
        total_reward = sum(
            components[component] * self.reward_weights[component]
            for component in RewardComponent
        )
        
        # Create reasoning summary
        reasoning = "; ".join(reasoning_parts)
        if flags:
            reasoning += f" | FLAGS: {', '.join(flags)}"
        
        return RewardBreakdown(
            total_reward=total_reward,
            components=components,
            reasoning=reasoning,
            flags=flags
        )
    
    def _reward_correct_action(self, email: Dict[str, Any], action: Dict[str, Any]) -> Tuple[float, str]:
        """Reward for choosing the correct action type"""
        intent = email.get('intent', '')
        tool = action.get('tool', '')
        
        optimal_tool = self.optimal_actions.get(intent, '')
        
        if tool == optimal_tool:
            if intent == "hr_sensitive" and tool == "route_to_human":
                return 1.0, f"Correctly routed sensitive HR issue to human"
            elif intent == "spear_phishing" and tool == "route_to_human":
                return 1.0, f"Correctly identified and routed phishing attempt"
            elif intent == "vip_server_outage" and tool == "route_to_human":
                return 1.0, f"Correctly escalated VIP emergency"
            elif intent == "routine_password_reset" and tool == "auto_reply":
                return 1.0, f"Correctly auto-replied to routine request"
            else:
                return 0.8, f"Selected appropriate action for {intent}"
        elif tool == "ask_for_clarification":
            # THE FIX: Punish the agent for asking for clarification when a clear action was required.
            return -0.5, "Lazy approach: Asked for clarification instead of making a routing decision."
        else:
            return -0.8, f"Suboptimal action: {tool} for {intent}"
    
    def _reward_format_compliance(self, action: Dict[str, Any]) -> Tuple[float, str]:
        """Reward for proper action format"""
        if not isinstance(action, dict):
            return -1.0, "Invalid action format"
        
        if 'tool' not in action:
            return -0.5, "Missing tool field"
        
        if 'arguments' not in action:
            return -0.5, "Missing arguments field"
        
        if not isinstance(action['arguments'], dict):
            return -0.5, "Invalid arguments format"
        
        return 0.2, "Proper action format"
    
    def _reward_argument_validity(self, action: Dict[str, Any]) -> Tuple[float, str]:
        """Reward for valid arguments"""
        tool = action.get('tool', '')
        arguments = action.get('arguments', {})
        
        if tool == "auto_reply":
            if 'email_id' not in arguments or 'message' not in arguments:
                return -0.3, "Missing required auto_reply arguments"
            if not arguments.get('message', '').strip():
                return -0.2, "Empty message in auto_reply"
            return 0.3, "Valid auto_reply arguments"
        
        elif tool == "route_to_human":
            if 'email_id' not in arguments or 'department' not in arguments:
                return -0.3, "Missing required route_to_human arguments"
            dept = arguments.get('department', '')
            if dept not in self.valid_departments:
                return -0.2, f"Invalid department: {dept}"
            return 0.3, "Valid route_to_human arguments"
        
        elif tool == "ask_for_clarification":
            if 'email_id' not in arguments:
                return -0.3, "Missing required email_id for clarification"
            return 0.3, "Valid ask_for_clarification arguments"
        
        return -0.5, f"Unknown tool: {tool}"
    
    def _reward_department_match(self, email: Dict[str, Any], action: Dict[str, Any]) -> Tuple[float, str]:
        """Reward for routing to correct department"""
        if action.get('tool') != 'route_to_human':
            return 0.0, "Not a routing action"
        
        intent = email.get('intent', '')
        actual_dept = action.get('arguments', {}).get('department', '')
        optimal_dept = self.optimal_departments.get(intent, '')
        
        if actual_dept == optimal_dept:
            return 0.5, f"Routed to optimal department: {optimal_dept}"
        elif actual_dept in self.valid_departments:
            return 0.1, f"Routed to valid but suboptimal department: {actual_dept}"
        else:
            return -0.3, f"Routed to invalid department: {actual_dept}"
    
    def _reward_security_compliance(self, email: Dict[str, Any], action: Dict[str, Any]) -> Tuple[float, str]:
        """Reward for security-aware decisions"""
        intent = email.get('intent', '')
        tool = action.get('tool', '')
        
        # High security risk emails
        if intent == "spear_phishing":
            if tool == "auto_reply":
                return -1.0, "CRITICAL: Auto-replied to phishing attempt"
            elif tool == "ask_for_clarification":
                return -0.8, "CRITICAL: Wasting time on phishing attempt instead of routing to Security"
            elif tool == "route_to_human" and action.get('arguments', {}).get('department') == 'Security':
                return 0.8, "Correctly routed phishing to Security"
        
        elif intent == "hr_sensitive":
            if tool == "auto_reply":
                return -0.8, "Auto-replied to sensitive HR issue"
            elif tool == "route_to_human" and action.get('arguments', {}).get('department') == 'HR':
                return 0.8, "Correctly routed HR issue to HR"
        
        elif intent == "vip_server_outage":
            if tool == "ask_for_clarification":
                return -0.8, "CRITICAL: Delayed VIP emergency by asking for clarification"
            if tool == "route_to_human":
                dept = action.get('arguments', {}).get('department', '')
                if dept == "Emergency Support":
                    return 0.8, "Correctly escalated VIP emergency"
                elif dept in self.valid_departments:
                    return 0.3, f"Routed VIP issue to {dept} (not emergency)"
        
        return 0.0, "No security considerations"
        
    
    def _reward_efficiency(self, email: Dict[str, Any], action: Dict[str, Any]) -> Tuple[float, str]:
        """Reward for efficient handling"""
        intent = email.get('intent', '')
        tool = action.get('tool', '')
        
        # Bonus for quick resolution of simple cases
        if intent == "routine_password_reset" and tool == "auto_reply":
            return 0.1, "Efficient handling of routine request"
        
        elif intent == "feature_request" and tool == "auto_reply":
            return 0.1, "Efficient handling of feature request"
        
        # Penalty for over-complicating simple cases
        elif intent == "routine_password_reset" and tool == "route_to_human":
            return -0.1, "Over-complicated handling of routine request"
        
        return 0.0, "Standard efficiency"
    
    def _detect_cheating_attempts(self, action: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Detect potential reward hacking attempts"""
        flags = []
        penalty = 0.0
        
        # Check action string for suspicious patterns
        action_str = str(action).lower()
        
        for pattern in self.suspicious_patterns:
            if re.search(pattern, action_str, re.IGNORECASE):
                flags.append(f"Suspicious pattern: {pattern}")
                penalty -= 0.5
        
        # Check for empty or None values that might indicate cheating
        arguments = action.get('arguments', {})
        for key, value in arguments.items():
            if value is None or (isinstance(value, str) and not value.strip()):
                flags.append(f"Empty argument: {key}")
                penalty -= 0.2
        
        # Check for unusually fast responses (potential caching)
        if hasattr(self, '_last_action_time'):
            time_diff = time.time() - self._last_action_time
            if time_diff < 0.01:  # Less than 10ms
                flags.append("Suspiciously fast response")
                penalty -= 0.3
        
        self._last_action_time = time.time()
        
        return penalty, flags
    
    def get_reward_summary(self, breakdown: RewardBreakdown) -> Dict[str, Any]:
        """Get detailed reward summary for monitoring"""
        return {
            "total_reward": breakdown.total_reward,
            "component_scores": {
                comp.value: score for comp, score in breakdown.components.items()
            },
            "flags": breakdown.flags,
            "reasoning": breakdown.reasoning,
            "weighted_components": {
                comp.value: score * self.reward_weights[comp]
                for comp, score in breakdown.components.items()
            }
        }

# Global reward system instance
reward_system = EnhancedRewardSystem()
