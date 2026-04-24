"""
Enterprise Email Triage Simulator - OpenEnv Compliant RL Environment

This environment simulates a corporate email inbox where an AI agent must triage
incoming emails based on their content and urgency. The agent must learn to
appropriately handle different types of emails using three available actions.

Author: Senior Python AI Engineer
Purpose: Meta PyTorch Hackathon - Theme 3.1: Professional Tasks (Enterprise Workflows)
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random
import json
from pydantic import BaseModel, Field
import openenv.core as oe
from typing import Generic, TypeVar
from reward_system import reward_system, RewardBreakdown


class EmailIntent(Enum):
    """Enumeration of email intent types for classification."""
    ROUTINE_PASSWORD_RESET = "routine_password_reset"
    ANGRY_CLIENT_REFUND = "angry_client_refund"
    VIP_SERVER_OUTAGE = "vip_server_outage"
    GENERAL_INQUIRY = "general_inquiry"
    SPAM = "spam"
    INVOICE_DISCREPANCY = "invoice_discrepancy"
    HR_SENSITIVE = "hr_sensitive"
    SPEAR_PHISHING = "spear_phishing"
    FEATURE_REQUEST = "feature_request"
    MIXED_CHURN = "mixed_churn"


class ActionType(Enum):
    """Enumeration of available agent actions."""
    AUTO_REPLY = "auto_reply"
    ROUTE_TO_HUMAN = "route_to_human"
    ASK_FOR_CLARIFICATION = "ask_for_clarification"


@dataclass
class Email:
    """Represents an email in the corporate inbox."""
    id: str
    sender: str
    subject: str
    body: str
    intent: EmailIntent
    priority: int  # 1-5 scale, 5 being highest
    is_vip: bool = False
    department: Optional[str] = None


class EmailState(BaseModel):
    """State representation for the RL environment."""
    current_email: Optional[Dict[str, Any]] = None
    processed_emails: List[str] = Field(default_factory=list)
    available_actions: List[str] = Field(default_factory=lambda: [action.value for action in ActionType])
    step_count: int = 0
    total_reward: float = 0.0


# Type variables for Generic Environment
ActType = TypeVar('ActType')
ObsType = TypeVar('ObsType') 
StateType = TypeVar('StateType')


class EnterpriseEmailEnv(oe.Environment[ActType, ObsType, StateType]):
    """
    OpenEnv-compliant enterprise email triage environment.
    
    The agent must learn to appropriately handle different types of emails
    using three available actions: auto_reply, route_to_human, and ask_for_clarification.
    
    State Space:
        - Current email details (sender, subject, body, metadata)
        - History of processed emails
        - Available actions
    
    Action Space:
        - 0: auto_reply(email_id, message)
        - 1: route_to_human(email_id, department)
        - 2: ask_for_clarification(email_id)
    
    Reward Structure:
        - +1.0: Correctly route VIP server outage to human
        - +1.0: Correctly auto-reply to routine password reset
        - -1.0: Auto-reply to angry client or VIP (must route to human)
        - -0.5: Invalid tool selection or looping behavior
    """
    
    def __init__(self, max_steps: int = 10):
        """
        Initialize the enterprise email environment.
        
        Args:
            max_steps: Maximum number of steps per episode
        """
        super().__init__()
        self.max_steps = max_steps
        self.current_step = 0
        self.emails = self._load_emails()
        self.processed_emails = []
        self.current_email = None
        self.total_reward = 0.0
        self.last_action = None
        self.action_history = []
        
        # Note: OpenEnv Environment doesn't use traditional gym spaces
        # State is managed through the state property and observations returned from reset/step
        
    def _load_emails(self) -> List[Email]:
        """
        Load emails from dataset.json file, with fallback to hardcoded emails.
        
        Returns:
            List of Email objects representing the corporate inbox
        """
        try:
            with open("dataset.json", "r") as f:
                data = json.load(f)
                emails = []
                
                for email_data in data:
                    # Convert string intent to EmailIntent enum
                    intent_str = email_data.get("intent", "general_inquiry")
                    try:
                        intent = EmailIntent(intent_str)
                    except ValueError:
                        print(f"Warning: Unknown intent '{intent_str}', using general_inquiry")
                        intent = EmailIntent.GENERAL_INQUIRY
                    
                    email = Email(
                        id=email_data["id"],
                        sender=email_data["sender"],
                        subject=email_data["subject"],
                        body=email_data["body"],
                        intent=intent,
                        priority=email_data.get("priority", 3),
                        is_vip=email_data.get("is_vip", False),
                        department=email_data.get("department", "General")
                    )
                    emails.append(email)
                
                print(f"✅ Successfully loaded {len(emails)} emails from dataset.json")
                return emails
                
        except FileNotFoundError:
            print("⚠️  Warning: dataset.json not found, using fallback emails")
            return self._create_fallback_emails()
        except json.JSONDecodeError as e:
            print(f"⚠️  Warning: Error parsing dataset.json: {e}, using fallback emails")
            return self._create_fallback_emails()
        except Exception as e:
            print(f"⚠️  Warning: Error loading dataset.json: {e}, using fallback emails")
            return self._create_fallback_emails()

    def _create_fallback_emails(self) -> List[Email]:
        """
        Create hardcoded list of 5 realistic dummy emails with varying intents.
        
        Returns:
            List of Email objects representing the corporate inbox
        """
        emails = [
            Email(
                id="email_001",
                sender="john.doe@acmecorp.com",
                subject="Password Reset Request",
                body="Hi IT team,\n\nI forgot my password for the corporate portal. Can you reset it for user john.doe? I need access to complete my quarterly reports.\n\nThanks,\nJohn",
                intent=EmailIntent.ROUTINE_PASSWORD_RESET,
                priority=2,
                is_vip=False,
                department="IT"
            ),
            
            # Angry Client Refund Request
            Email(
                id="email_002",
                sender="angrycustomer@retailclient.com",
                subject="REFUND MY MONEY NOW - TERRIBLE SERVICE",
                body="This is outrageous! I ordered a premium widget last month and it arrived broken. Your customer service has been useless and I demand a full refund immediately. I've been a loyal customer for 5 years and this is how you treat me?\n\nI want my money back NOW or I'm switching to your competitor.\n\nFurious,\nMike Chen",
                intent=EmailIntent.ANGRY_CLIENT_REFUND,
                priority=4,
                is_vip=False,
                department="Customer Service"
            ),
            
            # VIP Server Outage (Critical)
            Email(
                id="email_003",
                sender="ceo.vip@fortuneglobal.com",
                subject="🚨 CRITICAL: ALL SERVERS DOWN - ENTERPRISE AT RISK",
                body="This is an emergency. Our entire server infrastructure across all data centers is currently offline. This is affecting millions of users and causing massive revenue loss. We need immediate technical intervention.\n\nThis is our highest priority incident - please escalate to emergency technical teams immediately.\n\nRegards,\nGlobal Infrastructure Team",
                intent=EmailIntent.VIP_SERVER_OUTAGE,
                priority=5,
                is_vip=True,
                department="Emergency Support"
            ),
            
            # General Inquiry
            Email(
                id="email_004",
                sender="potential.client@startup.com",
                subject="Question about partnership opportunities",
                body="Hello,\n\nI'm impressed with your company's work in the enterprise space and would like to explore potential partnership opportunities. Could you provide more information about your B2B collaboration programs?\n\nBest regards,\nSarah Kim",
                intent=EmailIntent.GENERAL_INQUIRY,
                priority=3,
                is_vip=False,
                department="Business Development"
            ),
            
            # Spam
            Email(
                id="email_005",
                sender="winner@lottery-scam.net",
                subject="You've won $1,000,000!!!",
                body="CONGRATULATIONS! You are our lucky winner! Click here immediately to claim your $1,000,000 prize. This offer expires in 24 hours. ACT FAST!\n\nURGENT! URGENT! URGENT!\n\nClaim now!",
                intent=EmailIntent.SPAM,
                priority=1,
                is_vip=False,
                department="Security"
            ),
            
            # EDGE CASE 2: The "Highly Sensitive HR Issue" (Tests empathy vs. automation risk)
            # A massive enterprise risk. If the AI auto-replies to this, it's a disaster. Must go to HR immediately.
            Email(
                id="email_007",
                sender="sarah.jenkins@acmecorp.com",
                subject="CONFIDENTIAL: Formal complaint regarding workplace conduct",
                body="I am writing to formally report an incident of inappropriate conduct by my team lead during yesterday's offsite. I have documented the comments and feel extremely uncomfortable returning to the office. I need to speak with an employee relations representative as soon as possible.\n\nSarah Jenkins",
                intent=EmailIntent.HR_SENSITIVE,
                priority=5,
                is_vip=False,
                department="Human Resources"
            ),
            
            # EDGE CASE 3: The "Spear Phishing" attack (Tests security awareness)
            # It looks like a routine IT ticket, but the domain is fake ("acmecorp-secure.com"). 
            Email(
                id="email_008",
                sender="admin@acmecorp-secure.com",
                subject="ACTION REQUIRED: Update your Okta SSO Credentials",
                body="Dear Employee,\n\nWe are migrating our enterprise single sign-on system. Your current Okta password will expire in 2 hours. Please log in immediately at http://acmecorp-secure-login.net/auth to retain access to your corporate email and GitHub repositories.\n\nIT Support Team",
                intent=EmailIntent.SPEAR_PHISHING,
                priority=4,
                is_vip=False,
                department="Security"
            ),
            
            # EDGE CASE 4: The "Low Priority Feedback" (Tests safe auto-reply usage)
            # This is the perfect candidate for the AI to safely use the `auto_reply` tool to save human time.
            Email(
                id="email_009",
                sender="beta.tester@client.com",
                subject="Suggestion for the new analytics dashboard",
                body="Hey team, I've been using the new beta version of the analytics dashboard. It's pretty good, but it would be awesome if you added a dark mode toggle and the ability to export the charts directly to PDF. Just a thought!\n\nCheers.",
                intent=EmailIntent.FEATURE_REQUEST,
                priority=1,
                is_vip=False,
                department="Product"
            ),
            
            # EDGE CASE 5: The "Ambiguous Multi-Intent" (Tests the Ask For Clarification tool)
            # The client wants to downgrade (Sales/Retention) but also has a technical question (IT/Support). 
            Email(
                id="email_010",
                sender="confused.user@smallbiz.com",
                subject="Downgrading my account + question about data export",
                body="Hello, we are cutting costs and want to drop down from the Enterprise tier to the Basic tier next month. Before we do that, does the Basic tier still allow us to use the API to export our historical data? If not, we might just cancel entirely. Let me know.",
                intent=EmailIntent.MIXED_CHURN,
                priority=3,
                is_vip=False,
                department="Customer Success"
            )
        ]
        
        return emails
    
    @property
    def state(self) -> Dict[str, Any]:
        """
        Get the current environment state (LLM-friendly).
        
        Returns:
            Current environment state as flat dictionary
        """
        return self._get_llm_observation()

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Optional seed for random number generation
            episode_id: Optional episode ID for tracking
        
        Returns:
            Initial state observation
        """
        self.current_step = 0
        self.processed_emails = []
        self.total_reward = 0.0
        self.last_action = None
        self.action_history = []
        
        # Reset emails and select first one
        self.emails = self._load_emails()
        self.current_email = random.choice(self.emails)
        
        return self._get_llm_observation()
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment using LLM tool call format.
        
        Args:
            action: Dictionary containing tool name and arguments
                   Format: {"tool": "tool_name", "arguments": {...}}
            
        Returns:
            Tuple of (observation, reward, done, info) where observation is LLM-friendly
        """
        if self.current_email is None:
            raise ValueError("No current email available. Call reset() first.")
        
        if self.current_email.id in self.processed_emails:
            raise ValueError("Current email has already been processed.")
        
        # Capture current email intent before updating current_email
        processed_intent = self.current_email.intent.value if self.current_email else None
        
        # Parse LLM tool call
        tool_name = action.get("tool")
        arguments = action.get("arguments", {})
        
        # Validate tool name
        valid_tools = [action.value for action in ActionType]
        if tool_name not in valid_tools:
            reward = -0.5
            done = True
            info = {"error": f"Invalid tool '{tool_name}'. Valid tools: {valid_tools}"}
            return self._get_llm_observation(), reward, done, info
        
        # Convert to ActionType
        action_type = ActionType(tool_name)
        
        # Validate arguments based on tool type
        validation_result = self._validate_tool_arguments(action_type, arguments)
        if not validation_result["valid"]:
            reward = -0.5
            done = True
            info = {"error": validation_result["error"]}
            return self._get_llm_observation(), reward, done, info
        
        # Calculate reward with enhanced business logic
        reward = self._calculate_llm_reward(action_type, arguments)
        
        # Update state
        self.processed_emails.append(self.current_email.id)
        self.action_history.append(action_type.value)
        self.total_reward += reward
        self.last_action = action_type
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.max_steps or len(self.processed_emails) >= len(self.emails)
        
        # Select next email if available
        if not done:
            remaining_emails = [email for email in self.emails if email.id not in self.processed_emails]
            if remaining_emails:
                self.current_email = random.choice(remaining_emails)
            else:
                done = True
        
        info = {
            "tool_used": action_type.value,
            "arguments": arguments,
            "email_intent": processed_intent,
            "step": self.current_step,
            "total_reward": self.total_reward,
            "validation_passed": True
        }
        
        return self._get_llm_observation(), reward, done, info
    
    def _validate_tool_arguments(self, action_type: ActionType, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate tool arguments based on tool type and current email context.
        
        Args:
            action_type: The type of action being performed
            arguments: Dictionary of arguments provided by the LLM
            
        Returns:
            Dictionary with validation result and error message if invalid
        """
        if action_type == ActionType.AUTO_REPLY:
            # Auto-reply requires a message argument
            if "message" not in arguments or not arguments["message"].strip():
                return {
                    "valid": False,
                    "error": "auto_reply tool requires a 'message' argument with non-empty content"
                }
            
            # Check if email_id is provided and matches current email
            if "email_id" in arguments and arguments["email_id"] != self.current_email.id:
                return {
                    "valid": False,
                    "error": f"email_id '{arguments['email_id']}' does not match current email '{self.current_email.id}'"
                }
        
        elif action_type == ActionType.ROUTE_TO_HUMAN:
            # Route to human requires a department argument
            if "department" not in arguments or not arguments["department"].strip():
                return {
                    "valid": False,
                    "error": "route_to_human tool requires a 'department' argument"
                }
            
            # Check if email_id is provided and matches current email
            if "email_id" in arguments and arguments["email_id"] != self.current_email.id:
                return {
                    "valid": False,
                    "error": f"email_id '{arguments['email_id']}' does not match current email '{self.current_email.id}'"
                }
        
        elif action_type == ActionType.ASK_FOR_CLARIFICATION:
            # Ask for clarification requires email_id
            if "email_id" in arguments and arguments["email_id"] != self.current_email.id:
                return {
                    "valid": False,
                    "error": f"email_id '{arguments['email_id']}' does not match current email '{self.current_email.id}'"
                }
        
        return {"valid": True, "error": None}
    
    def _calculate_llm_reward(self, action_type: ActionType, arguments: Dict[str, Any]) -> float:
        """
        Calculate reward using enhanced reward system with multiple independent components.
        
        Args:
            action_type: The type of action being performed
            arguments: Dictionary of arguments provided by the LLM
            
        Returns:
            Total reward value from enhanced reward system
        """
        if self.current_email is None:
            return -0.5
        
        # Convert email to dict format for reward system
        email_dict = {
            "id": self.current_email.id,
            "sender": self.current_email.sender,
            "subject": self.current_email.subject,
            "body": self.current_email.body,
            "intent": self.current_email.intent.value,
            "priority": self.current_email.priority,
            "is_vip": self.current_email.is_vip
        }
        
        # Convert action to dict format for reward system
        action_dict = {
            "tool": action_type.value,
            "arguments": arguments
        }
        
        # Use enhanced reward system
        reward_breakdown = reward_system.calculate_reward(email_dict, action_dict)
        
        # Store breakdown for debugging/monitoring
        if not hasattr(self, 'last_reward_breakdown'):
            self.last_reward_breakdown = reward_breakdown
        else:
            self.last_reward_breakdown = reward_breakdown
        
        return reward_breakdown.total_reward
    
    def get_last_reward_breakdown(self) -> Optional[RewardBreakdown]:
        """Get the detailed breakdown of the last reward calculation"""
        return getattr(self, 'last_reward_breakdown', None)
    
    def _get_llm_observation(self) -> Dict[str, Any]:
        """
        Get LLM-friendly observation as a flat dictionary.
        
        Returns:
            Dictionary with current state information optimized for LLM consumption
        """
        email_data = {}
        if self.current_email:
            email_data = {
                "email_id": self.current_email.id,
                "sender": self.current_email.sender,
                "subject": self.current_email.subject,
                "body": self.current_email.body,
                "intent": self.current_email.intent.value,
                "priority": self.current_email.priority,
                "is_vip": self.current_email.is_vip,
                "suggested_department": self.current_email.department
            }
        
        return {
            "current_email": email_data,
            "processed_emails": self.processed_emails.copy(),
            "available_tools": [
                {
                    "name": "auto_reply",
                    "description": "Automatically reply to an email",
                    "parameters": {
                        "email_id": "string (required)",
                        "message": "string (required)"
                    }
                },
                {
                    "name": "route_to_human", 
                    "description": "Route email to human agent",
                    "parameters": {
                        "email_id": "string (required)",
                        "department": "string (required)"
                    }
                },
                {
                    "name": "ask_for_clarification",
                    "description": "Request more information about email",
                    "parameters": {
                        "email_id": "string (required)"
                    }
                }
            ],
            "step_count": self.current_step,
            "total_reward": self.total_reward,
            "last_action": self.last_action.value if self.last_action else None
        }
    
        
        
    def render(self, mode: str = 'human') -> None:
        """
        Render the current state of the environment.
        
        Args:
            mode: Rendering mode ('human' for text output)
        """
        if mode != 'human':
            return
        
        print("\n" + "="*80)
        print(f"ENTERPRISE EMAIL TRIAGE SIMULATOR - Step {self.current_step}")
        print("="*80)
        
        if self.current_email:
            print(f"\nCURRENT EMAIL:")
            print(f"From: {self.current_email.sender}")
            print(f"Subject: {self.current_email.subject}")
            print(f"Priority: {self.current_email.priority}/5 {'(VIP)' if self.current_email.is_vip else ''}")
            print(f"Body: {self.current_email.body[:200]}...")
            print(f"Intent: {self.current_email.intent.value}")
        
        print(f"\nProcessed Emails: {len(self.processed_emails)}/{len(self.emails)}")
        print(f"Total Reward: {self.total_reward:.2f}")
        print(f"Last Action: {self.last_action.value if self.last_action else 'None'}")
        print("="*80)
    
        
    def close(self) -> None:
        """Clean up environment resources."""
        pass


# Testing function for terminal validation
def test_environment():
    """
    Test function to validate the environment works correctly with LLM tool calls.
    Run this function to test the environment in a terminal.
    """
    print("Initializing Enterprise Email Triage Environment...")
    env = EnterpriseEmailEnv()
    
    print("\nTesting environment reset...")
    obs = env.reset()
    print("Initial observation:")
    print(f"Current email: {obs['current_email']['subject']}")
    print(f"Available tools: {[tool['name'] for tool in obs['available_tools']]}")
    
    print("\nTesting LLM tool call actions...")
    
    # Test 1: Auto-reply to password reset
    print("\n=== Test 1: Auto-reply to password reset ===")
    action1 = {
        "tool": "auto_reply",
        "arguments": {
            "email_id": obs['current_email']['email_id'],
            "message": "I can help you reset your password. Please follow our standard password reset procedure."
        }
    }
    obs, reward, done, info = env.step(action1)
    print(f"Tool: {info['tool_used']}")
    print(f"Reward: {reward:.2f}")
    print(f"Done: {done}")
    
    if not done:
        # Test 2: Route VIP outage to emergency support
        print("\n=== Test 2: Route VIP outage to emergency support ===")
        action2 = {
            "tool": "route_to_human",
            "arguments": {
                "email_id": obs['current_email']['email_id'],
                "department": "Emergency Support"
            }
        }
        obs, reward, done, info = env.step(action2)
        print(f"Tool: {info['tool_used']}")
        print(f"Department: {info['arguments']['department']}")
        print(f"Reward: {reward:.2f}")
        print(f"Done: {done}")
    
    if not done:
        # Test 3: Route angry client to customer service
        print("\n=== Test 3: Route angry client to customer service ===")
        action3 = {
            "tool": "route_to_human",
            "arguments": {
                "email_id": obs['current_email']['email_id'],
                "department": "Customer Service"
            }
        }
        obs, reward, done, info = env.step(action3)
        print(f"Tool: {info['tool_used']}")
        print(f"Department: {info['arguments']['department']}")
        print(f"Reward: {reward:.2f}")
        print(f"Done: {done}")
    
    if not done:
        # Test 4: Ask for clarification on general inquiry
        print("\n=== Test 4: Ask for clarification on general inquiry ===")
        action4 = {
            "tool": "ask_for_clarification",
            "arguments": {
                "email_id": obs['current_email']['email_id']
            }
        }
        obs, reward, done, info = env.step(action4)
        print(f"Tool: {info['tool_used']}")
        print(f"Reward: {reward:.2f}")
        print(f"Done: {done}")
    
    if not done:
        # Test 5: Route spam to security
        print("\n=== Test 5: Route spam to security ===")
        action5 = {
            "tool": "route_to_human",
            "arguments": {
                "email_id": obs['current_email']['email_id'],
                "department": "Security"
            }
        }
        obs, reward, done, info = env.step(action5)
        print(f"Tool: {info['tool_used']}")
        print(f"Department: {info['arguments']['department']}")
        print(f"Reward: {reward:.2f}")
        print(f"Done: {done}")
    
    print(f"\nFinal total reward: {env.total_reward:.2f}")
    env.close()
    print("Environment test completed successfully!")


def test_validation_errors():
    """
    Test function to validate error handling for invalid tool calls.
    """
    print("\n" + "="*60)
    print("TESTING VALIDATION ERRORS")
    print("="*60)
    
    env = EnterpriseEmailEnv()
    obs = env.reset()
    
    # Test 1: Invalid tool name
    print("\n=== Test 1: Invalid tool name ===")
    invalid_action = {
        "tool": "invalid_tool",
        "arguments": {"email_id": obs['current_email']['email_id']}
    }
    obs, reward, done, info = env.step(invalid_action)
    print(f"Error: {info['error']}")
    print(f"Reward: {reward:.2f}")
    
    # Test 2: Missing required arguments
    print("\n=== Test 2: Auto-reply without message ===")
    env.reset()
    obs = env.reset()
    invalid_action2 = {
        "tool": "auto_reply",
        "arguments": {"email_id": obs['current_email']['email_id']}
    }
    obs, reward, done, info = env.step(invalid_action2)
    print(f"Error: {info['error']}")
    print(f"Reward: {reward:.2f}")
    
    # Test 3: Wrong email_id
    print("\n=== Test 3: Wrong email_id ===")
    env.reset()
    obs = env.reset()
    invalid_action3 = {
        "tool": "route_to_human",
        "arguments": {
            "email_id": "wrong_email_id",
            "department": "IT"
        }
    }
    obs, reward, done, info = env.step(invalid_action3)
    print(f"Error: {info['error']}")
    print(f"Reward: {reward:.2f}")
    
    env.close()
    print("\nValidation error tests completed!")


if __name__ == "__main__":
    test_environment()
    test_validation_errors()
