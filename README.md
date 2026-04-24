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

# Enterprise Email Triage Simulator

OpenEnv-compliant RL environment for Meta PyTorch Hackathon - Theme 3.1: Professional Tasks (Enterprise Workflows)

## Overview

This environment simulates a corporate email inbox where an AI agent must triage incoming emails based on their content and urgency. The agent learns to appropriately handle different types of emails using three available actions.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from env import EnterpriseEmailEnv

# Initialize environment
env = EnterpriseEmailEnv()

# Reset and get initial LLM-friendly observation
obs = env.reset()
print(f"Current email: {obs['current_email']['subject']}")

# Take an LLM tool call action
action = {
    "tool": "route_to_human",
    "arguments": {
        "email_id": obs['current_email']['email_id'],
        "department": "Customer Service"
    }
}
obs, reward, done, info = env.step(action)

print(f"Tool used: {info['tool_used']}")
print(f"Reward: {reward:.2f}")
```

## Environment Details

### LLM-Friendly State Space
The environment returns observations as flat dictionaries optimized for LLM prompt injection:

```python
{
    "current_email": {
        "email_id": "email_001",
        "sender": "user@company.com",
        "subject": "Password Reset Request",
        "body": "Email content...",
        "intent": "routine_password_reset",
        "priority": 2,
        "is_vip": False,
        "suggested_department": "IT"
    },
    "processed_emails": ["email_002"],
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
    "step_count": 1,
    "total_reward": 0.5,
    "last_action": "route_to_human"
}
```

### LLM Tool Call Action Space
Actions are structured dictionaries that LLMs can generate naturally:

```python
# Auto-reply with message
{
    "tool": "auto_reply",
    "arguments": {
        "email_id": "email_001",
        "message": "I can help you reset your password..."
    }
}

# Route to human with department
{
    "tool": "route_to_human",
    "arguments": {
        "email_id": "email_002", 
        "department": "Emergency Support"
    }
}

# Ask for clarification
{
    "tool": "ask_for_clarification",
    "arguments": {
        "email_id": "email_003"
    }
}
```

### Email Types
1. **Routine Password Reset** - Low priority, can be auto-replied
2. **Angry Client Refund** - High priority, must route to human
3. **VIP Server Outage** - Critical priority, must route to human immediately
4. **General Inquiry** - Medium priority, flexible handling
5. **Spam** - Low priority, can be filtered
6. **Invoice Discrepancy** - Medium priority, requires Finance routing
7. **HR Sensitive** - Critical priority, must route to HR (never auto-reply)
8. **Spear Phishing** - High priority, requires Security routing (never engage)
9. **Feature Request** - Low priority, safe for auto-reply
10. **Mixed Churn** - Medium priority, perfect for clarification tool

### Enhanced Reward Structure (LLM-Aware)
- **+1.0**: Route VIP outage/HR issues/Invoice to correct departments
- **+1.0**: Auto-reply to password reset/feature request with relevant keywords
- **+1.0**: Clarify mixed intent emails or route phishing to Security
- **+0.8**: Route angry client/churn to Customer Service/Success
- **+0.5**: Route to suboptimal but acceptable departments
- **+0.3**: Route general inquiry to appropriate departments
- **+0.2**: Route spam to Security or ask for clarification
- **+0.1**: Route emails to non-optimal departments
- **-1.0**: Auto-reply to angry client/HR issues/engage phishing
- **-0.5**: Invalid tool calls, missing arguments

### Advanced Edge Cases
The environment includes sophisticated edge cases to test AI decision-making:

1. **Invoice Discrepancy** - Tests financial acumen and routing precision
2. **HR Sensitive** - Tests empathy vs automation risk assessment
3. **Spear Phishing** - Tests security awareness and threat detection
4. **Feature Request** - Tests safe automation opportunities
5. **Mixed Churn** - Tests clarification tool usage for ambiguous intents

### Argument Validation
- **auto_reply**: Requires `email_id` and non-empty `message`
- **route_to_human**: Requires `email_id` and `department`
- **ask_for_clarification**: Requires `email_id`
- **email_id**: Must match current email ID
- **department**: Context-aware validation (e.g., VIP → Emergency Support preferred)

### AI Agent Capabilities
- **Security Awareness**: Detects phishing attempts and routes to Security
- **Empathy Assessment**: Avoids auto-replying to sensitive HR issues
- **Financial Acumen**: Routes invoice discrepancies to Finance
- **Customer Retention**: Uses clarification for ambiguous churn scenarios
- **Automation Efficiency**: Safely auto-replies to feature requests

## Docker Deployment

### Quick Start with Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t email-triage .
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key email-triage
```

### Docker Configuration

The project includes:
- **Dockerfile**: Multi-stage build with Python 3.11 slim
- **docker-compose.yml**: Full stack with Redis caching
- **.dockerignore**: Optimized build context
- **.env.example**: Environment variables template

### Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit with your API key
nano .env
```

### Production Deployment

```bash
# Production mode
docker-compose -f docker-compose.yml up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Testing

Run comprehensive test suite:

```bash
python env.py
```

This runs two test suites:
1. **Functional Tests**: Demonstrates proper LLM tool call usage
2. **Validation Tests**: Tests error handling for invalid inputs

## Architecture

- **OpenEnv Compliance**: Inherits from `oe.Environment` with proper LLM integration
- **LLM Tool Calls**: Structured action format `{"tool": "...", "arguments": {...}}`
- **Pydantic Models**: Type-safe state representation
- **Enhanced Business Logic**: Context-aware rewards based on arguments
- **Comprehensive Validation**: Argument checking with helpful error messages
- **LLM-Friendly Observations**: Flat dictionaries optimized for prompt injection

## Hackathon Requirements Met

- [x] OpenEnv-compliant environment
- [x] LLM tool call action format (not integer-based)
- [x] Three specific actions with argument validation
- [x] Five realistic dummy emails with varying intents
- [x] Enhanced business logic with department validation
- [x] LLM-friendly observations for prompt injection
- [x] Terminal-testable with comprehensive validation
- [x] Complete documentation and examples
