import streamlit as st
import json
import random
import time
import os
from typing import Dict, Any
import openai
from groq import Groq
from reward_system import reward_system

# Set page config
st.set_page_config(page_title="Enterprise Email Triage", page_icon="📧", layout="wide")

# API Key Configuration
def configure_api_key():
    """Configure API key using Hugging Face Space secrets"""
    st.sidebar.markdown("## 🔑 API Configuration")
    
    # Initialize session state for API keys
    if 'api_key' not in st.session_state:
        # Try Hugging Face Space secrets first, then environment variables
        hf_groq_key = os.getenv("GROQ_API_KEY")  # HF Space secret
        hf_openai_key = os.getenv("OPENAI_API_KEY")  # HF Space secret
        
        if hf_groq_key:
            st.session_state.api_key = hf_groq_key
            st.session_state.api_provider = "groq"
        elif hf_openai_key:
            st.session_state.api_key = hf_openai_key
            st.session_state.api_provider = "openai"
        else:
            st.session_state.api_key = ""
            st.session_state.api_provider = "unknown"
    
    api_key = st.session_state.api_key
    
    # Detect and display API provider
    if api_key:
        if api_key.startswith("gsk_"):
            provider = "Groq"
            st.session_state.api_provider = "groq"
        elif api_key.startswith("sk-"):
            provider = "OpenAI"
            st.session_state.api_provider = "openai"
        else:
            provider = "Unknown"
            st.session_state.api_provider = "unknown"
        
        st.sidebar.success(f"✅ {provider} API Key configured (from HF Space secret)")
        st.sidebar.write("**Ready for AI actions!** 🚀")
        return True
    else:
        st.sidebar.error("❌ No API key found in HF Space secrets")
        st.sidebar.markdown("""
        **To fix this:**
        1. Go to your Hugging Face Space settings
        2. Add `GROQ_API_KEY` or `OPENAI_API_KEY` as a secret
        3. Restart the Space
        """)
        return False

def get_ai_decision(email: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """Get AI decision for email triage using OpenAI or Groq"""
    try:
        # Determine which API to use
        provider = st.session_state.get('api_provider', 'groq')  # Default to Groq
        
        prompt = f"""
You are an AI email triage assistant. Analyze the following email and decide the best action:

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

Departments for routing: IT, Customer Service, Emergency Support, HR, Security

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
"""

        if provider == "groq":
            # Use Groq API
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
        else:
            # Use OpenAI API
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
        
        decision_text = response.choices[0].message.content.strip()
        return json.loads(decision_text)
        
    except Exception as e:
        st.error(f"AI Error: {str(e)}")
        return None

def calculate_reward(email: Dict[str, Any], action: Dict[str, Any]) -> float:
    """Calculate reward using enhanced reward system"""
    reward_breakdown = reward_system.calculate_reward(email, action)
    
    # Store breakdown for display
    st.session_state.last_reward_breakdown = reward_breakdown
    
    return reward_breakdown.total_reward

def main():
    st.title("📧 Enterprise Email Triage Simulator")
    st.markdown("**Meta PyTorch Hackathon - AI-Powered Email Triage**")
    
    # Configure API
    api_configured = configure_api_key()
    
    # Sample email data
    sample_emails = [
        {
            "id": "email_001",
            "sender": "john.doe@acmecorp.com",
            "subject": "Password Reset Request",
            "body": "Hi IT team, I forgot my password for the corporate portal. Can you reset it for user john.doe?",
            "intent": "routine_password_reset",
            "priority": 2,
            "is_vip": False
        },
        {
            "id": "email_002", 
            "sender": "angrycustomer@retailclient.com",
            "subject": "REFUND MY MONEY NOW - TERRIBLE SERVICE",
            "body": "This is outrageous! I ordered a premium widget last month and it arrived broken. I want my money back immediately!",
            "intent": "angry_client_refund",
            "priority": 4,
            "is_vip": False
        },
        {
            "id": "email_003",
            "sender": "ceo.vip@fortuneglobal.com", 
            "subject": "🚨 CRITICAL: ALL SERVERS DOWN - ENTERPRISE AT RISK",
            "body": "This is an emergency. Our entire server infrastructure across all data centers is currently offline. We need immediate assistance!",
            "intent": "vip_server_outage",
            "priority": 5,
            "is_vip": True
        },
        {
            "id": "email_004",
            "sender": "employee@company.com",
            "subject": "HR Policy Question",
            "body": "I have a sensitive question about workplace accommodations and need to discuss this with HR privately.",
            "intent": "hr_sensitive",
            "priority": 3,
            "is_vip": False
        },
        {
            "id": "email_005",
            "sender": "suspicious@external.com",
            "subject": "URGENT: Verify Your Account Immediately",
            "body": "Click here to verify your account or it will be suspended. This is the final warning.",
            "intent": "spear_phishing",
            "priority": 4,
            "is_vip": False
        }
    ]
    
    # Initialize session state
    if 'current_email' not in st.session_state:
        st.session_state.current_email = random.choice(sample_emails)
        st.session_state.total_reward = 0.0
        st.session_state.actions = []
        st.session_state.email_count = 0
    
    email = st.session_state.current_email
    
    # Layout
    col1, col2, col3 = st.columns([1.2, 1, 1.2])
    
    with col1:
        st.subheader("📧 Current Email")
        st.markdown(f"**From:** {email['sender']}")
        st.markdown(f"**Subject:** {email['subject']}")
        st.markdown(f"**Priority:** {'⭐' * email['priority']} ({'VIP' if email['is_vip'] else 'Regular'})")
        st.markdown(f"**Intent:** `{email['intent']}`")
        st.markdown("---")
        st.markdown("**Body:**")
       st.text_area("Email Content", email['body'], height=200, disabled=True, label_visibility="collapsed")
    
    with col2:
        st.subheader("🤖 Action Selection")
        
        # Mode selection - default to AI-Assisted when API key is configured
        if api_configured:
            mode = st.radio("Mode:", ["AI-Assisted", "Manual"], horizontal=True, index=0)
        else:
            mode = st.radio("Mode:", ["Manual", "AI-Assisted"], horizontal=True, index=0)
        
        if mode == "Manual":
            tool = st.selectbox("Select Tool:", ["auto_reply", "route_to_human", "ask_for_clarification"])
            
            if tool == "auto_reply":
                message = st.text_area("Message:", height=100, placeholder="Enter your response...")
                if st.button("Execute Action", type="primary"):
                    if not message:
                        st.error("Please enter a message")
                    else:
                        action = {
                            "tool": tool,
                            "arguments": {
                                "email_id": email['id'],
                                "message": message
                            }
                        }
                        reward = calculate_reward(email, action)
                        st.session_state.total_reward += reward
                        st.session_state.actions.append({**action, "reward": reward, "mode": "Manual"})
                        st.session_state.current_email = random.choice(sample_emails)
                        st.session_state.email_count += 1
                        st.rerun()
                        
            elif tool == "route_to_human":
                department = st.selectbox("Department:", ["IT", "Customer Service", "Emergency Support", "HR", "Security"])
                if st.button("Execute Action", type="primary"):
                    action = {
                        "tool": tool,
                        "arguments": {
                            "email_id": email['id'],
                            "department": department
                        }
                    }
                    reward = calculate_reward(email, action)
                    st.session_state.total_reward += reward
                    st.session_state.actions.append({**action, "reward": reward, "mode": "Manual"})
                    st.session_state.current_email = random.choice(sample_emails)
                    st.session_state.email_count += 1
                    st.rerun()
                    
            else:  # ask_for_clarification
                if st.button("Execute Action", type="primary"):
                    action = {
                        "tool": tool,
                        "arguments": {
                            "email_id": email['id']
                        }
                    }
                    reward = calculate_reward(email, action)
                    st.session_state.total_reward += reward
                    st.session_state.actions.append({**action, "reward": reward, "mode": "Manual"})
                    st.session_state.current_email = random.choice(sample_emails)
                    st.session_state.email_count += 1
                    st.rerun()
        
        else:  # AI-Assisted
            if not api_configured:
                st.error("⚠️ Please configure API key to use AI features")
            else:
                st.write("🤖 AI will analyze the email and suggest the best action")
                
                if st.button("Get AI Recommendation", type="primary"):
                    with st.spinner("AI is analyzing..."):
                        api_key = st.session_state.api_key
                        if api_key:
                            ai_decision = get_ai_decision(email, api_key)
                            if ai_decision:
                                st.session_state.ai_decision = ai_decision
                                st.success("✅ AI Recommendation:")
                                st.json(ai_decision)
                        else:
                            st.error("Please enter API key in sidebar")
                
                # Show AI decision and execute button if available
                if 'ai_decision' in st.session_state and st.session_state.ai_decision:
                    if st.button("Execute AI Action", type="primary"):
                        ai_decision = st.session_state.ai_decision
                        reward = calculate_reward(email, ai_decision)
                        st.session_state.total_reward += reward
                        st.session_state.actions.append({**ai_decision, "reward": reward, "mode": "AI"})
                        st.session_state.current_email = random.choice(sample_emails)
                        st.session_state.email_count += 1
                        # Clear AI decision after execution
                        del st.session_state.ai_decision
                        st.rerun()
    
    with col3:
        st.subheader("📊 Metrics & History")
        
        # Total Reward
        reward_color = "green" if st.session_state.total_reward > 0 else "red"
        st.markdown(f"### Total Reward: :{reward_color}[{st.session_state.total_reward:.2f}]")
        
        # Progress
        progress = min(1.0, st.session_state.email_count / 10.0)
        st.progress(progress)
        st.write(f"Emails processed: {st.session_state.email_count}/10")
        
        # Action History
        st.markdown("---")
        st.markdown("**Action History:**")
        
        if st.session_state.actions:
            for i, action in enumerate(st.session_state.actions[-5:], 1):
                reward_color = "green" if action['reward'] > 0 else "red"
                mode_emoji = "🤖" if action.get('mode') == "AI" else "👤"
                st.write(f"{i}. {mode_emoji} {action['tool']} - :{reward_color}[{action['reward']:+.2f}]")
        else:
            st.info("No actions taken yet")
        
        # Reset button
        if st.button("🔄 Reset Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        # Enhanced Reward Breakdown
        if 'last_reward_breakdown' in st.session_state and st.session_state.last_reward_breakdown:
            st.markdown("---")
            st.markdown("**🔍 Reward Analysis:**")
            breakdown = st.session_state.last_reward_breakdown
            
            # Total reward
            reward_color = "green" if breakdown.total_reward > 0 else "red"
            st.markdown(f"**Total:** :{reward_color}[{breakdown.total_reward:+.2f}]")
            
            # Component breakdown
            with st.expander("View Component Breakdown"):
                for component, score in breakdown.components.items():
                    if score != 0:
                        comp_color = "green" if score > 0 else "red"
                        st.write(f"• {component.value}: :{comp_color}[{score:+.2f}]")
                
                if breakdown.flags:
                    st.markdown("**⚠️ Flags:**")
                    for flag in breakdown.flags:
                        st.write(f"• {flag}")
            
            # Reasoning
            st.markdown("**Reasoning:**")
            st.write(breakdown.reasoning)
    
    # Footer
    st.markdown("---")
    st.markdown("**Instructions:**")
    st.markdown("""
    1. **Column 1**: View the current email requiring triage
    2. **Column 2**: Choose Manual or AI-Assisted mode, then select action
    3. **Column 3**: Monitor performance metrics and action history
    4. **Sidebar**: Configure your OpenAI API key for AI features
    5. The system learns optimal triage decisions through reward feedback
    """)

if __name__ == "__main__":
    main()
