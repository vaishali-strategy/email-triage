import streamlit as st
import json
import random
import time

def main():
    st.set_page_config(page_title="Enterprise Email Triage", page_icon="📧", layout="wide")
    
    st.title("📧 Enterprise Email Triage Simulator")
    st.markdown("**Meta PyTorch Hackathon - Theme 3.1: Professional Tasks (Enterprise Workflows)**")
    
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
            "body": "This is outrageous! I ordered a premium widget last month and it arrived broken.",
            "intent": "angry_client_refund",
            "priority": 4,
            "is_vip": False
        },
        {
            "id": "email_003",
            "sender": "ceo.vip@fortuneglobal.com", 
            "subject": "🚨 CRITICAL: ALL SERVERS DOWN - ENTERPRISE AT RISK",
            "body": "This is an emergency. Our entire server infrastructure across all data centers is currently offline.",
            "intent": "vip_server_outage",
            "priority": 5,
            "is_vip": True
        }
    ]
    
    # Select random email
    if 'current_email' not in st.session_state:
        st.session_state.current_email = random.choice(sample_emails)
        st.session_state.total_reward = 0.0
        st.session_state.actions = []
    
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
        st.text_area("", email['body'], height=200, disabled=True, label_visibility="collapsed")
    
    with col2:
        st.subheader("🤖 Action Selection")
        
        tool = st.selectbox("Select Tool:", ["auto_reply", "route_to_human", "ask_for_clarification"])
        
        if tool == "auto_reply":
            message = st.text_area("Message:", height=100, placeholder="Enter your response...")
            if st.button("Execute Action", type="primary"):
                reward = 1.0 if email['intent'] == "routine_password_reset" else -1.0
                st.session_state.total_reward += reward
                st.session_state.actions.append({"tool": tool, "reward": reward})
                st.session_state.current_email = random.choice(sample_emails)
                st.rerun()
                
        elif tool == "route_to_human":
            department = st.selectbox("Department:", ["IT", "Customer Service", "Emergency Support", "HR", "Security"])
            if st.button("Execute Action", type="primary"):
                if email['intent'] == "vip_server_outage" and department == "Emergency Support":
                    reward = 1.0
                elif email['intent'] == "angry_client_refund" and department == "Customer Service":
                    reward = 0.8
                else:
                    reward = 0.3
                st.session_state.total_reward += reward
                st.session_state.actions.append({"tool": tool, "reward": reward})
                st.session_state.current_email = random.choice(sample_emails)
                st.rerun()
                
        else:  # ask_for_clarification
            if st.button("Execute Action", type="primary"):
                reward = 0.2
                st.session_state.total_reward += reward
                st.session_state.actions.append({"tool": tool, "reward": reward})
                st.session_state.current_email = random.choice(sample_emails)
                st.rerun()
    
    with col3:
        st.subheader("📊 Metrics & History")
        
        # Total Reward
        reward_color = "green" if st.session_state.total_reward > 0 else "red"
        st.markdown(f"### Total Reward: :{reward_color}[{st.session_state.total_reward:.2f}]")
        
        # Progress
        progress = len(st.session_state.actions) / 10.0
        st.progress(min(1.0, progress))
        st.write(f"Actions taken: {len(st.session_state.actions)}/10")
        
        # Action History
        st.markdown("---")
        st.markdown("**Action History:**")
        
        if st.session_state.actions:
            for i, action in enumerate(st.session_state.actions[-5:], 1):
                reward_color = "green" if action['reward'] > 0 else "red"
                st.write(f"{i}. {action['tool']} - :{reward_color}[{action['reward']:+.2f}]")
        else:
            st.info("No actions taken yet")
    
    # Footer
    st.markdown("---")
    st.markdown("**Instructions:**")
    st.markdown("""
    1. **Column 1**: View the current email requiring triage
    2. **Column 2**: Select a tool and execute the action
    3. **Column 3**: Monitor performance metrics and history
    4. The AI agent learns to maximize rewards by making optimal triage decisions
    """)

if __name__ == "__main__":
    main()
