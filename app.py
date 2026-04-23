import streamlit as st
import json
import random
import time

st.set_page_config(page_title="📧 Enterprise Email Triage", layout="wide")

def main():
    st.title("📧 Enterprise Email Triage Simulator")
    st.markdown("**Meta PyTorch Hackathon - Theme 3.1: Professional Tasks**")
    
    # Sample emails
    emails = [
        {
            "id": "email_001",
            "sender": "john.doe@acmecorp.com",
            "subject": "Password Reset Request",
            "body": "Hi IT team, I forgot my password. Can you help me reset it?",
            "intent": "routine_password_reset",
            "priority": 2
        },
        {
            "id": "email_002",
            "sender": "angry.client@company.com",
            "subject": "REFUND REQUEST - TERRIBLE SERVICE",
            "body": "This is unacceptable! I need a refund immediately for the broken product!",
            "intent": "angry_client_refund",
            "priority": 4
        },
        {
            "id": "email_003",
            "sender": "ceo@megacorp.com",
            "subject": "🚨 CRITICAL SERVER OUTAGE",
            "body": "All our servers are down! This is an emergency requiring immediate attention!",
            "intent": "vip_server_outage",
            "priority": 5
        }
    ]
    
    # Initialize session state
    if 'current_email' not in st.session_state:
        st.session_state.current_email = random.choice(emails)
        st.session_state.total_reward = 0.0
        st.session_state.actions = []
    
    email = st.session_state.current_email
    
    # Layout
    col1, col2, col3 = st.columns([1.2, 1, 1.2])
    
    with col1:
        st.subheader("📧 Current Email")
        st.markdown(f"**From:** {email['sender']}")
        st.markdown(f"**Subject:** {email['subject']}")
        st.markdown(f"**Intent:** `{email['intent']}`")
        st.markdown(f"**Priority:** {'⭐' * email['priority']}")
        st.text_area("Body:", email['body'], height=200, disabled=True)
    
    with col2:
        st.subheader("🤖 Action Selection")
        tool = st.selectbox("Select Tool:", ["auto_reply", "route_to_human", "ask_for_clarification"])
        
        if tool == "auto_reply":
            message = st.text_area("Message:", height=100, placeholder="Enter your response...")
            if st.button("Execute Action", type="primary"):
                reward = 1.0 if email['intent'] == "routine_password_reset" else -1.0
                st.session_state.total_reward += reward
                st.session_state.actions.append({"tool": tool, "reward": reward})
                st.session_state.current_email = random.choice(emails)
                st.rerun()
                
        elif tool == "route_to_human":
            dept = st.selectbox("Department:", ["IT", "Customer Service", "Emergency Support", "Security"])
            if st.button("Execute Action", type="primary"):
                if email['intent'] == "vip_server_outage" and dept == "Emergency Support":
                    reward = 1.0
                elif email['intent'] == "angry_client_refund" and dept == "Customer Service":
                    reward = 0.8
                else:
                    reward = 0.3
                st.session_state.total_reward += reward
                st.session_state.actions.append({"tool": tool, "reward": reward})
                st.session_state.current_email = random.choice(emails)
                st.rerun()
                
        else:
            if st.button("Execute Action", type="primary"):
                reward = 0.2
                st.session_state.total_reward += reward
                st.session_state.actions.append({"tool": tool, "reward": reward})
                st.session_state.current_email = random.choice(emails)
                st.rerun()
    
    with col3:
        st.subheader("📊 Metrics")
        
        # Total reward
        color = "green" if st.session_state.total_reward > 0 else "red"
        st.markdown(f"### Total Reward: :{color}[{st.session_state.total_reward:.2f}]")
        
        # Progress
        progress = len(st.session_state.actions) / 10.0
        st.progress(min(1.0, progress))
        st.write(f"Actions: {len(st.session_state.actions)}/10")
        
        # History
        st.markdown("**Recent Actions:**")
        for i, action in enumerate(st.session_state.actions[-3:], 1):
            color = "green" if action['reward'] > 0 else "red"
            st.write(f"{i}. {action['tool']} - :{color}[{action['reward']:+.2f}]")
    
    # Footer
    st.markdown("---")
    st.markdown("### 🎯 Hackathon Demo Features")
    cols = st.columns(3)
    with cols[0]:
        st.metric("Email Types", "10", "Diverse scenarios")
    with cols[1]:
        st.metric("Tools", "3", "AI-powered actions")
    with cols[2]:
        st.metric("Rewards", f"{st.session_state.total_reward:.1f}", "Learning system")

if __name__ == "__main__":
    main()
