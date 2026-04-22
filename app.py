"""
Enterprise Email Triage Simulator - Streamlit Dashboard

Visual dashboard for demonstrating the EnterpriseEmailEnv environment.
Perfect for Meta PyTorch Hackathon judges to visualize the AI agent's performance.

Author: Senior Python AI Engineer
Purpose: Meta PyTorch Hackathon - Theme 3.1: Professional Tasks (Enterprise Workflows)
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List
import json
from env import EnterpriseEmailEnv
from openai import OpenAI


def initialize_session_state():
    """Initialize Streamlit session state with environment and tracking variables."""
    if 'env' not in st.session_state:
        st.session_state.env = EnterpriseEmailEnv()
        st.session_state.observation = st.session_state.env.reset()
        st.session_state.action_history = []
        st.session_state.reward_history = []
        st.session_state.episode_done = False
        st.session_state.openai_api_key = ""
        st.session_state.last_ai_response = None


def reset_environment():
    """Reset the environment and clear history."""
    st.session_state.env = EnterpriseEmailEnv()
    st.session_state.observation = st.session_state.env.reset()
    st.session_state.action_history = []
    st.session_state.reward_history = []
    st.session_state.episode_done = False
    st.rerun()


def get_llm_decision(observation: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """
    Get AI decision from OpenAI based on current email observation.
    
    Args:
        observation: Current environment state as dictionary
        api_key: OpenAI API key
        
    Returns:
        Dictionary containing tool call in required format
    """
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        
        # System prompt for AI agent
        system_prompt = """You are a strict Enterprise Email Triage Agent. You do not explain yourself. You ONLY output a single, valid JSON object.

The JSON must have EXACTLY this structure: {"tool": "<tool_name>", "arguments": {<args>}}

You must choose ONE tool from this exact list: auto_reply, route_to_human, ask_for_clarification.

If you use route_to_human, your arguments MUST include department.

If you use auto_reply, your arguments MUST include message.

All arguments MUST include email_id of the current email.

Available tools:
1. auto_reply - Automatically respond to routine emails (requires: email_id, message)
2. route_to_human - Escalate complex/urgent issues to human agents (requires: email_id, department)  
3. ask_for_clarification - Request more information when unclear (requires: email_id)

Business Rules:
- VIP server outages: IMMEDIATELY route to Emergency Support/IT (never auto-reply)
- Angry client complaints: Route to Customer Service (never auto-reply)
- Password resets: Auto-reply with helpful guidance
- General inquiries: Route to appropriate department or ask for clarification
- Spam: Route to Security

DO NOT hallucinate. DO NOT invent tools. DO NOT explain your reasoning. ONLY output the JSON object."""

        # User prompt with current observation
        user_prompt = f"""Current email requiring triage:

{json.dumps(observation, indent=2)}

Analyze this email and decide the best action. Consider the email intent, priority, VIP status, and content. Respond with the appropriate tool call in JSON format."""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=500
        )
        
        # Parse and return the JSON response
        ai_decision = json.loads(response.choices[0].message.content)
        st.session_state.last_ai_response = {
            "prompt": user_prompt,
            "response": ai_decision,
            "raw_response": response.choices[0].message.content
        }
        
        return ai_decision
        
    except Exception as e:
        st.error(f"AI Error: {str(e)}")
        return None


def execute_action(tool_name: str, arguments: Dict[str, Any]):
    """Execute an action in the environment and update state."""
    if st.session_state.episode_done:
        st.warning("Episode is complete! Please reset the environment.")
        return
    
    action = {
        "tool": tool_name,
        "arguments": arguments
    }
    
    try:
        observation, reward, done, info = st.session_state.env.step(action)
        st.session_state.observation = observation
        st.session_state.episode_done = done
        
        # Update history
        action_record = {
            "step": len(st.session_state.action_history) + 1,
            "tool": tool_name,
            "arguments": arguments,
            "reward": reward,
            "email_intent": info.get("email_intent", "Unknown"),
            "error": info.get("error", None)
        }
        st.session_state.action_history.append(action_record)
        st.session_state.reward_history.append(reward)
        
        if done:
            st.success(f"Episode Complete! Total Reward: {st.session_state.env.total_reward:.2f}")
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Error executing action: {str(e)}")


def execute_ai_action():
    """Execute AI decision in the environment."""
    if st.session_state.episode_done:
        st.warning("Episode is complete! Please reset the environment.")
        return
    
    if not st.session_state.openai_api_key.strip():
        st.error("Please enter your OpenAI API key in the sidebar first.")
        return
    
    with st.spinner('AI is analyzing the email...'):
        try:
            ai_decision = get_llm_decision(st.session_state.observation, st.session_state.openai_api_key)
            
            if ai_decision and "tool" in ai_decision and "arguments" in ai_decision:
                # Execute the AI's decision
                execute_action(ai_decision["tool"], ai_decision["arguments"])
            else:
                st.error("AI returned invalid response format.")
                
        except Exception as e:
            st.error(f"Error executing AI action: {str(e)}")


def render_email_inbox():
    """Render Column 1: The Email Inbox"""
    st.subheader("Current Email")
    
    if st.session_state.episode_done:
        st.info("No more emails to process. Episode complete!")
        return
    
    email = st.session_state.observation['current_email']
    
    if email:
        # Email header with priority indicator
        priority_color = {
            5: "red",
            4: "orange", 
            3: "yellow",
            2: "blue",
            1: "green"
        }.get(email['priority'], "gray")
        
        st.markdown(f"**Priority:** :{priority_color}[{email['priority']}/5] {'(VIP)' if email['is_vip'] else ''}")
        st.markdown(f"**From:** {email['sender']}")
        st.markdown(f"**Subject:** {email['subject']}")
        st.markdown(f"**Intent:** `{email['intent']}`")
        
        # Email body
        st.markdown("---")
        st.markdown("**Email Body:**")
        st.text_area("", email['body'], height=200, disabled=True, label_visibility="collapsed")
        
        # Suggested department
        if email.get('suggested_department'):
            st.info(f"Suggested Department: {email['suggested_department']}")
    else:
        st.warning("No current email available")


def render_manual_override():
    """Render Column 2: Manual Override / Agent Simulation"""
    st.subheader("Manual Override (Agent Simulation)")
    
    if st.session_state.episode_done:
        st.warning("Episode complete! Reset to continue.")
        return
    
    # Tool selection
    current_email = st.session_state.observation['current_email']
    email_id = current_email['email_id'] if current_email else ""
    
    tool_options = ["auto_reply", "route_to_human", "ask_for_clarification"]
    selected_tool = st.selectbox("Select Tool:", tool_options)
    
    # Dynamic argument inputs based on tool
    arguments = {"email_id": email_id}
    
    if selected_tool == "auto_reply":
        st.markdown("**Auto Reply Arguments:**")
        message = st.text_area("Message:", height=100, 
                             placeholder="Enter your auto-reply message here...")
        if message.strip():
            arguments["message"] = message.strip()
        else:
            st.warning("Message is required for auto_reply")
            
    elif selected_tool == "route_to_human":
        st.markdown("**Route to Human Arguments:**")
        department = st.text_input("Department:", 
                                 placeholder="e.g., Customer Service, IT, Emergency Support")
        if department.strip():
            arguments["department"] = department.strip()
        else:
            st.warning("Department is required for route_to_human")
            
    elif selected_tool == "ask_for_clarification":
        st.markdown("**Ask for Clarification Arguments:**")
        st.info(f"Email ID: {email_id} (automatically set)")
    
    # Execute button
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Execute Action", type="primary", use_container_width=True, disabled=st.session_state.episode_done):
            # Validate required arguments
            if selected_tool == "auto_reply" and "message" not in arguments:
                st.error("Please provide a message for auto_reply")
            elif selected_tool == "route_to_human" and "department" not in arguments:
                st.error("Please provide a department for route_to_human")
            else:
                execute_action(selected_tool, arguments)
    
    with col2:
        # Make reset button more prominent when episode is done
        button_type = "primary" if st.session_state.episode_done else "secondary"
        if st.button("Reset Environment", type=button_type, use_container_width=True, key="column2_reset"):
            reset_environment()
    
    # Show prominent reset message if episode is complete
    if st.session_state.episode_done:
        st.success("Episode Complete! Click 'Reset Environment' to start a new episode.")
    
    # Tool descriptions
    st.markdown("---")
    st.markdown("**Tool Descriptions:**")
    
    tool_descriptions = {
        "auto_reply": "Automatically respond to the email with a custom message.",
        "route_to_human": "Escalate the email to a human agent in a specific department.",
        "ask_for_clarification": "Request more information about the email."
    }
    
    st.info(tool_descriptions.get(selected_tool, ""))
    
    # AI Integration Section
    st.markdown("---")
    st.markdown("### AI-Powered Triage")
    st.markdown("Let the AI agent analyze and handle the email automatically.")
    
    # Let AI Handle This Button
    if st.button("### :robot_face: Let AI Handle This", 
                type="primary", 
                use_container_width=True,
                disabled=st.session_state.episode_done or not st.session_state.openai_api_key.strip()):
        execute_ai_action()
    
    # Disable message for AI button
    if st.session_state.episode_done:
        st.info("Episode complete. Reset to enable AI features.")
    elif not st.session_state.openai_api_key.strip():
        st.info("Enter API key in sidebar to enable AI features.")
    
    # AI Thought Process Expander
    if st.session_state.last_ai_response:
        with st.expander("### :brain: AI Thought Process", expanded=True):
            st.markdown("**Latest AI Decision:**")
            st.json(st.session_state.last_ai_response["response"])
            
            st.markdown("---")
            st.markdown("**Raw AI Response:**")
            st.code(st.session_state.last_ai_response["raw_response"], language="json")
            
            st.markdown("---")
            st.markdown("**Prompt Sent to AI:**")
            st.text_area("", st.session_state.last_ai_response["prompt"], height=200, disabled=True, label_visibility="collapsed")


def render_metrics_history():
    """Render Column 3: Metrics & History"""
    st.subheader("Metrics & History")
    
    # Total Reward Metric
    total_reward = st.session_state.env.total_reward
    reward_color = "green" if total_reward > 0 else "red" if total_reward < 0 else "gray"
    
    st.markdown(f"### Total Reward: :{reward_color}[{total_reward:.2f}]")
    
    # Progress
    processed = len(st.session_state.observation['processed_emails'])
    total_emails = len(st.session_state.env.emails)  # Dynamic total from environment
    progress_val = processed / total_emails
    progress_clamped = min(1.0, progress_val)  # Clamp to maximum of 1.0
    
    st.markdown("**Progress:**")
    st.progress(progress_clamped)
    st.write(f"{processed}/{total_emails} emails processed")
    
    # Action History
    st.markdown("---")
    st.markdown("**Action History:**")
    
    if st.session_state.action_history:
        # Create DataFrame for better display
        history_df = pd.DataFrame(st.session_state.action_history)
        
        # Format for display
        display_df = history_df.copy()
        display_df['Arguments'] = display_df['arguments'].apply(
            lambda x: json.dumps(x, indent=2) if isinstance(x, dict) else str(x)
        )
        display_df['Reward'] = display_df['reward'].apply(
            lambda x: f":green[{x:+.2f}]" if x > 0 else f":red[{x:+.2f}]" if x < 0 else f":gray[{x:+.2f}]"
        )
        
        # Display as table
        st.dataframe(
            display_df[['step', 'tool', 'Arguments', 'email_intent', 'Reward']], 
            use_container_width=True,
            hide_index=True
        )
        
        # Reward trend chart
        if len(st.session_state.reward_history) > 1:
            st.markdown("**Reward Trend:**")
            reward_df = pd.DataFrame({
                'Step': range(1, len(st.session_state.reward_history) + 1),
                'Cumulative Reward': pd.Series(st.session_state.reward_history).cumsum(),
                'Step Reward': st.session_state.reward_history
            })
            
            st.line_chart(reward_df.set_index('Step')[['Cumulative Reward', 'Step Reward']])
            
    else:
        st.info("No actions taken yet. Start by executing an action!")
    
    # Environment Info
    st.markdown("---")
    st.markdown("**Environment Info:**")
    st.write(f"Step Count: {st.session_state.observation['step_count']}")
    st.write(f"Last Action: {st.session_state.observation['last_action'] or 'None'}")
    
    if st.session_state.episode_done:
        st.success("Episode Complete! All emails have been processed.")


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Enterprise Email Triage Simulator",
        page_icon=":email:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title(":email: Enterprise Email Triage Simulator")
    st.markdown("**Meta PyTorch Hackathon - Theme 3.1: Professional Tasks (Enterprise Workflows)**")
    st.markdown("Visual dashboard for AI-powered email triage using LLM tool calls.")
    
    # Prominent Reset Button at the top
    if st.button("Reset Environment", type="secondary", use_container_width=True, key="top_reset"):
        reset_environment()
    
    # Main content - 3 columns
    col1, col2, col3 = st.columns([1.2, 1, 1.2])
    
    with col1:
        render_email_inbox()
    
    with col2:
        render_manual_override()
    
    with col3:
        render_metrics_history()
    
    # Footer
    st.markdown("---")
    st.markdown("**Instructions:**")
    st.markdown("""
    1. **Column 1**: View the current email requiring triage
    2. **Column 2**: Select a tool and provide arguments, then execute the action
    3. **Column 3**: Monitor performance metrics and action history
    4. The AI agent learns to maximize rewards by making optimal triage decisions
    """)
    
    # Sidebar with additional info
    with st.sidebar:
        st.header("Environment Details")
        
        # API Key Input
        st.markdown("**OpenAI API Key:**")
        api_key = st.text_input(
            "Enter your OpenAI API key:",
            type="password",
            value=st.session_state.openai_api_key,
            help="Get your API key from https://platform.openai.com/api-keys"
        )
        st.session_state.openai_api_key = api_key
        
        if api_key.strip():
            st.success("API key set! AI features enabled.")
        else:
            st.warning("API key required for AI features.")
        
        # Test Sandbox
        with st.sidebar.expander("🧪 Sandbox / Manual Test"):
            st.markdown("**Test Custom Email Scenarios**")
            
            sandbox_input = st.text_area(
                "Input Custom Email Body", 
                key="sandbox_input",
                height=150,
                placeholder="Enter a custom email scenario to test..."
            )
            
            if st.button("Inject into Environment", key="inject_sandbox"):
                if sandbox_input.strip():
                    # Create a temporary Email object for testing
                    from env import Email, EmailIntent
                    
                    test_email = Email(
                        id="sandbox_test",
                        sender="manual.test@demo.com",
                        subject="Manual Test Scenario",
                        body=sandbox_input.strip(),
                        intent=EmailIntent.GENERAL_INQUIRY,  # Default to general inquiry
                        priority=3,
                        is_vip=False,
                        department="General"
                    )
                    
                    # Inject into environment
                    st.session_state.env.current_email = test_email
                    st.session_state.observation = st.session_state.env._get_llm_observation()
                    
                    st.success("✅ Test email injected into environment!")
                    st.rerun()
        
        # Prominent Reset Button in Sidebar
        st.markdown("---")
        if st.button("Reset Environment", type="primary", use_container_width=True, key="sidebar_reset"):
            reset_environment()
        st.markdown("---")
        
        st.markdown("**Available Tools:**")
        st.markdown("""
        - **auto_reply**: Automatically respond to routine emails
        - **route_to_human**: Escalate complex issues to humans
        - **ask_for_clarification**: Request more information
        """)
        
        st.markdown("**Reward Structure:**")
        st.markdown("""
        - **+1.0**: Route VIP outage to Emergency/IT
        - **+1.0**: Auto-reply password reset with keywords
        - **+0.8**: Route angry client to Customer Service
        - **-1.0**: Auto-reply to angry client/VIP
        - **-0.5**: Invalid arguments or looping
        """)
        
        st.markdown("**Email Types:**")
        st.markdown("""
        - Password Reset (Routine)
        - Angry Client (Urgent)
        - VIP Server Outage (Critical)
        - General Inquiry (Medium)
        - Spam (Low Priority)
        """)


if __name__ == "__main__":
    main()
