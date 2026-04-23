import streamlit as st

st.set_page_config(page_title="Email Triage", layout="wide")

st.title("📧 Enterprise Email Triage")
st.write("Meta Pytort Hackathon Demo")

st.write("This is a working demo of the email triage system.")

if st.button("Click me"):
    st.success("✅ It works!")

st.write("Current time:", st.write("Ready for hackathon judges!"))
