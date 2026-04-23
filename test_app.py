import streamlit as st
import time

def main():
    st.set_page_config(page_title="Test App", page_icon="🧪", layout="wide")
    
    st.title("🧪 Test Application")
    st.write("This is a test to verify the Hugging Face Space is working.")
    
    if st.button("Test Button"):
        st.success("✅ Button clicked successfully!")
        st.write("Current time:", time.strftime("%Y-%m-%d %H:%M:%S"))
    
    st.write("### Environment Test")
    st.write(f"Python version: 3.11")
    st.write(f"Streamlit version: {st.__version__}")
    
    st.write("### Files in Directory")
    import os
    files = os.listdir(".")
    for file in files:
        st.write(f"- {file}")

if __name__ == "__main__":
    main()
