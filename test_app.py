import streamlit as st

st.title("✅ Streamlit is working!")
st.write("If you see this text, your Streamlit front-end is fine.")

x = st.slider("Select a number", 0, 10, 5)
st.write("You selected:", x)
