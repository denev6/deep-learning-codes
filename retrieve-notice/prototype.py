import streamlit as st
from dotenv import load_dotenv

from retrieve import iter_retrieve_for_testing

load_dotenv()

st.title("SKKU 공지 검색")

query = st.text_input("입력: ")

if st.button("검색"):
    for info in iter_retrieve_for_testing(query):
        st.write(info)
