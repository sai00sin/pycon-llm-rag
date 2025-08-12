import streamlit as st
import requests

st.title("Local Mini-LLM × RAG Demo")
q = st.text_input("質問を入力")
if st.button("Ask") and q:
    r = requests.post("http://localhost:8000/ask", json={"query": q, "top_k": 3})
    data = r.json()
    st.subheader("回答")
    st.write(data.get("answer", ""))
    st.subheader("参考（インデックス情報の簡易表示）")
    st.write(data.get("refs", []))
