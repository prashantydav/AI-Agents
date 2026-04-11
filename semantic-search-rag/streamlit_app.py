import streamlit as st
import requests
from requests.exceptions import RequestException

from config import BACKEND_URL

st.set_page_config(page_title="Semantic Search RAG", layout="wide")
st.title("Semantic Search Engine with RAG")

with st.sidebar:
    st.markdown("## Settings")
    backend_url = st.text_input("Backend URL", value=BACKEND_URL)
    strategy = st.selectbox("Chunking strategy", ["fixed", "recursive", "semantic"], index=2)
    top_k = st.slider("Top results", min_value=1, max_value=10, value=5)

query = st.text_area("Enter a query", height=140)

col1, col2 = st.columns([2, 1])
with col1:
    if st.button("Run retrieval"):
        if not query.strip():
            st.warning("Please provide a query before searching.")
        else:
            try:
                response = requests.post(
                    f"{backend_url}/search",
                    json={"query": query, "strategy": strategy, "top_k": top_k},
                    timeout=20,
                )
                response.raise_for_status()
                payload = response.json()
                st.success("Search completed")
                for item in payload.get("results", []):
                    with st.expander(f"{item['metadata'].get('title', 'Source')} — score {item['score']:.3f}"):
                        st.write(item["text"])
                        st.write(item["metadata"])
            except RequestException as exc:
                st.error(f"Search request failed: {exc}")

    if st.button("Ask with RAG"):
        if not query.strip():
            st.warning("Please provide a query before generating an answer.")
        else:
            try:
                response = requests.post(
                    f"{backend_url}/chat",
                    json={"query": query, "strategy": strategy, "top_k": top_k},
                    timeout=40,
                )
                response.raise_for_status()
                payload = response.json()
                st.markdown("### Answer")
                st.write(payload.get("answer", ""))

                st.markdown("### Retrieved sources")
                for item in payload.get("sources", []):
                    with st.expander(f"{item['metadata'].get('title', 'Source')} — score {item['score']:.3f}"):
                        st.write(item["text"])
                        st.write(item["metadata"])
            except RequestException as exc:
                st.error(f"RAG request failed: {exc}")

with col2:
    st.markdown("## Instructions")
    st.markdown(
        "- Use `Run retrieval` to inspect the top chunks returned by Chroma and MMR.\n"
        "- Use `Ask with RAG` to generate a context-aware answer with cited sources.\n"
        "- If the backend is not running, start it with `uvicorn app.api:app --reload --port 8000`."
    )
