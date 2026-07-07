from __future__ import annotations

import os
from typing import Dict, List

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 120

st.set_page_config(page_title="Semantic RAG", page_icon="📚", layout="wide")

def api_get(path: str) -> Dict:
    resp = requests.get(f"{API_BASE_URL}{path}", timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def api_post(path: str, json: Dict) -> Dict:
    resp = requests.post(f"{API_BASE_URL}{path}", json=json, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def api_upload(path: str, files) -> Dict:
    resp = requests.post(f"{API_BASE_URL}{path}", files=files, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_documents() -> List[Dict]:
    try:
        return api_get("/documents").get("documents", [])
    except Exception:  # noqa: BLE001
        return []


def confidence_color(score: float) -> str:
    if score >= 0.6:
        return "green"
    if score >= 0.4:
        return "orange"
    return "red"
def render_sidebar() -> None:
    st.sidebar.title("📚 Semantic RAG")
    st.sidebar.caption(f"Backend: `{API_BASE_URL}`")

    try:
        health = api_get("/health")
        st.sidebar.success("Backend online")
        st.sidebar.write(f"**LLM:** {health['llm_provider']}")
        st.sidebar.write(f"**Indexed chunks:** {health['indexed_chunks']}")
    except Exception:  # noqa: BLE001
        st.sidebar.error("Backend offline. Start the FastAPI server.")

    st.sidebar.divider()
    st.sidebar.subheader("Upload documents")
    uploaded = st.sidebar.file_uploader(
        "Add PDF files", type=["pdf"], accept_multiple_files=True
    )
    if uploaded and st.sidebar.button("Ingest uploaded files", type="primary"):
        files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded]
        with st.spinner("Ingesting and indexing (semantic chunking)…"):
            try:
                result = api_upload("/ingest", files)
                st.sidebar.success(
                    f"Ingested {len(result['ingested_files'])} file(s), "
                    f"{result['total_chunks']} chunks."
                )
            except Exception as exc:  # noqa: BLE001
                st.sidebar.error(f"Ingestion failed: {exc}")

    st.sidebar.divider()
    st.sidebar.subheader("Indexed documents")
    docs = get_documents()
    if docs:
        for d in docs:
            st.sidebar.write(f"• {d['source']} ({d['chunk_count']} chunks)")
    else:
        st.sidebar.info("No documents indexed yet.")

def render_citations(citations: List[Dict]) -> None:
    if not citations:
        st.info("No citations were returned for this answer.")
        return
    for i, c in enumerate(citations, start=1):
        with st.expander(
            f"📎 Citation {i}: {c['source']} — page {c['page']} "
            f"(similarity {c['similarity']:.2f})"
        ):
            st.markdown(f"**Chunk ID:** `{c['chunk_id']}`")
            st.markdown(f"> {c['snippet']}")


def render_retrieved(chunks: List[Dict]) -> None:
    if not chunks:
        return
    with st.expander(f"🔍 Retrieved semantic chunks ({len(chunks)})"):
        for i, ch in enumerate(chunks, start=1):
            st.markdown(
                f"**{i}. {ch['source']}** — page {ch['page']} · "
                f"similarity `{ch['similarity']:.3f}` · id `{ch['chunk_id']}`"
            )
            st.write(ch["text"])
            st.divider()


def page_ask() -> None:
    st.header("Ask a question")
    st.caption("Answers are generated only from your ingested documents.")

    docs = get_documents()
    doc_names = [d["source"] for d in docs]

    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input(
            "Your question (any language)",
            placeholder="e.g. What is semantic chunking and why is it used?",
        )
    with col2:
        source_filter = st.selectbox(
            "Limit to document", ["All documents", *doc_names], index=0
        )

    if st.button("Ask", type="primary") and question.strip():
        payload = {"question": question}
        if source_filter != "All documents":
            payload["source_filter"] = source_filter
        with st.spinner("Retrieving context and generating a grounded answer…"):
            try:
                data = api_post("/ask", payload)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Request failed: {exc}")
                return

        st.subheader("Answer")
        st.write(data["answer"])

        m1, m2, m3 = st.columns(3)
        conf = data["confidence"]
        m1.metric("Confidence", f"{conf:.2f}")
        m2.metric("Detected language", data["detected_language"])
        m3.metric("Human review", "Yes" if data["needs_human_review"] else "No")

        st.progress(min(max(conf, 0.0), 1.0))
        if data["needs_human_review"]:
            st.warning(
                "⚠️ Confidence is below the review threshold (0.40). "
                "A human should verify this answer."
            )

        st.subheader("Citations")
        render_citations(data.get("citations", []))
        render_retrieved(data.get("retrieved_chunks", []))

def page_contradict() -> None:
    st.header("Contradiction check")
    st.caption("Compare how two documents treat the same topic.")

    docs = get_documents()
    doc_names = [d["source"] for d in docs]
    if len(doc_names) < 2:
        st.info("Index at least two documents to use this feature.")
        return

    col1, col2 = st.columns(2)
    with col1:
        doc1 = st.selectbox("Document 1", doc_names, index=0)
    with col2:
        default_idx = 1 if len(doc_names) > 1 else 0
        doc2 = st.selectbox("Document 2", doc_names, index=default_idx)

    topic = st.text_input("Topic", placeholder="e.g. remote work eligibility")

    if st.button("Check for contradiction", type="primary") and topic.strip():
        if doc1 == doc2:
            st.warning("Please choose two different documents.")
            return
        with st.spinner("Comparing evidence from both documents…"):
            try:
                data = api_post(
                    "/contradict",
                    {"document_1": doc1, "document_2": doc2, "topic": topic},
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Request failed: {exc}")
                return

        if data["conflict"]:
            st.error("❌ Conflict detected")
        else:
            st.success("✅ No contradiction detected")

        st.subheader("Reasoning")
        st.write(data["reasoning"])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Evidence — {data['document_1']}**")
            render_citations(data.get("citations_document_1", []))
        with c2:
            st.markdown(f"**Evidence — {data['document_2']}**")
            render_citations(data.get("citations_document_2", []))


def main() -> None:
    render_sidebar()
    tab_ask, tab_contradict = st.tabs(["💬 Ask", "⚖️ Contradiction Check"])
    with tab_ask:
        page_ask()
    with tab_contradict:
        page_contradict()


if __name__ == "__main__":
    main()
