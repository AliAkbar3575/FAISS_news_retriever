import logging
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from config import load_config
from rag_pipeline.data_loader import url_loading, pdf_loading
from rag_pipeline.text_splitter import text_splitting
from rag_pipeline.vectorestore import vector_store
from rag_pipeline.retriever import retriever

# ---------------------------------------------------------------------------
# Hybrid environment loader — works locally (.env) and on Streamlit Cloud
# (st.secrets).  Secrets set in the Cloud dashboard take precedence.
# ---------------------------------------------------------------------------
load_dotenv()

_secrets_loaded = False
try:
    for key, value in st.secrets.items():
        if value:
            os.environ[key] = str(value)
    _secrets_loaded = bool(st.secrets)
except Exception:
    pass

config = load_config()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("newslens")

st.set_page_config(page_title="NewsLens AI", page_icon="🔎", layout="wide", initial_sidebar_state="expanded")
if not os.environ.get("GROQ_API_KEY"):
    st.warning(
        "⚠️ **GROQ_API_KEY** is not set. "
        "Set it in a local `.env` file or in the [Streamlit Cloud Secrets]"
        "(https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management) dashboard."
    )

for key, default in [("vector_index", None), ("processed", False)]:
    if key not in st.session_state:
        st.session_state[key] = default

_LOGO_CSS = """
<style>
    .block-container { padding-top: 1.5rem !important; }
    .main-header { text-align: center; padding: 0 0 1rem 0; }
    .main-header h1 { font-size: 2.8rem; font-weight: 800; background: linear-gradient(135deg, #1e3c72, #2a5298); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.3rem; letter-spacing: -0.5px; }
    .main-header p { font-size: 1.1rem; color: #555; margin-top: 0; }
    .answer-card { background: #1a1a2e; border-radius: 12px; padding: 2rem; border-left: 5px solid #2a5298; margin: 1rem 0; line-height: 1.7; font-size: 1.05rem; color: #ffffff; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
    .source-meta { font-size: 0.8rem; color: #888; margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid #eee; }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
    .stTabs [data-baseweb="tab"] { font-size: 1.1rem; font-weight: 600; }
</style>
"""
st.markdown(_LOGO_CSS, unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("---")
        st.markdown("### System Status")
        if st.session_state.processed:
            st.success("✅ Vector store ready")
        else:
            st.info("⏳ No articles processed yet")
        st.markdown("---")
        st.markdown("### Configuration")
        st.caption(f"**LLM:** `{config.llm_model}`")
        st.caption(f"**Embeddings:** `all-MiniLM-L6-v2`")
        st.caption(f"**Chunk size:** `{config.chunk_size}`")
        st.caption(f"**Chunk overlap:** `{config.chunk_overlap}`")
        st.caption(f"**Retriever k:** `{config.retriever_k}`")


def render_header():
    st.markdown(
        '<div class="main-header">'
        "<h1>🔎 NewsLens AI</h1>"
        "<p>AI-Powered News Research & Retrieval System</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_process_tab():
    st.markdown("### Input Sources")
    st.caption("Enter news article URLs and/or upload PDF files, then click **Process**.")

    url_col, pdf_col = st.columns(2)

    urls = []
    with url_col:
        st.markdown("**🌐 News URLs**")
        for i in range(config.num_urls):
            url = st.text_input(
                f"URL {i + 1}",
                key=f"url_{i}",
                placeholder="https://example.com/news-article",
                label_visibility="collapsed",
            )
            urls.append(url)

    pdf_paths = []
    with pdf_col:
        st.markdown("**📄 PDF Documents**")
        uploaded = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded:
            for f in uploaded:
                tmp = os.path.join(tempfile.gettempdir(), f.name)
                with open(tmp, "wb") as out:
                    out.write(f.getbuffer())
                pdf_paths.append(tmp)

    if st.button("🚀 Process Sources", type="primary", use_container_width=True):
        valid_urls = [u.strip() for u in urls if u and u.strip()]
        if not valid_urls and not pdf_paths:
            st.error("⚠️ Please enter at least one URL or upload a PDF.")
            return

        progress = st.progress(0, text="Initializing...")
        try:
            all_docs = []

            if valid_urls:
                progress.progress(10, text="Loading web articles...")
                all_docs.extend(url_loading(valid_urls))

            if pdf_paths:
                progress.progress(30, text="Loading PDFs...")
                all_docs.extend(pdf_loading(pdf_paths))

            progress.progress(50, text="Splitting text into chunks...")
            chunks = text_splitting(all_docs, config.chunk_size, config.chunk_overlap)

            progress.progress(70, text="Creating embeddings & building FAISS index...")
            st.session_state.vector_index = vector_store(
                chunks, config.embedding_model, config.vector_index_path
            )
            st.session_state.processed = True

            progress.progress(100, text="Complete!")

            web_count = sum(1 for d in all_docs if d.metadata.get("source_type") == "web")
            pdf_count = sum(1 for d in all_docs if d.metadata.get("source_type") == "pdf")
            parts = []
            if web_count:
                parts.append(f"{web_count} web article(s)")
            if pdf_count:
                parts.append(f"{pdf_count} PDF page(s)")
            st.success(f"✅ Processed {' & '.join(parts)} into {len(chunks)} text chunks.")
        except Exception as e:
            logger.exception("Processing pipeline failed")
            st.error(f"❌ Processing failed: {e}")


def render_ask_tab():
    if not st.session_state.processed:
        st.info("📌 No knowledge base loaded. Go to **Process Articles** to ingest articles first.")
        return

    st.markdown("### Ask a Question")
    st.caption("Ask anything about the processed articles.")

    query = st.text_input(
        "Question",
        placeholder="e.g., What are the main talking points?",
        label_visibility="collapsed",
    )

    if st.button("💡 Get Answer", type="primary", use_container_width=True):
        if not query or not query.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Retrieving documents and generating answer..."):
            try:
                response, docs = retriever(
                    st.session_state.vector_index,
                    query,
                    config.llm_model,
                    config.llm_temperature,
                    config.retriever_k,
                )

                st.markdown("### Answer")
                st.markdown(f'<div class="answer-card">{response}</div>', unsafe_allow_html=True)

                st.markdown("### Sources")
                for i, doc in enumerate(docs):
                    source = doc.metadata.get("source", "Unknown source")
                    stype = doc.metadata.get("source_type", "web")
                    badge = "🌐 Web" if stype == "web" else "📄 PDF"
                    short = source[:60] + "..." if len(source) > 60 else source
                    with st.expander(f"{badge} Source {i + 1}: {short}"):
                        st.markdown(doc.page_content)
                        st.markdown(f'<div class="source-meta">{badge} — {source}</div>', unsafe_allow_html=True)
            except Exception as e:
                logger.exception("Query pipeline failed")
                st.error(f"❌ Query failed: {e}")


def main():
    render_sidebar()
    render_header()

    tab_process, tab_ask = st.tabs(["📥 Process Articles", "💬 Ask Questions"])

    with tab_process:
        render_process_tab()

    with tab_ask:
        render_ask_tab()


if __name__ == "__main__":
    main()
