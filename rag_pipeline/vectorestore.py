import logging
import pickle
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def vector_store(
    chunks: List[Document],
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    vector_index_path: str = "vector_index.pkl",
) -> FAISS:
    if not chunks:
        raise ValueError("No chunks to index")

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vector_index = FAISS.from_documents(chunks, embeddings)

    with open(vector_index_path, "wb") as f:
        pickle.dump(vector_index, f)

    logger.info("Saved vector store with %d documents to %s", len(chunks), vector_index_path)
    return vector_index
