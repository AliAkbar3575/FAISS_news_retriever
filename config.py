import os
from dataclasses import dataclass


@dataclass
class RAGConfig:
    chunk_size: int = 800
    chunk_overlap: int = 150
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "llama-3.1-8b-instant"
    llm_temperature: float = 0.3
    retriever_k: int = 2
    vector_index_path: str = "vector_index.pkl"
    num_urls: int = 3


def load_config() -> RAGConfig:
    return RAGConfig(
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        llm_model=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
        retriever_k=int(os.getenv("RETRIEVER_K", "2")),
        vector_index_path=os.getenv("VECTOR_INDEX_PATH", "vector_index.pkl"),
        num_urls=int(os.getenv("NUM_URLS", "3")),
    )
