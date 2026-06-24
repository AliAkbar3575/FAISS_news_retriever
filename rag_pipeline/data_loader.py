import logging
import os
import tempfile
from typing import List

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def url_loading(urls: List[str]) -> List[Document]:
    if not urls:
        return []
    try:
        loader = WebBaseLoader(urls)
        data = loader.load()
        logger.info("Loaded %d documents from %d URLs", len(data), len(urls))
        for doc in data:
            doc.metadata["source_type"] = "web"
        return data
    except Exception as e:
        logger.exception("Failed to load URLs: %s", urls)
        raise RuntimeError(f"Failed to load URLs: {e}") from e


def pdf_loading(pdf_files: List[str]) -> List[Document]:
    if not pdf_files:
        return []
    documents = []
    for path in pdf_files:
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_type"] = "pdf"
            documents.extend(docs)
            logger.info("Loaded %d pages from PDF: %s", len(docs), os.path.basename(path))
        except Exception as e:
            logger.exception("Failed to load PDF: %s", path)
            raise RuntimeError(f"Failed to load PDF {os.path.basename(path)}: {e}") from e
    return documents
