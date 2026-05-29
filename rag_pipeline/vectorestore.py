import time
import pickle

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langsmith import traceable


@traceable(name="vector_store")
def vector_store(chunks):
    time.sleep(2)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_index = FAISS.from_documents(chunks, embeddings)

    file_path = "vector_index.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(vector_index, f)

    return vector_index