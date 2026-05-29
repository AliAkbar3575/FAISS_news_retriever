import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable


@traceable(name="text_splitting")
def text_splitting(data):
    time.sleep(2)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(data)
    return chunks