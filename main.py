import os
import streamlit as st
import pickle
from dotenv import load_dotenv

from rag_pipeline.data_loader import data_loading
from rag_pipeline.text_splitter import text_splitting
from rag_pipeline.vectorestore import vector_store
from rag_pipeline.retriever import retriever

from langsmith import traceable

load_dotenv()

@traceable(name="main_function")
def main():

    st.title("NewsLens AI 🔎")
    st.markdown(
    """
    ### AI-Powered News Research & Retrieval System
    
    Analyze, summarize, and query multiple news articles using Retrieval-Augmented Generation (RAG).
    """
    )
    st.sidebar.title("Enter News Articles URLs")

    st.divider()

    # ----------------- sidebar structure -----------------

    file_path = "vector_index.pkl"

    with st.sidebar:

        urls = []
        for i in range(3):
            url = st.text_input(f"URL {i+1} 🔗")
            urls.append(url)

        # uploaded_files = st.file_uploader(
        #     "Upload data", accept_multiple_files=False, type="pdf"
        # )

        process_urls_button = st.button("Process URLs")

        
    if process_urls_button:
        with st.spinner("Data loading started...", show_time=True):
            data = data_loading(urls)
        with st.spinner("Text Splitting started...", show_time=True):
            chunks = text_splitting(data)
        with st.spinner("Creating embeddings and saving to FAISS vector store...", show_time=True):
            vector_index = vector_store(chunks)

        st.success('Successfully loaded all documents!', icon="✅")

    # -------------- front page ------------------


    query = st.text_input("Enter your question 🤔")
    answer_button = st.button("Answer")    


    if answer_button:

        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                vector_index = pickle.load(f)

        with st.spinner("Retrieving you answer...", show_time=True):
            response, retrieved_docs = retriever(vector_index, query)

        st.header("Question")
        st.write(query)

        st.header("Answer")
        st.write(response)

        st.header("Sources")
        st.write(f"{retrieved_docs[0].metadata['source']}")



if __name__ == "__main__":
    main()

    # import streamlit as st

    # option = st.selectbox(
    #     "How would you like to be contacted?",
    #     ("Email", "Home phone", "Mobile phone"),
    #     index=None,
    #     placeholder="Select contact method...",
    # )

    # st.write("You selected:", option)