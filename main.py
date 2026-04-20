import os
import streamlit as st
import pickle
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

st.title("NewsLens AI 🔎")
st.write("News research tools...")
st.sidebar.title("Enter News Articles URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1} 🔗")
    urls.append(url)

main_placeholder = st.empty()
query = main_placeholder.text_input("Enter your question: 🤔")

process_url_clicked = st.button("Process URLs")

load_dotenv()  # Load environment variables from .env file
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    temperature=0.7,
    max_tokens=500,
    api_key=api_key
)

file_path = "vector_index.pkl"

if process_url_clicked:
    
    # data loading
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading started...🔃")
    time.sleep(2)
    data = loader.load()  # This will load the data from the URLs and print the content in the console

    # text splitting
    main_placeholder.text("Text Splitting started...🔃")
    time.sleep(2)
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1000,
    )
    chunks = splitter.split_documents(data)

    # embedding and save to FAISS vector store
    main_placeholder.text("Embedding and saving to FAISS vector store started...🔃")
    time.sleep(2)

    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # Small, fast, free
)
    vector_index = FAISS.from_documents(chunks, embeddings)

    with open(file_path, "wb") as f:
        pickle.dump(vector_index, f)

    

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vector_index = pickle.load(f)

        # retriever pipeline
        # ---------------------------------------------------------

        retriever = vector_index.as_retriever()
        
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])

        # Function to extract sources from docs
        def extract_sources(docs):
            sources = []
            for doc in docs:
                # Try different metadata keys where URL might be stored
                url = (doc.metadata.get('url') or 
                    doc.metadata.get('source') or 
                    doc.metadata.get('link') or
                    doc.metadata.get('URL'))
                if url:
                    sources.append(url)
            return list(set(sources))  # Remove duplicates
        
        prompt = ChatPromptTemplate.from_template(
            """
        Answer the question based only on the context.

        Context:
        {context}

        Question:
        {question}
        """
        )

        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        retrieved_docs = retriever.invoke(query)
        answer = rag_chain.invoke(query)

        sources = extract_sources(retrieved_docs)

        st.header("Answer:")
        st.write(answer)

        st.header("Sources:")
        for i, source in enumerate(sources, 1):
            st.write(f"{i}. {source}")
        
