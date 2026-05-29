from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langsmith import traceable

load_dotenv()

@traceable(name="retriever")
def retriever(vector_index, query):

    llm = ChatGroq(
        model='llama-3.1-8b-instant',
        temperature=0.3
    )
    
    retriever = vector_index.as_retriever()
    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = ChatPromptTemplate.from_template(
    """
    You are a highly reliable AI assistant.

    Answer the user's question ONLY using the provided context.
    Do NOT use prior knowledge.
    Do NOT make assumptions.
    Do NOT fabricate information.

    If the answer is not explicitly present in the context, say:
    "I could not find sufficient information in the provided context."

    Instructions:
    - Be concise and accurate.
    - Quote important facts directly when necessary.
    - If multiple pieces of context conflict, mention the conflict.
    - Do not invent citations, statistics, names, or dates.
    - If the context is incomplete, clearly state the limitation.
    - Use only grounded information from the context.

    Context:
    {context}

    Question:
    {query}

    Grounded Answer:
    """
    )

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        'context': context,
        'query': query
    })

    return response, retrieved_docs
    
    

    