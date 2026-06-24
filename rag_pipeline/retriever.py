import logging
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
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


def retriever(
    vector_index: FAISS,
    query: str,
    llm_model: str = "llama-3.1-8b-instant",
    temperature: float = 0.3,
    k: int = 4,
) -> Tuple[str, List[Document]]:
    if not query.strip():
        raise ValueError("Query cannot be empty")

    retriever_obj = vector_index.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever_obj.invoke(query)

    if not retrieved_docs:
        return "No relevant documents found in the knowledge base.", []

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    llm = ChatGroq(model=llm_model, temperature=temperature)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({"context": context, "query": query})
    logger.info("Generated response for query: %s", query[:50])

    return response, retrieved_docs
