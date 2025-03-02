"""
rag.py

This module implements the Retrieval Augmented Generation (RAG) pipeline.
It retrieves relevant document chunks from the vector store and queries the LLM
to generate an answer with citations.
"""

import os
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain


def run_qa_chain(vectorstore, question, k=5):
    """
    Run a Retrieval Augmented Generation (RAG) pipeline on the provided vector store.

    :param vectorstore: A FAISS vector store with embedded documents.
    :param question: The query to answer.
    :param k: Number of documents to retrieve.
    :return: Answer string with citations.
    """
    # Retrieve top k relevant documents.
    retrieved_docs = vectorstore.similarity_search(question, k=k)

    # Load the OpenAI API key from the environment variable.
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Configure the LLM with the API key.
    llm = ChatOpenAI(
        api_key=openai_api_key,
        model_name="o3-mini",
        temperature=0
    )

    # Create a QA chain using the map_reduce strategy.
    qa_chain = load_qa_chain(llm, chain_type="map_reduce")
    answer = qa_chain.run(input_documents=retrieved_docs, question=question)

    # Extract citation information from document metadata.
    citations = "\n".join(
        [f"- {doc.metadata.get('source', 'Unknown')}" for doc in retrieved_docs]
    )
    final_answer = f"{answer}\n\nCitations:\n{citations}"
    return final_answer
