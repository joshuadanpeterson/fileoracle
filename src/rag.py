"""
rag.py

This module implements the Retrieval Augmented Generation (RAG) pipeline.
It retrieves relevant document chunks from the vector store and queries the LLM
to generate an answer with citations.
"""

import os
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate


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
        temperature=None  # Explicitly set temperature to None for o3-mini model
    )

    # Create a document processing chain using the modern approach
    prompt_template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    {context}
    
    Question: {question}
    """
    prompt = PromptTemplate.from_template(prompt_template)
    
    # Create document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create retrieval chain from the vectorstore retriever and document chain
    retriever = vectorstore.as_retriever()
    qa_chain = create_retrieval_chain(retriever, document_chain)
    
    # Execute the chain using invoke() instead of the deprecated run()
    response = qa_chain.invoke({"question": question})
    answer = response["answer"]

    # Extract citation information from document metadata.
    citations = "\n".join(
        [f"- {doc.metadata.get('source', 'Unknown')}" for doc in retrieved_docs]
    )
    final_answer = f"{answer}\n\nCitations:\n{citations}"
    return final_answer
