"""
vector_store.py

This module handles document loading, text splitting, embedding, and
building a FAISS vector store for fast similarity search using LangChain.
"""

import os
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def load_documents(directory):
    """
    Load documents from a directory using LangChain's UnstructuredFileLoader.

    Each document is augmented with metadata containing its source path.

    :param directory: The directory to load documents from.
    :return: List of documents.
    """
    docs = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        try:
            loader = UnstructuredLoader(filepath)
            # The loader returns a list of Document objects; add metadata if needed.
            for doc in loader.load():
                doc.metadata["source"] = filepath
                docs.append(doc)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    return docs


def build_vector_store(docs):
    """
    Build a FAISS vector store from a list of documents.

    :param docs: List of Document objects.
    :return: FAISS vector store.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()  # Ensure OPENAI_API_KEY is set in your environment.
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore
