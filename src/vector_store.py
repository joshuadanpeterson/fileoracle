"""
vector_store.py

This module handles document loading, text splitting, embedding, and
building a FAISS vector store for fast similarity search using LangChain.
It now filters out hidden files, directories, and files that do not have an allowed extension.
"""

import os
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Restrict processing to only these file types.
ALLOWED_EXTENSIONS = {'.pdf', '.doc', '.docx', '.txt', '.md'}

def is_relevant_file(filepath):
    """
    Return True if the file has an allowed extension.
    """
    ext = os.path.splitext(filepath)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def load_documents(directory):
    """
    Load documents from a directory using LangChain's UnstructuredFileLoader.
    Filters out hidden files, directories, and files that do not have an allowed extension.
    
    Each document is augmented with metadata containing its source path.
    
    :param directory: The directory to load documents from.
    :return: List of documents.
    """
    docs = []
    # Define an exclusion set for filenames known to be irrelevant.
    exclusion_set = {'.DS_Store', '.dropbox', 'Trash', 'Recovered Data'}

    try:
        for filename in os.listdir(directory):
            # Skip hidden files and those in the exclusion set.
            if filename.startswith('.') or filename in exclusion_set:
                continue

            filepath = os.path.join(directory, filename)
            # Skip if the file doesn't exist or isn't a regular file.
            if not os.path.exists(filepath) or not os.path.isfile(filepath):
                continue

            # Skip files that do not have an allowed extension.
            if not is_relevant_file(filepath):
                continue

            try:
                loader = UnstructuredFileLoader(filepath)
                for doc in loader.load():
                    doc.metadata["source"] = filepath
                    docs.append(doc)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    except Exception as dir_error:
        print(f"Error accessing directory {directory}: {dir_error}")
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
