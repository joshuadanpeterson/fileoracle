"""
alfred_integration.py

This module integrates FileOracle with Alfred/Raycast.
It handles searching across multiple directories, extracting text,
loading documents into the vector store, and running the RAG pipeline.
"""

import sys
import os
from src.file_search import search_files
from src.file_extractor import extract_text
from src.rag import run_qa_chain
from src.vector_store import load_documents, build_vector_store

# Define the directories to search and load documents from.
DIRECTORIES = [
    "/Users/joshpeterson/Library/CloudStorage/Dropbox/",
    "/Users/joshpeterson/Library/CloudStorage/GoogleDrive-joshuadanpeterson@gmail.com/My Drive/",
    os.path.expanduser("~/Documents"),
]


def alfred_main(query):
    """
    Integrates with Alfred/Raycast to run a query.

    1. Searches for files matching the query across multiple directories.
    2. Extracts text from the first matching file.
    3. Loads documents from all directories.
    4. Builds a vector store from these documents.
    5. Runs the RAG pipeline to answer the query with citations.

    :param query: The query string from Alfred.
    """
    matching_files = []
    # Search each directory for matching files.
    for directory in DIRECTORIES:
        files = search_files(query, [directory])
        matching_files.extend(files)

    if not matching_files:
        print("No files found matching your query.")
        return

    first_file = matching_files[0]
    text = extract_text(first_file)
    print(f"Extracted text from: {first_file}\n")

    # Load documents from all directories.
    docs = []
    for directory in DIRECTORIES:
        docs.extend(load_documents(directory))

    vectorstore = build_vector_store(docs)

    # Use the query as the question for the RAG pipeline.
    answer = run_qa_chain(vectorstore, query)
    print(answer)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        alfred_main(query)
    else:
        print("Please provide a query.")
