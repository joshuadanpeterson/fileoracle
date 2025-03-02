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
from src.directory_selector import iterative_directory_traversal as select_relevant_directories  # New import

# Default directories are now defined in directory_selector.py


def alfred_main(query):
    """
    Integrates with Alfred/Raycast to run a query.

    1. Uses the LLM to filter directories based on the query.
    2. Searches for files matching the query in the selected directories.
    3. Extracts text from the first matching file.
    4. Loads documents from all selected directories.
    5. Builds a vector store and runs the RAG pipeline to answer the query with citations.

    :param query: The query string from Alfred.
    """
    # Use the LLM to determine which directories are relevant.
    relevant_directories = select_relevant_directories(query)

    matching_files = []
    # Search each recommended directory.
    for directory in relevant_directories:
        files = search_files(query, [directory])
        matching_files.extend(files)

    if not matching_files:
        print("No files found matching your query.")
        return

    first_file = matching_files[0]
    text = extract_text(first_file)
    print(f"Extracted text from: {first_file}\n")

    # Load documents from the selected directories.
    docs = []
    for directory in relevant_directories:
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
