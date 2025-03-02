"""
alfred_integration.py

This module integrates FileOracle with Alfred/Raycast.
It handles searching across multiple directories, extracting text,
loading documents into the vector store, and running the RAG pipeline.
"""

import sys
import os
from src.file_search import search_files, refine_fzf_selection  # Updated import
from src.file_extractor import extract_text
from src.rag import run_qa_chain
from src.vector_store import load_documents, build_vector_store
from src.directory_selector import (
    iterative_directory_traversal as select_relevant_directory,
)

# List of root directories in prioritized order: Dropbox, Google Drive, and ~/Documents.
ROOT_DIRS = [
    "/Users/joshpeterson/Library/CloudStorage/Dropbox/",
    "/Users/joshpeterson/Library/CloudStorage/GoogleDrive-joshuadanpeterson@gmail.com/My Drive/",
    os.path.expanduser("~/Documents"),
]


def alfred_main(query):
    """
    Integrates with Alfred/Raycast to run a query.

    1. For each root directory, uses the LLM to iteratively traverse and select the most relevant subdirectory.
    2. Searches for files matching the query in the selected directories.
    3. Optionally refines fzf results using the LLM for further ranking.
    4. Extracts text from the first matching file.
    5. Loads documents from all selected directories.
    6. Builds a vector store and runs the RAG pipeline to answer the query with citations.

    :param query: The query string from Alfred.
    """
    # For each root directory, use iterative traversal to select the most relevant subdirectory.
    relevant_directories = []
    for root in ROOT_DIRS:
        selected_directory = select_relevant_directory(root, query)
        relevant_directories.append(selected_directory)

    # Now, relevant_directories is a list of directories from Dropbox, Google Drive, and Documents.
    matching_files = []
    # Search each recommended directory.
    for directory in relevant_directories:
        files = search_files(query, [directory])
        matching_files.extend(files)

    if not matching_files:
        print("No files found matching your query.")
        return

    # Optionally, refine fzf results via LLM refinement.
    # For example, if interactive selection was used, you can refine that selection:
    # best_file = refine_fzf_selection(matching_files, query)
    # For now, we'll use the first file:
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
