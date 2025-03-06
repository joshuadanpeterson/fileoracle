"""
agentic_search.py

This module implements an agentic search loop for FileOracle.
The search agent uses LLM-driven decision-making to autonomously select a search strategy.
In this version, the agent prioritizes name-based search results:
- For each candidate directory (narrowed down via iterative traversal), it first runs a name-based search.
- If the number of name-based hits is below a specified threshold, it supplements with content-based search.
- After combining results, an additional filtering step retains only files whose paths contain
  a minimum number of the generated keywords.
- Finally, the documents corresponding to these candidate file paths are read,
  aggregated into a vector store, and a RAG pipeline is used to answer the user’s query.
"""

import os
from openai import OpenAI
from src.file_search import (
    name_based_search,
    content_based_search,
    generate_keywords,
    refine_fzf_selection
)
from langchain.docstore.document import Document
from src.directory_selector import iterative_directory_traversal
from src.file_extractor import extract_text
from src.vector_store import build_vector_store
from src.rag import run_qa_chain

client = OpenAI()

# Prioritized root directories.
ROOT_DIRS = [
    "/Users/joshpeterson/Library/CloudStorage/Dropbox/",
    "/Users/joshpeterson/Library/CloudStorage/GoogleDrive-joshuadanpeterson@gmail.com/My Drive/",
    os.path.expanduser("~/Documents")
]

def call_llm_for_decision(prompt):
    """
    Calls the LLM with the given prompt and returns its decision as a string.
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            store=True,
            messages=[{"role": "user", "content": prompt}]
        )
        decision = completion.choices[0].message.content.strip()
        return decision
    except Exception as e:
        print(f"Error in call_llm_for_decision: {e}")
        return ""

def refine_query(query):
    """
    Uses the LLM to refine the search query for better results.
    """
    prompt = f"Refine the following search query to be more effective for finding relevant files: '{query}'"
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            store=True,
            messages=[{"role": "user", "content": prompt}]
        )
        refined_query = completion.choices[0].message.content.strip()
        return refined_query
    except Exception as e:
        print(f"Error in refine_query: {e}")
        return query

def filter_relevant_files(file_paths, keywords, min_matches=2):
    """
    Filters the list of file paths by retaining only those where the file name
    (in lowercase) contains at least min_matches of the generated keywords.
    
    :param file_paths: List of file paths.
    :param keywords: List of keywords generated for the query.
    :param min_matches: Minimum number of keyword matches required.
    :return: Filtered list of file paths.
    """
    filtered = []
    for fp in file_paths:
        count = sum(1 for kw in keywords if kw.lower() in fp.lower())
        if count >= min_matches:
            filtered.append(fp)
    return filtered

def search_agent(query, timeout=30, name_threshold=3):
    """
    Agentic search loop that uses LLM-driven decision making to choose a search strategy.
    This version prioritizes name-based search results. For each candidate directory:
      - Runs name-based search for each generated keyword.
      - If the number of name-based hits is below name_threshold, supplements with content-based search.
      - After combining results, filters them to retain only files whose paths contain
        at least a minimum number of the generated keywords.
    
    :param query: The search query.
    :param timeout: Timeout in seconds for ripgrep calls.
    :param name_threshold: Minimum number of name-based results to consider them sufficient.
    :return: Tuple of (best_file, all_results).
    """
    # Step 1: Narrow down candidate directories using iterative traversal.
    candidate_dirs = []
    for root in ROOT_DIRS:
        selected_dir = iterative_directory_traversal(root, query)
        candidate_dirs.append(selected_dir)
    candidate_dirs = list(set(candidate_dirs))
    print(f"Candidate directories: {candidate_dirs}")
    
    # Generate effective keywords using the LLM.
    keywords = generate_keywords(query)
    print(f"Generated keywords: {keywords}")
    
    results = set()
    
    # Step 2: For each candidate directory, prioritize name-based search.
    for d in candidate_dirs:
        d = os.path.expanduser(d)
        dir_name_results = []
        dir_content_results = []
        # Run name-based search for all keywords.
        for kw in keywords:
            res = name_based_search(kw, d, timeout=timeout)
            dir_name_results.extend(res)
        # If the name-based search in this directory yields fewer than the threshold, add content-based search.
        if len(dir_name_results) < name_threshold:
            for kw in keywords:
                res = content_based_search(kw, d, timeout=timeout)
                dir_content_results.extend(res)
        # Combine results (name-based results are given higher weight).
        combined = set(dir_name_results + dir_content_results)
        results.update(combined)
    
    all_results = list(results)
    
    # Step 3: Filter results to keep only those that have at least min_matches keyword occurrences.
    filtered_results = filter_relevant_files(all_results, keywords, min_matches=2)
    if filtered_results:
        all_results = filtered_results
    else:
        print("No files passed the relevance filter; using unfiltered results.")
    
    # If no results are found, refine the query and try again.
    if not all_results:
        print("No results found, refining query...")
        refined = refine_query(query)
        return search_agent(refined, timeout=timeout, name_threshold=name_threshold)
    
    # Optionally, use fzf to re-rank/refine the results.
    best_file = refine_fzf_selection(all_results, query) if all_results else None
    return best_file, all_results

def answer_query_from_files(query, file_paths):
    """
    Given a list of file paths, read their contents, build a vector store, and
    use a RAG pipeline to answer the query.
    
    :param query: The original query.
    :param file_paths: List of file paths to process.
    :return: The answer generated by the LLM with citations.
    """
    documents = []
    print("\n--- Document Extraction ---")
    for fp in file_paths:
        print(f"Processing file: {fp}")
        try:
            text = extract_text(fp)
        except Exception as e:
            print(f"Error extracting text from {fp}: {e}")
            text = ""
        if text and len(text) > 50:
            documents.append(Document(page_content=text, metadata={"source": fp}))
            print(f"Extracted {len(text)} characters. Preview: {text[:200]}...")
        else:
            extracted_length = len(text) if text else 0
            print(f"Skipped {fp}: Insufficient content extracted (length={extracted_length}).")
    
    if not documents:
        print("No documents with sufficient content were extracted.")
        return "No readable content found in the candidate files."
    
    print(f"\nTotal documents extracted: {len(documents)}")
    
    print("\n--- Building Vector Store ---")
    try:
        vectorstore = build_vector_store(documents)
        # Debug: print number of chunks in the vector store.
        if hasattr(vectorstore, "docstore"):
            num_chunks = len(vectorstore.docstore.search("dummy query", k=1000))
            print(f"Vector store built with approximately {num_chunks} chunks (approximation).")
        else:
            print("Vector store built.")
    except Exception as e:
        print(f"Error building vector store: {e}")
        return "Error building vector store from documents."
    
    print("\n--- Running RAG Pipeline ---")
    try:
        answer = run_qa_chain(vectorstore, query)
        print("RAG pipeline produced an answer.")
    except Exception as e:
        print(f"Error running RAG pipeline: {e}")
        answer = "Error generating answer from the documents."
    
    if not answer or answer.strip() == "":
        answer = "The documents did not provide enough context to generate an answer."
    
    return answer

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        best_file, results = search_agent(query)
        print(f"Best file: {best_file}")
        print("All results:")
        for r in results:
            print(r)
        print("\nAnswering query using the candidate files...")
        answer = answer_query_from_files(query, results)
        print("\nFinal Answer:")
        print(answer)
    else:
        print("Please provide a query.")
