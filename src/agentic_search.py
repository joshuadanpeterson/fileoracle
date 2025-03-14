"""
agentic_search.py

This module implements an agentic search loop for FileOracle.
The search agent uses LLM-driven decision-making to autonomously select a search strategy.
In this version, the agent prioritizes name-based search results:
- For each candidate directory (narrowed down via iterative traversal), it first runs a name-based search.
- If the number of name-based hits is below a specified threshold, it supplements with content-based search.
- After combining results, an additional filtering step retains only files whose paths contain
  a minimum number of the generated keywords.
- Optionally, a filter keyword can further restrict results, and max_results limits the number returned.
- Finally, the documents corresponding to these candidate file paths are read,
  aggregated into a vector store, and a RAG pipeline is used to answer the user’s query.
"""

import os
from openai import OpenAI
from src.file_search import (
    name_based_search,
    content_based_search,
    generate_keywords,
    refine_fzf_selection,
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
    os.path.expanduser("~/Documents"),
]

def extract_override_directory(query):
    """
    Checks the query for an override instruction such as 
    "restrict searches to Dropbox in the DropSyncFiles folder"
    and returns the corresponding root directory path if found.
    
    For example, if the query contains both "dropbox" and "dropsyncfiles",
    returns "/Users/joshpeterson/Library/CloudStorage/Dropbox/DropsyncFiles".
    """
    lower_query = query.lower()
    phrase = "restrict searches to"
    if phrase in lower_query:
        # Extract the portion after the phrase.
        override_text = lower_query.split(phrase, 1)[1].strip()
        # Simple heuristic: if override_text mentions both "dropbox" and "dropsyncfiles",
        # return the Dropbox DropsyncFiles folder.
        if "dropbox" in override_text and "dropsyncfiles" in override_text:
            return "/Users/joshpeterson/Library/CloudStorage/Dropbox/DropsyncFiles"
    return None

def call_llm_for_decision(prompt):
    """
    Calls the LLM with the given prompt and returns its decision as a string.
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-4o", store=True, messages=[{"role": "user", "content": prompt}]
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
            model="gpt-4o", store=True, messages=[{"role": "user", "content": prompt}]
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

def search_agent(query, timeout=30, name_threshold=3, max_results=None, filter_keyword=None):
    """
    Agentic search loop that uses LLM-driven decision making to choose a search strategy.
    This version prioritizes name-based search results. For each candidate directory:
      - Runs name-based search for each generated keyword.
      - If the number of name-based hits is below name_threshold, supplements with content-based search.
      - After combining results, filters them to retain only files whose paths contain
        at least a minimum number of the generated keywords.
      - Optionally applies an additional filter using filter_keyword and limits results to max_results.
    
    :param query: The search query.
    :param timeout: Timeout in seconds for ripgrep calls.
    :param name_threshold: Minimum number of name-based results to consider them sufficient.
    :param max_results: Optional maximum number of results to return.
    :param filter_keyword: Optional additional keyword to filter file paths.
    :return: Tuple of (best_file, all_results).
    """
    # Check for an override directory in the query.
    override_dir = extract_override_directory(query)
    if override_dir:
        print(f"Overriding candidate directories with: {override_dir}")
        candidate_dirs = [iterative_directory_traversal(override_dir, query)]
    else:
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
    
    # For each candidate directory, prioritize name-based search.
    for d in candidate_dirs:
        d = os.path.expanduser(d)
        dir_name_results = []
        dir_content_results = []
        for kw in keywords:
            res = name_based_search(kw, d, timeout=timeout)
            dir_name_results.extend(res)
        if len(dir_name_results) < name_threshold:
            for kw in keywords:
                res = content_based_search(kw, d, timeout=timeout)
                dir_content_results.extend(res)
        combined = set(dir_name_results + dir_content_results)
        results.update(combined)
    
    all_results = list(results)
    filtered_results = filter_relevant_files(all_results, keywords, min_matches=2)
    if filtered_results:
        all_results = filtered_results
    else:
        print("No files passed the relevance filter; using unfiltered results.")
    
    if filter_keyword:
        all_results = [fp for fp in all_results if filter_keyword.lower() in fp.lower()]
    
    if max_results is not None:
        all_results = all_results[:max_results]
    
    if not all_results:
        print("No results found, refining query...")
        refined = refine_query(query)
        return search_agent(refined, timeout=timeout, name_threshold=name_threshold,
                            max_results=max_results, filter_keyword=filter_keyword)
    
    best_file = refine_fzf_selection(all_results, query) if all_results else None
    return best_file, all_results

def answer_query_from_files(query, file_paths, use_responses_api=False, vector_store_id=None):
    """
    Given a list of file paths, read their contents, build a vector store, and
    use a RAG pipeline to answer the query.
    
    Optionally, if use_responses_api is True and a vector_store_id is provided,
    the function will use OpenAI's Responses API.
    
    :param query: The original query.
    :param file_paths: List of file paths to process.
    :param use_responses_api: Boolean flag to use the Responses API.
    :param vector_store_id: The hosted vector store ID.
    :return: The answer generated by the LLM with citations, or an API response.
    """
    if use_responses_api and vector_store_id:
        print("Using OpenAI's Responses API for answer generation...")
        response = client.responses.create(
            model="gpt-4o-mini",
            input=query,
            tools=[{
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
                "max_num_results": 10
            }]
        )
        return response
    else:
        documents = []
        print("\n--- Document Extraction ---")
        for fp in file_paths:
            print(f"Extracting text from: {fp}")
            try:
                text = extract_text(fp)
            except Exception as e:
                print(f"Error extracting text from {fp}: {e}")
                text = ""
            if text and len(text) > 50:
                documents.append(Document(page_content=text, metadata={"source": fp}))
                print(f"Extracted {len(text)} characters from {fp}")
            else:
                extracted_length = len(text) if text else 0
                print(f"Skipped {fp}: Insufficient content extracted (length={extracted_length}).")
    
        if not documents:
            print("No documents with sufficient content were extracted.")
            return "No readable content found in the candidate files."
    
        print(f"\nBuilding vector store from {len(documents)} documents...")
        try:
            vectorstore = build_vector_store(documents)
        except Exception as e:
            print(f"Error building vector store: {e}")
            return "Error building vector store from documents."
    
        print("Running RAG pipeline to answer the query...")
        try:
            answer = run_qa_chain(vectorstore, query)
        except Exception as e:
            print(f"Error running RAG pipeline: {e}")
            answer = "Error generating answer from the documents."
    
        if not answer or answer.strip() == "":
            answer = "The documents did not provide enough context to generate an answer."
    
        print("RAG pipeline output received.")
        return answer

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        best_file, results = search_agent(query, max_results=10, filter_keyword="dropsyncfiles")
        print(f"Best file: {best_file}")
        print("All candidate files:")
        for r in results:
            print(r)
        print("\nAnswering query using the candidate files...")
        # Ensure you set use_responses_api=False if you don't have a hosted vector store.
        answer = answer_query_from_files(query, results, use_responses_api=False)
        print("\nFinal Answer:")
        print(answer)
    else:
        print("Please provide a query.")
