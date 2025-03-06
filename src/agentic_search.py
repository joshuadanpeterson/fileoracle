"""
agentic_search.py

This module implements an agentic search loop for FileOracle.
The search agent uses LLM-driven decision-making to autonomously select a search strategy.
In this version, the agent prioritizes name-based search results:
- For each candidate directory (narrowed down via iterative traversal), it first runs a name-based search.
- If the number of name-based hits is below a specified threshold, it supplements with content-based search.
- After combining results, an additional filtering step scores and retains only those files
  that include at least a minimum number of the generated keywords in their file path.
- If no results remain, the query is refined and the search repeats.
"""

import os
from openai import OpenAI
from src.file_search import (
    name_based_search,
    content_based_search,
    generate_keywords,
    refine_fzf_selection
)
from src.directory_selector import iterative_directory_traversal

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

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        best_file, results = search_agent(query)
        print(f"Best file: {best_file}")
        print("All results:")
        for r in results:
            print(r)
    else:
        print("Please provide a query.")
