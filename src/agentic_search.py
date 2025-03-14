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

def detect_relevant_paths(query):
    """
    Uses the LLM to analyze the query for path references and map them to actual filesystem paths.
    
    The function detects mentions of common folders like 'Dropbox', 'Google Drive', 'Documents', 
    'Downloads', etc., and maps them to their actual paths in the filesystem.
    
    It also handles explicit directory requests like "restrict searches to ..." or
    "look in my Documents folder".
    
    :param query: The search query string
    :return: List of relevant directories to search, or None if no specific paths detected
    """
    # Define common folder mappings
    FOLDER_MAPPINGS = {
        "dropbox": "/Users/joshpeterson/Library/CloudStorage/Dropbox/",
        "dropsyncfiles": "/Users/joshpeterson/Library/CloudStorage/Dropbox/DropsyncFiles",
        "google drive": "/Users/joshpeterson/Library/CloudStorage/GoogleDrive-joshuadanpeterson@gmail.com/My Drive/",
        "documents": os.path.expanduser("~/Documents"),
        "downloads": os.path.expanduser("~/Downloads"),
        "desktop": os.path.expanduser("~/Desktop"),
        "pictures": os.path.expanduser("~/Pictures"),
        "music": os.path.expanduser("~/Music"),
        "videos": os.path.expanduser("~/Videos"),
        "home": os.path.expanduser("~"),
    }
    
    # First, check for explicit override instructions (keeping the old logic for backward compatibility)
    lower_query = query.lower()
    phrase = "restrict searches to"
    if phrase in lower_query:
        override_text = lower_query.split(phrase, 1)[1].strip()
        if "dropbox" in override_text and "dropsyncfiles" in override_text:
            return ["/Users/joshpeterson/Library/CloudStorage/Dropbox/DropsyncFiles"]
    
    # Use the LLM to identify potential directories mentioned in the query
    prompt = f"""
    Analyze the following search query and identify any folder or location references that could 
    help narrow down where to search for files:
    
    Query: "{query}"
    
    If the query mentions or implies any specific locations like Dropbox, Google Drive, Documents, etc.,
    list them one per line. If the query does not mention or strongly imply any specific location,
    respond with "None".
    
    For example:
    - "find my tax documents from 2022" might imply "Documents" or "Dropbox"
    - "code for my Python project" might imply "Documents" or a programming folder
    - "photos from my vacation" might imply "Pictures"
    
    Output only the folder names, one per line, or "None".
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        
        folder_suggestions = response.choices[0].message.content.strip().split('\n')
        
        # Process the suggestions
        if len(folder_suggestions) == 1 and folder_suggestions[0].lower() == "none":
            return None
        
        # Map the suggested folders to actual paths
        relevant_paths = []
        for suggestion in folder_suggestions:
            suggestion = suggestion.lower().strip()
            
            # Check if the suggestion matches any of our mapped folders
            for folder_name, folder_path in FOLDER_MAPPINGS.items():
                if folder_name in suggestion:
                    if os.path.exists(folder_path) and folder_path not in relevant_paths:
                        relevant_paths.append(folder_path)
        
        return relevant_paths if relevant_paths else None
        
    except Exception as e:
        print(f"Error in detect_relevant_paths: {e}")
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
    # Detect relevant paths based on the query
    relevant_paths = detect_relevant_paths(query)
    if relevant_paths:
        print(f"Detected relevant directories: {relevant_paths}")
        candidate_dirs = []
        for path in relevant_paths:
            selected_dir = iterative_directory_traversal(path, query)
            candidate_dirs.append(selected_dir)
    else:
        print("No specific directories detected, using default ROOT_DIRS")
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
        try:
            print(f"Sending query to OpenAI Responses API with vector_store_id: {vector_store_id}")
            response = client.responses.create(
                model="gpt-4o-mini",
                input=query,
                context=[{
                    "type": "file_search",
                    "vector_store_id": vector_store_id
                }],
                temperature=0.0
            )
            
            # Debug: Print response structure
            print(f"Response received. Type: {type(response)}")
            print(f"Response attributes: {dir(response)}")
            
            # Validate response structure
            if not hasattr(response, 'choices') or not response.choices:
                print(f"Error: Invalid response structure - missing choices: {response}")
                return "Error: The API response did not contain expected 'choices' data."
                
            # Access first choice
            choice = response.choices[0]
            if not hasattr(choice, 'message'):
                print(f"Error: First choice missing message attribute: {choice}")
                return "Error: The API response contained malformed choice data."
                
            # Get message content
            message = choice.message
            if not hasattr(message, 'content') or not message.content:
                print(f"Error: Message has no content. Message attributes: {dir(message)}")
                return "No response content was generated by the API."
                
            answer_text = message.content
            print(f"Answer text successfully extracted, length: {len(answer_text)}")
            
            # Format citations if available
            citations = []
            if hasattr(message, 'annotations') and message.annotations:
                print(f"Found {len(message.annotations)} annotations")
                for i, annotation in enumerate(message.annotations):
                    print(f"Annotation {i+1} type: {type(annotation)}, attributes: {dir(annotation)}")
                    
                    # Try different ways to access file information
                    file_path = None
                    if hasattr(annotation, 'file_path') and annotation.file_path:
                        file_path = annotation.file_path
                    elif hasattr(annotation, 'file_citation') and annotation.file_citation:
                        if hasattr(annotation.file_citation, 'file_path'):
                            file_path = annotation.file_citation.file_path
                    elif hasattr(annotation, 'text'):
                        # Sometimes annotations include citation info in text
                        file_path = f"Citation: {annotation.text}"
                        
                    if file_path:
                        citation = f"[{i+1}] {file_path}"
                        citations.append(citation)
                        print(f"Added citation: {citation}")
            else:
                print("No annotations found in the response")
            
            # Combine answer with citations
            if citations:
                final_answer = f"{answer_text}\n\nSources:\n" + "\n".join(citations)
            else:
                final_answer = answer_text
                
            return final_answer
            
        except Exception as e:
            import traceback
            print(f"Error using OpenAI's Responses API: {e}")
            print(f"Error details: {traceback.format_exc()}")
            print("Falling back to traditional RAG pipeline...")
            # Fall back to traditional RAG if the API call fails
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
