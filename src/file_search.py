"""
file_search.py

This module provides functionality to search for files using a combined approach:
1. Name-based search using ripgrep's --files mode with glob patterns.
2. Content-based search using ripgrep's --files-with-matches mode.
It leverages an LLM to generate effective search keywords from the user's query,
and includes a function to refine fzf results to select the most relevant file.
"""

import subprocess
import os
from openai import OpenAI

client = OpenAI()

# Define default directories (used if not provided externally)
DEFAULT_DIRECTORIES = [
    '/Users/joshpeterson/Library/CloudStorage/Dropbox/',
    '/Users/joshpeterson/Library/CloudStorage/GoogleDrive-joshuadanpeterson@gmail.com/My Drive/',
    os.path.expanduser('~/Documents')
]

def generate_keywords(query, num_keywords=3):
    """
    Use the LLM to generate a list of effective search keywords for the query,
    and expand each keyword with underscore and dash variants.
    
    :param query: The original search query.
    :param num_keywords: Number of keywords to generate.
    :return: List of keywords.
    """
    prompt = (
        f"Given the search query: '{query}', generate {num_keywords} concise keywords "
        "that best capture the essence of the query for file searching. "
        "Return the keywords as a comma-separated list."
    )
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            store=True,
            messages=[{"role": "user", "content": prompt}]
        )
        content = completion.choices[0].message.content
        base_keywords = [kw.strip() for kw in content.split(",") if kw.strip()]
        expanded_keywords = set()
        for kw in base_keywords:
            expanded_keywords.add(kw)
            if " " in kw:
                expanded_keywords.add(kw.replace(" ", "_"))
                expanded_keywords.add(kw.replace(" ", "-"))
        return list(expanded_keywords) if expanded_keywords else [query]
    except Exception as e:
        print(f"Error generating keywords: {e}")
        return [query]

def name_based_search(keyword, directory, timeout=30):
    """
    Search file names in the given directory for matches to the keyword
    using ripgrep's --files mode with a glob pattern.
    
    :param keyword: The search keyword.
    :param directory: Directory to search.
    :param timeout: Timeout for the ripgrep command.
    :return: List of matching file paths.
    """
    command = [
        "rg",
        "--files",
        "--ignore-case",
        f"-g*{keyword}*",
        directory
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        files = result.stdout.strip().split("\n")
        return [f for f in files if f]
    except subprocess.TimeoutExpired:
        print(f"Timeout expired for name-based search '{keyword}' in '{directory}'")
    except Exception as e:
        print(f"Error in name-based search for '{keyword}' in '{directory}': {e}")
    return []

def content_based_search(keyword, directory, timeout=30):
    """
    Search file contents in the given directory for matches to the keyword
    using ripgrep's --files-with-matches mode.
    
    :param keyword: The search keyword.
    :param directory: Directory to search.
    :param timeout: Timeout for the ripgrep command.
    :return: List of matching file paths.
    """
    command = [
        "rg",
        "--files-with-matches",
        "--ignore-case",
        keyword,
        directory
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        files = result.stdout.strip().split("\n")
        return [f for f in files if f]
    except subprocess.TimeoutExpired:
        print(f"Timeout expired for content-based search '{keyword}' in '{directory}'")
    except Exception as e:
        print(f"Error in content-based search for '{keyword}' in '{directory}': {e}")
    return []

def search_files(query, directories=DEFAULT_DIRECTORIES, timeout=30):
    """
    Perform a combined search for files matching the query across multiple directories.
    It uses LLM-generated keywords to perform both name-based and content-based searches,
    and returns a list of unique file paths.
    
    :param query: The search query.
    :param directories: List of directories to search.
    :param timeout: Timeout (in seconds) for each ripgrep command.
    :return: List of file paths matching any generated keyword.
    """
    results = set()
    keywords = generate_keywords(query)
    print(f"Generated keywords: {keywords}")
    
    for directory in directories:
        directory = os.path.expanduser(directory)
        for kw in keywords:
            name_results = name_based_search(kw, directory, timeout=timeout)
            content_results = content_based_search(kw, directory, timeout=timeout)
            for f in (name_results + content_results):
                results.add(f)
    return list(results)

def refine_fzf_selection(selected_files, query):
    """
    Given a list of candidate file paths and the original query,
    use the LLM to determine which file is most likely to contain the relevant information.
    
    :param selected_files: List of file paths.
    :param query: The search query.
    :return: The file path deemed most relevant by the LLM.
    """
    prompt = (
        f"I have the following candidate files from a search based on the query '{query}':\n"
        f"{chr(10).join(selected_files)}\n\n"
        "Based on their file names and context, which file is most likely to contain the information needed for the query? "
        "Please respond with only the file path."
    )
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            store=True,
            messages=[{"role": "user", "content": prompt}]
        )
        best_file = completion.choices[0].message.content.strip()
        return best_file
    except Exception as e:
        print(f"Error refining fzf selection: {e}")
        return selected_files[0] if selected_files else None
