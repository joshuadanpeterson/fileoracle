"""
file_search.py

This module provides functionality to search for files using a combined approach:
1) Name-based search using ripgrep's --files and glob patterns for each keyword.
2) Content-based search using ripgrep's --files-with-matches.

It leverages an LLM to generate effective search keywords from the user's query,
including underscore and dash variants, and uses a longer timeout to allow ripgrep
to fully search large directories.
"""

import subprocess
import os
from openai import OpenAI

client = OpenAI()

# Define the default directories to search.
DEFAULT_DIRECTORIES = [
    "/Users/joshpeterson/Library/CloudStorage/Dropbox/",
    "/Users/joshpeterson/Library/CloudStorage/GoogleDrive-joshuadanpeterson@gmail.com/My Drive/",
    os.path.expanduser("~/Documents"),
]


def generate_keywords(query, num_keywords=3):
    """
    Use the LLM to generate a list of effective search keywords for the query,
    and then expand each keyword with underscore and dash variants if applicable.

    :param query: The original search query.
    :param num_keywords: Number of keywords to generate.
    :return: List of keywords including underscore and dash variants.
    """
    prompt = (
        f"Given the search query: '{query}', generate {num_keywords} concise keywords "
        "that best capture the essence of the query for file searching. "
        "Return the keywords as a comma-separated list."
    )

    try:
        completion = client.chat.completions.create(
            model="o3-mini", store=True, messages=[{"role": "user", "content": prompt}]
        )
        content = completion.choices[0].message.content
        base_keywords = [kw.strip() for kw in content.split(",") if kw.strip()]

        # Expand keywords to include underscore and dash variants.
        expanded_keywords = set()
        for kw in base_keywords:
            expanded_keywords.add(kw)
            if " " in kw:
                expanded_keywords.add(kw.replace(" ", "_"))
                expanded_keywords.add(kw.replace(" ", "-"))
        return list(expanded_keywords) if expanded_keywords else [query]
    except Exception as e:
        print(f"Error generating keywords: {e}")
        # Fallback to original query if LLM fails.
        return [query]


def name_based_search(keyword, directory, timeout=30):
    """
    Search file names (not contents) in the given directory for matches to 'keyword'.
    This uses ripgrep's --files mode plus a glob pattern.

    :param keyword: The keyword to match in file names.
    :param directory: Directory path to search in.
    :param timeout: Timeout in seconds for the ripgrep command (default 30s).
    :return: A list of file paths whose names match the keyword.
    """
    command = ["rg", "--files", "--ignore-case", f"-g*{keyword}*", directory]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout
        )
        files = result.stdout.strip().split("\n")
        return [f for f in files if f]
    except subprocess.TimeoutExpired:
        print(f"Timeout expired for name-based search '{keyword}' in '{directory}'")
    except Exception as e:
        print(f"Error in name-based search for '{keyword}' in '{directory}': {e}")
    return []


def content_based_search(keyword, directory, timeout=30, max_depth=5, max_filesize="5M", threads=4):
    """
    Search file contents in the given directory for matches to 'keyword'.
    This uses ripgrep's --files-with-matches mode with optimizations:
    - Limits to text files only
    - Sets maximum directory depth
    - Sets maximum file size
    - Excludes common large directories
    - Controls CPU usage with threads parameter

    :param keyword: The keyword to match in file contents.
    :param directory: Directory path to search in.
    :param timeout: Timeout in seconds for the ripgrep command (default 30s).
    :param max_depth: Maximum directory depth to search (default 5).
    :param max_filesize: Maximum file size to search (default "5M").
    :param threads: Number of threads to use for searching (default 4).
    :return: A list of file paths whose contents match the keyword.
    """
    # Common directories and files to exclude
    exclude_patterns = [
        "--glob", "!**/.git/**",
        "--glob", "!**/node_modules/**",
        "--glob", "!**/.venv/**",
        "--glob", "!**/venv/**",
        "--glob", "!**/__pycache__/**",
        "--glob", "!**/build/**",
        "--glob", "!**/dist/**",
        "--glob", "!**/*.jpg",
        "--glob", "!**/*.jpeg",
        "--glob", "!**/*.png",
        "--glob", "!**/*.gif",
        "--glob", "!**/*.mp4",
        "--glob", "!**/*.mp3",
        "--glob", "!**/*.zip",
        "--glob", "!**/*.tar",
        "--glob", "!**/*.gz",
    ]
    
    command = [
        "rg",
        "--files-with-matches",
        "--ignore-case",
        "--type", "text",          # Only search text files
        "--max-depth", str(max_depth),  # Limit directory depth
        "--max-filesize", max_filesize, # Limit file size
        "--threads", str(threads),      # Control CPU usage
    ]
    
    # Add all exclude patterns
    command.extend(exclude_patterns)
    
    # Add the search keyword and directory
    command.extend([keyword, directory])
    
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout
        )
        files = result.stdout.strip().split("\n")
        return [f for f in files if f]
    except subprocess.TimeoutExpired:
        print(f"Timeout expired for content-based search '{keyword}' in '{directory}'")
    except Exception as e:
        print(f"Error in content-based search for '{keyword}' in '{directory}': {e}")
    return []


def search_files(query, directories=DEFAULT_DIRECTORIES, timeout=30, name_threshold=5, max_depth=5, max_filesize="5M", threads=4):
    """
    Perform a combined search: run a name-based search and, if the results are insufficient,
    supplement with a content-based search.
    
    :param query: The search query.
    :param directories: A list of directories to search.
    :param timeout: Timeout (in seconds) for each ripgrep query.
    :param name_threshold: Minimum number of results from name-based search before skipping content search.
    :return: List of file paths matching any of the generated keywords.
    """
    results = set()
    
    # Generate effective keywords using the LLM.
    keywords = generate_keywords(query)
    print(f"Generated keywords: {keywords}")
    
    for directory in directories:
        directory = os.path.expanduser(directory)
        for keyword in keywords:
            # First, perform a name-based search.
            name_results = name_based_search(keyword, directory, timeout=timeout)
            # Add name-based results.
            results.update(name_results)
            
            # Only perform content-based search if name-based results are low.
            if len(name_results) < name_threshold:
                content_results = content_based_search(
                    keyword, 
                    directory, 
                    timeout=timeout,
                    max_depth=max_depth,
                    max_filesize=max_filesize,
                    threads=threads
                )
                results.update(content_results)
                
    return list(results)
