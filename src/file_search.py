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


def content_based_search(keyword, directory, timeout=30):
    """
    Search file contents in the given directory for matches to 'keyword'.
    This uses ripgrep's --files-with-matches mode.

    :param keyword: The keyword to match in file contents.
    :param directory: Directory path to search in.
    :param timeout: Timeout in seconds for the ripgrep command (default 30s).
    :return: A list of file paths whose contents match the keyword.
    """
    command = ["rg", "--files-with-matches", "--ignore-case", keyword, directory]
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


def search_files(query, directories=DEFAULT_DIRECTORIES, timeout=30):
    """
    Perform a combined name-based and content-based search for files matching the query
    across multiple directories. Uses LLM-generated keywords (with underscore and dash variants).

    :param query: The search query.
    :param directories: A list of directories to search.
    :param timeout: Timeout (in seconds) for each ripgrep query (default 30s).
    :return: List of file paths matching any of the generated keywords (by name or content).
    """
    results = set()

    # Generate effective keywords using the LLM.
    keywords = generate_keywords(query)
    print(f"Generated keywords: {keywords}")

    # Iterate through each directory and keyword.
    for directory in directories:
        directory = os.path.expanduser(directory)
        for keyword in keywords:
            # Combine name-based and content-based search results.
            name_results = name_based_search(keyword, directory, timeout=timeout)
            content_results = content_based_search(keyword, directory, timeout=timeout)
            combined = name_results + content_results
            for f in combined:
                results.add(f)
    return list(results)
