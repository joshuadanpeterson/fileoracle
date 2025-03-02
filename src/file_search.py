"""
file_search.py

This module provides functionality to search for keywords in text-based files
across multiple directories using ripgrep. It leverages an LLM to generate
effective search keywords from the user's query, expands them with underscore
and dash variants, and limits each ripgrep call to a few seconds to prevent long searches.
"""

import subprocess
import os
from openai import OpenAI

client = OpenAI()

# Define the default directories to search.
DEFAULT_DIRECTORIES = [
    '/Users/joshpeterson/Library/CloudStorage/Dropbox/',
    '/Users/joshpeterson/Library/CloudStorage/GoogleDrive-joshuadanpeterson@gmail.com/My Drive/',
    os.path.expanduser('~/Documents')
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
            model="gpt-4o",
            store=True,
            messages=[
                {"role": "user", "content": prompt}
            ]
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

def search_files(query, directories=DEFAULT_DIRECTORIES, timeout=3):
    """
    Search for files matching the query across multiple directories.
    Uses LLM-generated keywords (with underscore and dash variants) to run multiple ripgrep searches with a timeout.
    
    :param query: The search query.
    :param directories: A list of directories to search.
    :param timeout: Timeout (in seconds) for each ripgrep query.
    :return: List of file paths matching any of the generated keywords.
    """
    results = []
    
    # Generate effective keywords using the LLM.
    keywords = generate_keywords(query)
    print(f"Generated keywords: {keywords}")
    
    # Iterate through each directory and keyword.
    for directory in directories:
        directory = os.path.expanduser(directory)
        for keyword in keywords:
            command = ["rg", "--files-with-matches", keyword, directory, "--ignore-case"]
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
                files = result.stdout.strip().split("\n")
                results.extend([f for f in files if f])
            except subprocess.TimeoutExpired:
                print(f"Timeout expired for keyword '{keyword}' in directory '{directory}'")
            except Exception as e:
                print(f"Error running ripgrep for keyword '{keyword}' in '{directory}': {e}")
    # Remove duplicates.
    return list(set(results))
