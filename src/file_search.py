"""
file_search.py

This module provides functionality to search for keywords in text-based files
across multiple directories using ripgrep. It leverages an LLM to generate
effective search keywords from the user's query and limits each ripgrep call
to a few seconds to prevent long searches. Additionally, it integrates fzf
to allow for interactive, fuzzy selection of files from the ripgrep results.
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
    Use the LLM to generate a list of effective search keywords for the query.

    :param query: The original search query.
    :param num_keywords: Number of keywords to generate.
    :return: List of keywords.
    """
    prompt = (
        f"Given the search query: '{query}', generate {num_keywords} concise keywords "
        "that best capture the essence of the query for file searching. Return the keywords as a comma-separated list."
    )

    try:
        completion = client.chat.completions.create(
            model="gpt-4o", store=True, messages=[{"role": "user", "content": prompt}]
        )
        content = completion.choices[0].message.content
        keywords = [kw.strip() for kw in content.split(",") if kw.strip()]
        return keywords if keywords else [query]
    except Exception as e:
        print(f"Error generating keywords: {e}")
        # Fallback to original query if LLM fails.
        return [query]


def search_files(query, directories=DEFAULT_DIRECTORIES, timeout=3):
    """
    Search for files matching the query across multiple directories.
    Uses LLM-generated keywords to run multiple ripgrep searches with a timeout.

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
            command = [
                "rg",
                "--files-with-matches",
                keyword,
                directory,
                "--ignore-case",
            ]
            try:
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=timeout
                )
                files = result.stdout.strip().split("\n")
                results.extend([f for f in files if f])
            except subprocess.TimeoutExpired:
                print(
                    f"Timeout expired for keyword '{keyword}' in directory '{directory}'"
                )
            except Exception as e:
                print(
                    f"Error running ripgrep for keyword '{keyword}' in '{directory}': {e}"
                )
    # Remove duplicates.
    return list(set(results))


def search_files_with_fzf(query, directories=DEFAULT_DIRECTORIES, timeout=3):
    """
    Enhanced file search that combines ripgrep with fzf for interactive selection.

    :param query: The search query.
    :param directories: A list of directories to search.
    :param timeout: Timeout (in seconds) for each ripgrep query.
    :return: List of selected file paths.
    """
    results = []
    # Generate keywords from the query.
    keywords = generate_keywords(query)
    print(f"Generated keywords: {keywords}")

    for directory in directories:
        directory = os.path.expanduser(directory)
        for keyword in keywords:
            rg_command = [
                "rg",
                "--files-with-matches",
                keyword,
                directory,
                "--ignore-case",
            ]
            try:
                # Run ripgrep with a timeout.
                rg_result = subprocess.run(
                    rg_command, capture_output=True, text=True, timeout=timeout
                )
                files = rg_result.stdout.strip().split("\n")
                files = [f for f in files if f]
                if files:
                    # Use fzf to interactively select one or more files.
                    fzf_command = "fzf --multi --select-1 --exit-0"
                    fzf_proc = subprocess.Popen(
                        fzf_command,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        shell=True,
                        text=True,
                    )
                    input_data = "\n".join(files)
                    selected, _ = fzf_proc.communicate(input=input_data)
                    if selected:
                        selected_files = [
                            line.strip()
                            for line in selected.splitlines()
                            if line.strip()
                        ]
                        results.extend(selected_files)
            except subprocess.TimeoutExpired:
                print(
                    f"Timeout expired for keyword '{keyword}' in directory '{directory}'"
                )
            except Exception as e:
                print(
                    f"Error running search for keyword '{keyword}' in '{directory}': {e}"
                )
    return list(set(results))
