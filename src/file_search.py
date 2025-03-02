"""
file_search.py

This module provides functionality to search for keywords in text-based files
across multiple directories using ripgrep.
"""

import subprocess
import os

# Define the default directories to search.
DEFAULT_DIRECTORIES = [
    "/Users/joshpeterson/Library/CloudStorage/Dropbox/",
    "/Users/joshpeterson/Library/CloudStorage/GoogleDrive-joshuadanpeterson@gmail.com/My Drive/",
    "~/Documents",
]


def search_files(query, directories=DEFAULT_DIRECTORIES):
    """
    Search for a keyword in all text-based files using ripgrep across multiple directories.

    :param query: The search query.
    :param directories: A list of directories to search.
    :return: List of file paths matching the query.
    """
    results = []
    for directory in directories:
        # Expand user home directory symbol if present.
        directory = os.path.expanduser(directory)
        command = ["rg", "--files-with-matches", query, directory, "--ignore-case"]
        result = subprocess.run(command, capture_output=True, text=True)
        files = result.stdout.strip().split("\n")
        results.extend([f for f in files if f])
    return results
