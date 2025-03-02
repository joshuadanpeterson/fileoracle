"""
directory_selector.py

This module uses an LLM to filter default directories based on a query.
It summarizes each directory's contents and asks the LLM which directories 
are most likely to contain relevant documents.
"""

import os
import openai

# Default directories to consider.
DEFAULT_DIRECTORIES = [
    '/Users/joshpeterson/Library/CloudStorage/Dropbox/',
    '/Users/joshpeterson/Library/CloudStorage/GoogleDrive-joshuadanpeterson@gmail.com/My Drive/',
    os.path.expanduser('~/Documents')
]

def get_directory_summary(directories):
    """
    Creates a summary for each directory by listing a few sample filenames.
    
    :param directories: List of directory paths.
    :return: Dictionary mapping directory paths to a list of sample filenames.
    """
    summary = {}
    for directory in directories:
        try:
            # List up to 5 items for brevity.
            files = os.listdir(directory)[:5]
            summary[directory] = files
        except Exception as e:
            summary[directory] = [f"Error: {e}"]
    return summary

def select_relevant_directories(query, directories=DEFAULT_DIRECTORIES):
    """
    Uses an LLM to select which directories are most likely to contain relevant documents.
    
    :param query: The user's query.
    :param directories: List of directory paths to consider.
    :return: List of directories recommended for searching.
    """
    # Generate a summary of the directories.
    summary = get_directory_summary(directories)
    # Create a prompt that explains the context.
    prompt = (
        "I have the following directories with sample file names:\n"
        f"{summary}\n\n"
        f"Based on the query '{query}', which directories are most likely to contain relevant documents? "
        "List one directory per line, exactly as shown in the keys above."
    )
    
    # Call the LLM. Ensure OPENAI_API_KEY is set and openai is configured.
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    # Split the response into lines.
    selected = response["choices"][0]["message"]["content"].splitlines()
    # Validate that the directories are from our list.
    recommended = [d.strip() for d in selected if d.strip() in directories]
    # If none are recommended, fall back to all directories.
    return recommended if recommended else directories
