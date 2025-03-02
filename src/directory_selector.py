"""
directory_selector.py

This module implements an iterative directory traversal function.
At each level, it uses an LLM to decide which subdirectory is most likely
to contain files relevant to a given query, thereby winnowing down the search path.
"""

import os
from openai import OpenAI

client = OpenAI()


def select_best_subdirectory(current_dir, query):
    """
    Use the LLM to select the best subdirectory within current_dir based on the query.

    :param current_dir: The current directory to evaluate.
    :param query: The user's search query.
    :return: The selected subdirectory name or None if none is suitable.
    """
    try:
        subdirs = [
            d
            for d in os.listdir(current_dir)
            if os.path.isdir(os.path.join(current_dir, d))
        ]
    except Exception as e:
        print(f"Error listing subdirectories in {current_dir}: {e}")
        return None

    if not subdirs:
        return None

    # Construct a prompt listing the subdirectories.
    prompt = (
        f"Given the query: '{query}', which of the following subdirectories under '{current_dir}' is most likely "
        "to contain relevant files? Respond with the exact subdirectory name. If none seem relevant, respond with 'none'.\n"
        f"Subdirectories: {', '.join(subdirs)}"
    )

    try:
        completion = client.chat.completions.create(
            model="gpt-4o", store=True, messages=[{"role": "user", "content": prompt}]
        )
        answer = completion.choices[0].message.content.strip().lower()
        if answer == "none" or answer not in [d.lower() for d in subdirs]:
            return None
        # Return the matching subdirectory (case sensitive) from the list.
        for d in subdirs:
            if d.lower() == answer:
                return d
    except Exception as e:
        print(f"Error during LLM subdirectory selection: {e}")
        return None


def iterative_directory_traversal(root_dir, query):
    """
    Iteratively traverse directories starting from root_dir by using the LLM to narrow
    down the most likely path that contains relevant files for the given query.

    :param root_dir: The starting directory.
    :param query: The search query.
    :return: The final directory path that is most likely to contain the relevant files.
    """
    current_dir = root_dir
    while True:
        best_subdir = select_best_subdirectory(current_dir, query)
        if best_subdir is None:
            break
        new_dir = os.path.join(current_dir, best_subdir)
        print(f"Descending into: {new_dir}")
        current_dir = new_dir
    return current_dir


# Example usage:
if __name__ == "__main__":
    root = "/Users/joshpeterson/Library/CloudStorage/Dropbox/"
    query = "budget projections"
    final_path = iterative_directory_traversal(root, query)
    print(f"Final selected directory: {final_path}")
