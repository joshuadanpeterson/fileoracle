"""
main.py

The entry point for FileOracle. This script loads environment variables,
prompts the user for a query, and uses the agentic search loop to process
and display the search results.
"""

from dotenv import load_dotenv

load_dotenv()  # Load variables from .env


def main():
    print("Welcome to FileOracle!")
    query = input("Enter your query: ")
    # Use the new agentic search loop.
    from src.agentic_search import search_agent

    best_file, results = search_agent(query)
    print(f"Best file: {best_file}")
    print("All results:")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
