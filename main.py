"""
main.py

The entry point for FileOracle. This script loads environment variables,
prompts the user for a query, uses the agentic search loop to find candidate files,
and then reads those documents to answer the query using the RAG pipeline.
"""

from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env


def main():
    print("Welcome to FileOracle!")
    query = input("Enter your query: ")

    # Import the agentic search functions
    from src.agentic_search import search_agent, answer_query_from_files

    # Run the agentic search loop to get candidate files.
    best_file, results = search_agent(query)
    print(f"\nBest file: {best_file}")
    print("All candidate files:")
    for r in results:
        print(r)

    # Retrieve the vector store ID from the environment
    vector_store_id = os.getenv("VECTOR_STORE_ID")

    # Generate an answer using the candidate files.
    print("\nAnswering query using the candidate files...")
    answer = answer_query_from_files(
        query, results, use_responses_api=True, vector_store_id=vector_store_id
    )
    print("\nFinal Answer:")
    print(answer)


if __name__ == "__main__":
    main()

