"""
main.py

The entry point for FileOracle. This script loads environment variables,
prompts the user for a query, and uses the Alfred integration module
to process and display the answer.
"""

from dotenv import load_dotenv

load_dotenv()  # Load variables from .env


def main():
    print("Welcome to FileOracle!")
    query = input("Enter your query: ")
    # For this demo, we'll use the Alfred integration function.
    from src.alfred_integration import alfred_main

    alfred_main(query)


if __name__ == "__main__":
    main()
