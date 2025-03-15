#!/usr/bin/env python3
"""
create_vector_store.py

This script uploads files to OpenAI's Files API and creates a vector store
using the OpenAI VectorStores API. It saves the vector store ID to the .env file.
"""

import os
import argparse
import glob
from pathlib import Path
from dotenv import load_dotenv, set_key
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

def upload_file(file_path):
    """
    Upload a file to OpenAI's Files API.
    
    Args:
        file_path (str): Path to the file to upload.
        
    Returns:
        str: The file ID if upload is successful, None otherwise.
    """
    try:
        print(f"Uploading {file_path}...")
        with open(file_path, "rb") as file:
            response = client.files.create(
                file=file,
                purpose="assistants"
            )
        print(f"Successfully uploaded {file_path}, file ID: {response.id}")
        return response.id
    except Exception as e:
        print(f"Error uploading {file_path}: {e}")
        return None

def create_vector_store(file_ids, name=None):
    """
    Create a vector store using OpenAI's VectorStores API.
    
    Args:
        file_ids (list): List of file IDs to include in the vector store.
        name (str, optional): Name for the vector store.
        
    Returns:
        str: The vector store ID if creation is successful, None otherwise.
    """
    try:
        print(f"Creating vector store with {len(file_ids)} files...")
        expiration = None  # Set to None for no expiration, or use ISO format date
        
        response = client.vector_stores.create(
            name=name or "FileOracle Vector Store",
            file_ids=file_ids,
            expires_after=expiration
        )
        
        print(f"Successfully created vector store: {response.id}")
        return response.id
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

def update_env_file(vector_store_id):
    """
    Update the .env file with the new vector store ID.
    
    Args:
        vector_store_id (str): The vector store ID to save.
        
    Returns:
        bool: True if update is successful, False otherwise.
    """
    try:
        env_path = ".env"
        set_key(env_path, "VECTOR_STORE_ID", vector_store_id)
        print(f"Updated .env file with VECTOR_STORE_ID={vector_store_id}")
        return True
    except Exception as e:
        print(f"Error updating .env file: {e}")
        return False

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Create an OpenAI vector store from files.")
    parser.add_argument(
        "--files", 
        nargs="+", 
        help="Files to include in the vector store. Accepts glob patterns."
    )
    parser.add_argument(
        "--dir", 
        help="Directory containing files to include in the vector store."
    )
    parser.add_argument(
        "--extensions", 
        nargs="+", 
        default=[".txt", ".md", ".pdf", ".docx"],
        help="File extensions to include when using --dir (default: .txt .md .pdf .docx)"
    )
    parser.add_argument(
        "--name", 
        default=None,
        help="Name for the vector store (optional)"
    )
    parser.add_argument(
        "--update-env", 
        action="store_true",
        help="Update the .env file with the vector store ID"
    )
    return parser.parse_args()

def main():
    """Main function to run the script."""
    args = parse_arguments()
    
    # Collect files to upload
    files_to_upload = []
    
    # If specific files are provided
    if args.files:
        for file_pattern in args.files:
            matched_files = glob.glob(file_pattern)
            files_to_upload.extend(matched_files)
    
    # If a directory is provided
    if args.dir:
        dir_path = Path(args.dir)
        if not dir_path.exists() or not dir_path.is_dir():
            print(f"Error: {args.dir} is not a valid directory")
            return
        
        for ext in args.extensions:
            pattern = f"*{ext}"
            matched_files = list(dir_path.glob(pattern))
            files_to_upload.extend([str(f) for f in matched_files])
    
    # If no files found
    if not files_to_upload:
        print("No files found to upload. Please specify files with --files or --dir.")
        return
    
    print(f"Found {len(files_to_upload)} files to upload")
    
    # Upload files
    file_ids = []
    for file_path in files_to_upload:
        file_id = upload_file(file_path)
        if file_id:
            file_ids.append(file_id)
    
    if not file_ids:
        print("No files were successfully uploaded. Aborting vector store creation.")
        return
    
    # Create vector store
    vector_store_id = create_vector_store(file_ids, args.name)
    
    if not vector_store_id:
        print("Failed to create vector store.")
        return
    
    print(f"Vector store ID: {vector_store_id}")
    
    # Update .env file if requested
    if args.update_env:
        update_env_file(vector_store_id)

if __name__ == "__main__":
    main()

