#!/usr/bin/env python3
import os
import sys
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

def inspect_database():
    """Inspect database contents to verify it's properly populated"""
    print(f"Inspecting medical database at {CHROMA_PATH}...")
    
    if not os.path.exists(CHROMA_PATH):
        print(f"Database directory {CHROMA_PATH} does not exist!")
        return False
        
    try:
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        # Get collection stats
        collection_stats = db._collection.count()
        print(f"Database contains {collection_stats} documents")
        
        if collection_stats == 0:
            print("ERROR: Database is empty! Run populate_database.py to add medical documents.")
            return False
        
        # Sample some documents to verify content
        sample_results = db.similarity_search("sample", k=1)
        if sample_results:
            print("\nSample document content:")
            print("------------------------")
            print(f"Content preview: {sample_results[0].page_content[:200]}...")
            print(f"Metadata: {sample_results[0].metadata}")
        else:
            print("WARNING: Could not retrieve sample document")
        
        return True
        
    except Exception as e:
        print(f"Error inspecting database: {str(e)}")
        return False

if __name__ == "__main__":
    success = inspect_database()
    sys.exit(0 if success else 1)