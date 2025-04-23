import argparse
import os
import shutil
import stat
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma  # Updated import
from tqdm import tqdm  # Import tqdm for progress bars


CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    print("📄 Loading documents...")
    documents = load_documents()
    print(f"Found {len(documents)} documents")
    
    print("✂️ Splitting documents...")
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    print("💾 Adding to database...")
    add_to_chroma(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    try:
        # Load the existing database.
        db = Chroma(
            persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
        )

        # Calculate Page IDs.
        chunks_with_ids = calculate_chunk_ids(chunks)
        
        # Try to get existing items
        try:
            existing_items = db.get(include=[])
            existing_ids = set(existing_items["ids"])
            print(f"Number of existing documents in DB: {len(existing_ids)}")
            
            # Only add documents that don't exist in the DB.
            new_chunks = []
            for chunk in tqdm(chunks_with_ids, desc="Filtering chunks"):
                if chunk.metadata["id"] not in existing_ids:
                    new_chunks.append(chunk)
        except Exception as e:
            print(f"Error accessing existing database: {e}")
            print("🚨 Database may be corrupted. Recreating database...")
            clear_database()
            db = Chroma(
                persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
            )
            new_chunks = chunks_with_ids
        
        if len(new_chunks):
            print(f"👉 Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            
            # Add documents in batches with progress bar
            batch_size = 50
            for i in tqdm(range(0, len(new_chunks), batch_size), desc="Adding to database"):
                batch = new_chunks[i:min(i+batch_size, len(new_chunks))]
                batch_ids = new_chunk_ids[i:min(i+batch_size, len(new_chunk_ids))]
                db.add_documents(batch, ids=batch_ids)
            
            # Remove this line - no longer needed with langchain_chroma
            # db.persist()
        else:
            print("✅ No new documents to add")
    except Exception as e:
        print(f"❌ Error with Chroma database: {e}")
        print("🔄 Attempting to reset and rebuild the database...")
        clear_database()
        # Create a fresh database with the chunks
        db = Chroma(
            persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
        )
        chunks_with_ids = calculate_chunk_ids(chunks)
        
        # Add all documents as new
        print(f"👉 Adding all documents to fresh database: {len(chunks_with_ids)}")
        chunk_ids = [chunk.metadata["id"] for chunk in chunks_with_ids]
        
        # Add documents in batches with progress bar
        batch_size = 50
        for i in tqdm(range(0, len(chunks_with_ids), batch_size), desc="Adding to database"):
            batch = chunks_with_ids[i:min(i+batch_size, len(chunks_with_ids))]
            batch_ids = chunk_ids[i:min(i+batch_size, len(chunk_ids))]
            db.add_documents(batch, ids=batch_ids)
        
        db.persist()


def calculate_chunk_ids(chunks):
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    print("🔢 Calculating chunk IDs...")
    last_page_id = None
    current_chunk_index = 0

    # Add progress bar for chunk ID calculation
    for chunk in tqdm(chunks, desc="Processing chunks"):
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        print(f"Removing existing database at {CHROMA_PATH}")
        try:
            # First, try to fix permissions recursively
            for root, dirs, files in os.walk(CHROMA_PATH):
                for dir_name in dirs:
                    try:
                        os.chmod(os.path.join(root, dir_name), 0o755)
                    except:
                        pass
                for file_name in files:
                    try:
                        os.chmod(os.path.join(root, file_name), 0o644)
                    except:
                        pass
            
            # Then try to remove the directory
            shutil.rmtree(CHROMA_PATH)
        except Exception as e:
            print(f"Failed to remove directory: {e}")
            print("Please manually run the following command:")
            print(f"sudo rm -rf {os.path.abspath(CHROMA_PATH)}")
            exit(1)
    
    # Create the directory with proper permissions
    os.makedirs(CHROMA_PATH, exist_ok=True)
    os.chmod(CHROMA_PATH, 0o755)  # rwx r-x r-x


if __name__ == "__main__":
    main()
