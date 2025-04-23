import argparse
import os
import shutil
import logging
import time
import json
import sqlite3
from pathlib import Path
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function

# Configure logging with more detailed information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_query.log'),
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger('rag-query')

CHROMA_PATH = "chroma"
BACKUP_DIR = "chroma_backups"
DB_VERSION = "1.0.0"  # Used to track database schema version

# Enhanced medical prompt template optimized for clinical accuracy
PROMPT_TEMPLATE = """
## MEDICAL CONTEXT INFORMATION
The following information is extracted from reliable medical resources:

{context}

## QUERY
{question}

## RESPONSE INSTRUCTIONS
Provide a clinically accurate response based exclusively on the medical context above.
If the information is insufficient to answer confidently, state this clearly.
Structure your response to be clinically useful, evidence-based, and aligned with medical best practices.
Format your response in the following sections:
1. Clinical Assessment
2. Diagnosis Considerations 
3. Management Recommendations
4. Follow-up/Monitoring
5. Patient Education Points

Cite specific sections from the context when possible.
"""


def main():
    # Create CLI with enhanced options
    parser = argparse.ArgumentParser(description="Medical Information Retrieval System")
    parser.add_argument("query_text", nargs="?", type=str, help="The medical query text.")
    parser.add_argument("--reset-db", action="store_true", help="Reset the Chroma database before querying.")
    parser.add_argument("--verify-db", action="store_true", help="Verify database integrity.")
    parser.add_argument("--force-repair", action="store_true", help="Force database repair if corrupted.")
    parser.add_argument("--clinical-mode", action="store_true", help="Use enhanced clinical response mode.")
    parser.add_argument("--backup", action="store_true", help="Create a database backup before operations.")
    args = parser.parse_args()
    
    # Create backup directory if it doesn't exist
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
    
    # Backup database if requested
    if args.backup and os.path.exists(CHROMA_PATH):
        backup_database()
    
    if args.reset_db:
        reset_database()
        print("Medical database has been reset. Please run populate_database.py to repopulate it.")
        return
        
    if args.verify_db:
        if verify_database():
            print("✅ Database verification passed!")
        else:
            print("❌ Database verification failed.")
            if args.force_repair:
                print("🔄 Attempting automatic database repair...")
                if repair_database():
                    print("✅ Database repair successful!")
                else:
                    print("❌ Database repair failed. Please reset database manually with --reset-db")
            else:
                print("Consider using --reset-db to reset the database or --force-repair to attempt repair.")
        return
    
    if not args.query_text:
        print("Please provide a medical query. Use --help for more information.")
        return
        
    try:
        query_text = args.query_text
        if args.clinical_mode:
            query_text = f"[CLINICAL MODE] {query_text}"
        
        # Auto-verify database before querying
        if not verify_database(silent=True):
            print("⚠️ Database integrity issues detected.")
            if args.force_repair or check_auto_repair_enabled():
                print("🔄 Attempting automatic database repair...")
                if not repair_database():
                    print("❌ Database repair failed, resetting database...")
                    reset_database()
                    print("Please run populate_database.py to repopulate the database.")
                    return
                print("✅ Database repair successful!")
            else:
                print("❌ Database verification failed. Run with --force-repair or --reset-db")
                return
        
        response = query_rag(query_text, max_retries=3, clinical_mode=args.clinical_mode)
        print("\n--- MEDICAL INFORMATION RESPONSE ---\n")
        print(response)
        return response
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        print(f"❌ Error querying medical database: {str(e)}")
        print("Database might be corrupted. Try resetting with --reset-db flag and repopulating it.")


def backup_database():
    """Create a backup of the database"""
    backup_name = f"{BACKUP_DIR}/chroma_backup_{int(time.time())}"
    try:
        logger.info(f"Creating database backup at {backup_name}")
        print(f"Creating database backup at {backup_name}")
        shutil.copytree(CHROMA_PATH, backup_name)
        print("✅ Database backup created successfully.")
        
        # Keep only last 5 backups to save space
        backups = sorted(Path(BACKUP_DIR).glob("chroma_backup_*"))
        if len(backups) > 5:
            for old_backup in backups[:-5]:
                shutil.rmtree(old_backup)
                logger.info(f"Removed old backup: {old_backup}")
                
        return True
    except Exception as e:
        logger.error(f"Backup creation failed: {str(e)}")
        print(f"❌ Backup creation failed: {str(e)}")
        return False


def repair_database():
    """Attempt to repair a corrupted database"""
    try:
        logger.info("Attempting database repair")
        
        # First, attempt to fix database files permissions
        for root, dirs, files in os.walk(CHROMA_PATH):
            for file in files:
                os.chmod(os.path.join(root, file), 0o666)  # Read/write for all
        
        # Second, try to recover corrupted SQLite files
        sqlite_files = list(Path(CHROMA_PATH).glob("**/*.sqlite3"))
        for db_file in sqlite_files:
            try:
                # Attempt to open and recover the database
                conn = sqlite3.connect(str(db_file))
                cursor = conn.cursor()
                cursor.execute("PRAGMA integrity_check")
                integrity_result = cursor.fetchone()[0]
                if integrity_result != "ok":
                    logger.warning(f"SQLite integrity issue in {db_file}: {integrity_result}")
                    cursor.execute("VACUUM")  # Compact database
                    cursor.execute("REINDEX")  # Rebuild indexes
                conn.commit()
                conn.close()
                logger.info(f"Repaired SQLite database: {db_file}")
            except Exception as e:
                logger.error(f"Failed to repair SQLite file {db_file}: {str(e)}")
        
        # Third, verify if database works now
        if verify_database(silent=True):
            logger.info("Database repair successful")
            return True
        
        # If still not working, try more aggressive repair approach
        logger.warning("Basic repair failed, attempting more aggressive approach")
        try:
            # Save collection information before reset
            embedding_function = get_embedding_function()
            try:
                db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
                collection_info = {"name": db._collection_name}
                with open(f"{BACKUP_DIR}/collection_info.json", "w") as f:
                    json.dump(collection_info, f)
                logger.info("Saved collection information")
            except:
                logger.warning("Could not retrieve collection information")
            
            # Reset database but preserve schema files if possible
            schema_files = list(Path(CHROMA_PATH).glob("**/schema.json"))
            schema_contents = {}
            for schema_file in schema_files:
                try:
                    with open(schema_file, "r") as f:
                        schema_contents[str(schema_file)] = f.read()
                except:
                    pass
            
            # Reset database
            reset_database(quiet=True)
            
            # Recreate base directory
            os.makedirs(CHROMA_PATH, exist_ok=True)
            
            # Restore schema files if available
            for path, content in schema_contents.items():
                try:
                    dir_path = os.path.dirname(path)
                    os.makedirs(dir_path, exist_ok=True)
                    with open(path, "w") as f:
                        f.write(content)
                except:
                    pass
            
            logger.info("Aggressive repair completed")
            
            # Verify again
            if verify_database(silent=True):
                logger.info("Aggressive repair successful")
                return True
            else:
                logger.error("Aggressive repair failed")
                return False
            
        except Exception as e:
            logger.error(f"Aggressive repair failed: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Database repair failed: {str(e)}")
        return False


def check_auto_repair_enabled():
    """Check if automatic repair is enabled in config"""
    try:
        config_file = "rag_config.json"
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
                return config.get("auto_repair", False)
        return False
    except:
        return False


def reset_database(quiet=False):
    """Remove the existing Chroma database directory with enhanced logging and safety."""
    if os.path.exists(CHROMA_PATH):
        if not quiet:
            logger.info(f"Removing existing database at {CHROMA_PATH}")
            print(f"Removing existing medical database at {CHROMA_PATH}")
        
        # Create backup before removal if not in quiet mode
        if not quiet:
            backup_database()
            
        try:
            shutil.rmtree(CHROMA_PATH)
            if not quiet:
                logger.info("Database successfully removed")
                print("Medical database successfully removed.")
        except Exception as e:
            logger.error(f"Error removing database: {str(e)}")
            if not quiet:
                print(f"Error removing database: {str(e)}")
            
            # Try alternative removal method
            try:
                if not quiet:
                    print("Attempting alternative removal method...")
                for root, dirs, files in os.walk(CHROMA_PATH, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir(CHROMA_PATH)
                if not quiet:
                    print("Alternative removal successful.")
            except Exception as alt_e:
                logger.error(f"Alternative removal failed: {str(alt_e)}")
                if not quiet:
                    print(f"Alternative removal failed: {str(alt_e)}")
    else:
        if not quiet:
            logger.info(f"No existing database found at {CHROMA_PATH}")
            print(f"No existing medical database found at {CHROMA_PATH}")

    # Create version tracking file
    try:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        with open(os.path.join(CHROMA_PATH, "db_version.txt"), "w") as f:
            f.write(DB_VERSION)
    except:
        pass


def verify_database(silent=False):
    """Check if the database is functioning correctly with enhanced diagnostics."""
    try:
        if not silent:
            logger.info("Verifying database integrity")
        
        # Check if directory exists
        if not os.path.exists(CHROMA_PATH):
            if not silent:
                logger.warning("Database directory does not exist")
                print("❌ Database directory does not exist.")
            return False
        
        # Check for required files and directories
        required_files = ["chroma.sqlite3"]
        for file in required_files:
            file_path = Path(CHROMA_PATH) / file
            if not any(Path(CHROMA_PATH).glob(f"**/{file}")):
                if not silent:
                    logger.warning(f"Required file not found: {file}")
                    print(f"❌ Required file not found: {file}")
                return False
        
        # Try to initialize and query the database
        embedding_function = get_embedding_function()
        try:
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
            
            # Try a simple search
            results = db.similarity_search_with_score("test query for verification", k=1)
            if not silent:
                print(f"✅ Database connection successful. Found {len(results)} results.")
                
            return True
        except Exception as e:
            if not silent:
                logger.error(f"Database verification failed during query: {str(e)}")
                print(f"❌ Database verification failed: {str(e)}")
            return False
    except Exception as e:
        if not silent:
            logger.error(f"Database verification failed: {str(e)}")
            print(f"❌ Database verification failed: {str(e)}")
        return False


def query_rag(query_text: str, max_retries=1, clinical_mode=False):
    """
    Query the RAG system with enhanced error handling, retries, and clinical mode.
    
    Args:
        query_text: The medical query to process
        max_retries: Maximum number of retry attempts
        clinical_mode: Enable enhanced clinical response formatting
    """
    retries = 0
    last_error = None
    
    while retries <= max_retries:
        try:
            # Prepare the DB with error checks
            embedding_function = get_embedding_function()
            
            try:
                db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
            except Exception as db_error:
                logger.error(f"Database connection error: {str(db_error)}")
                if "KeyError: '_type'" in str(db_error):
                    # This is the specific error we've been encountering
                    logger.warning("Detected '_type' error, attempting database repair")
                    if repair_database():
                        logger.info("Database repaired, retrying connection")
                        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
                    else:
                        raise Exception("Database repair failed for '_type' error. Please reset database.")
                else:
                    raise
            
            # Enhanced query processing for medical context
            if clinical_mode:
                k_value = 20  # More context for clinical queries - increased from 10
                initial_threshold = 0.45  # Much more lenient threshold
                logger.info(f"Using enhanced clinical mode parameters (threshold: {initial_threshold})")
            else:
                k_value = 15  # Increased from 7
                initial_threshold = 0.55  # More lenient for general queries
            
            # Search the DB with clinically relevant parameters
            results = db.similarity_search_with_score(
                query_text, 
                k=k_value
            )

            # Log raw results for debugging
            if results:
                logger.info(f"Query returned {len(results)} raw results")
                logger.info(f"Score range: {min(score for _, score in results):.4f} to {max(score for _, score in results):.4f}")
            else:
                logger.info("Query returned no raw results")

            # Implement adaptive threshold with fallback
            threshold = initial_threshold
            filtered_results = [(result, score) for result, score in results if score <= threshold]

            # If no results, gradually increase threshold until we find something
            while not filtered_results and threshold < 0.95 and results:
                threshold += 0.1
                logger.info(f"No results at threshold {threshold-0.1:.2f}, increasing to {threshold:.2f}")
                filtered_results = [(result, score) for result, score in results if score <= threshold]

            if threshold > initial_threshold and filtered_results:
                logger.info(f"Found {len(filtered_results)} results using relaxed threshold of {threshold:.2f}")

            results = filtered_results

            if not results:
                logger.warning("No relevant medical information found")
                return "No relevant medical information found in our database. Please refine your query or ensure the medical database is properly populated with appropriate resources."

            # Process results with clinical importance weighting
            weighted_results = []
            for doc, score in results:
                # Check if document has clinical_importance in metadata
                clinical_importance = doc.metadata.get('clinical_importance', 1.0)
                
                # Apply semantic relevance boost based on query and content similarity
                semantic_boost = 1.0
                if "urgent" in query_text.lower() or "emergency" in query_text.lower():
                    if "urgent" in doc.page_content.lower() or "emergency" in doc.page_content.lower():
                        semantic_boost = 1.5
                
                weighted_score = score * (1 + clinical_importance) * semantic_boost
                weighted_results.append((doc, weighted_score))
            
            # Sort by weighted score (lower is better in similarity search)
            weighted_results.sort(key=lambda x: x[1])
            
            # Use top results for context
            context_docs = weighted_results[:5]
            context_text = "\n\n## Source Document " + "\n\n## Source Document ".join(
                [f"({i+1}) - {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" 
                 for i, (doc, _) in enumerate(context_docs)]
            )

            # Create prompt with medical context
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            
            # Enhance prompt for clinical mode
            if clinical_mode:
                enhanced_query = f"""[CLINICAL REQUEST]
Please provide a detailed clinical assessment for the following query.
Emphasize evidence-based approaches and clinical reasoning.

{query_text}

Structure your response with clear clinical recommendations and guidance.
"""
                prompt = prompt_template.format(context=context_text, question=enhanced_query)
            else:
                prompt = prompt_template.format(context=context_text, question=query_text)

            # Query LLM with configurable parameters
            logger.info("Sending query to LLM")
            
            # Set parameters based on mode
            if clinical_mode:
                temperature = 0.05  # Very low temperature for clinical precision
                threads = 12  # More processing power for clinical responses
            else:
                temperature = 0.1
                threads = 8
                
            model = Ollama(
                model="llama3.2",
                num_thread=threads,
                temperature=temperature
            )
            
            # Execute query with timeout protection
            try:
                response_text = model.invoke(prompt)
            except Exception as model_error:
                logger.error(f"Model error: {str(model_error)}")
                if "timeout" in str(model_error).lower():
                    raise Exception("Model response timed out. Try a more specific query.")
                raise
                
            # Log success and returned sources
            sources = [doc.metadata.get("source", "Unknown") for doc, _ in context_docs]
            logger.info(f"Query successful. Sources used: {sources}")
            
            # Format clinical response
            if clinical_mode:
                formatted_response = f"""# Clinical Assessment Report

{response_text}

## Medical References
""" + "\n".join(
                    [f"* **Source {i+1}**: {doc.metadata.get('source', 'Unknown')}" 
                     for i, (doc, _) in enumerate(context_docs)]
                )
            else:
                # Add sources to standard response for traceability
                formatted_response = f"{response_text}\n\n**References**:\n" + "\n".join(
                    [f"[{i+1}] {doc.metadata.get('source', 'Unknown')}" 
                     for i, (doc, _) in enumerate(context_docs)]
                )
            
            return formatted_response
            
        except Exception as e:
            retries += 1
            last_error = str(e)
            logger.warning(f"Query attempt {retries} failed: {last_error}")
            
            if retries <= max_retries:
                logger.info(f"Retrying... ({retries}/{max_retries})")
                time.sleep(1)  # Wait before retry
                
                # If database error, try to repair before next attempt
                if "database" in last_error.lower() or "chroma" in last_error.lower():
                    logger.info("Attempting database repair before retry")
                    repair_database()
            else:
                logger.error(f"All retry attempts failed")
                raise Exception(f"Failed to query medical database after {max_retries} attempts. Last error: {last_error}")


if __name__ == "__main__":
    main()
