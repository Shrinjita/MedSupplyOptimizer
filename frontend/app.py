import os
import json
import logging
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import shutil
from datetime import datetime

# Import functions from your existing modules
from populate_database import load_documents, split_documents, add_to_chroma, clear_database
from query_data import query_rag, reset_database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='medical_rag.log'
)
logger = logging.getLogger('medical-rag')

app = Flask(__name__, static_folder='static')
app.secret_key = "ragappsecretkey"
app.config['UPLOAD_FOLDER'] = 'data'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
ALLOWED_EXTENSIONS = {'pdf'}

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Database health check
def check_database_health():
    try:
        # Simple test query to check if database is responsive
        test_result = query_rag("test database health")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return False

# Home route
@app.route('/')
def home():
    # Get list of currently loaded documents
    documents = []
    db_status = check_database_health()
    
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        documents = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.pdf')]
    
    return render_template('index.html', documents=documents, db_status=db_status)

# Route for document upload
@app.route('/upload', methods=['POST'])
def upload_document():
    # Check if a file was uploaded
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('home'))
    
    file = request.files['file']
    
    # If user doesn't select a file
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('home'))
    
    if file and allowed_file(file.filename):
        # Create upload folder if it doesn't exist
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        # Save the file with timestamp to avoid duplicates
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = secure_filename(file.filename)
        filename = f"{timestamp}_{base_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process the newly uploaded document
            documents = load_documents()
            chunks = split_documents(documents)
            add_to_chroma(chunks)
            
            # Save metadata about the document
            save_document_metadata(filename, len(chunks))
            
            flash(f'Medical document {base_filename} uploaded and processed successfully!')
            logger.info(f"Document uploaded and processed: {filename}")
        except Exception as e:
            logger.error(f"Error processing document {filename}: {str(e)}")
            flash(f'Error processing document: {str(e)}. Attempting database repair...')
            
            # Try to recover by resetting and rebuilding the database
            try:
                reset_database()
                documents = load_documents()
                chunks = split_documents(documents)
                add_to_chroma(chunks)
                flash('Database has been repaired and documents reprocessed!')
                logger.info("Database repaired successfully")
            except Exception as recovery_error:
                logger.error(f"Failed to recover database: {str(recovery_error)}")
                flash(f'Database repair failed: {str(recovery_error)}. Please contact support.')
        
        return redirect(url_for('home'))
    
    flash('Invalid file type. Only PDF files are allowed.')
    return redirect(url_for('home'))

def save_document_metadata(filename, chunk_count):
    """Save metadata about processed documents"""
    metadata_file = "document_metadata.json"
    metadata = {}
    
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            try:
                metadata = json.load(f)
            except:
                metadata = {}
    
    metadata[filename] = {
        'chunks': chunk_count,
        'processed_date': datetime.now().isoformat(),
        'file_path': os.path.join(app.config['UPLOAD_FOLDER'], filename)
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

# Route for resetting the database
@app.route('/reset', methods=['POST'])
def reset_db():
    try:
        clear_database()
        flash('Medical database has been reset successfully!')
        logger.info("Database reset by user")
        
        # Optional: rebuild database with existing documents
        rebuild = request.form.get('rebuild', 'false') == 'true'
        if rebuild:
            documents = load_documents()
            if documents:
                chunks = split_documents(documents)
                add_to_chroma(chunks)
                flash('Database has been rebuilt with existing documents!')
                logger.info("Database rebuilt with existing documents")
    except Exception as e:
        logger.error(f"Error during database reset: {str(e)}")
        flash(f'Error resetting database: {str(e)}')
    return redirect(url_for('home'))

# Route for structured medical queries
@app.route('/api/medical-query', methods=['POST'])
def medical_query():
    data = request.json
    query_type = data.get('queryType', '')
    symptoms = data.get('symptoms', '')
    patient_details = data.get('patientDetails', '')
    
    if not symptoms and not patient_details:
        return jsonify({'error': 'Please provide patient symptoms or details'})
    
    try:
        # Format the query based on query type with improved medical prompts
        if query_type == 'disease':
            # Add medical terminology to increase matching
            formatted_query = f"""
Medical assessment for patient with following symptoms:
{symptoms}

Patient details: {patient_details}

Relevant medical conditions, diagnoses, or differential diagnoses to consider.
Keywords: diagnosis, symptoms, medical condition, disease, assessment, examination
"""
        elif query_type == 'recovery':
            formatted_query = f"""
Recovery and treatment plan for patient with:
{symptoms}

Patient details: {patient_details}

Keywords: treatment, recovery, prognosis, therapy, rehabilitation, management
"""
        elif query_type == 'resources':
            formatted_query = f"""
Medical resources and equipment needed for patient with:
{symptoms}

Patient details: {patient_details}

Keywords: equipment, resources, supplies, materials, procedure, management
"""
        else:
            formatted_query = f"""
Medical information regarding:
{symptoms}
{patient_details}

Keywords: medical, clinical, healthcare, patient care, treatment, diagnosis
"""

        # Added logging of the actual query being sent
        app.logger.info(f"Sending medical query: {formatted_query[:100]}...")
        
        # Process the query
        response_text = query_rag(formatted_query)
        
        # Log successful query
        logger.info(f"Medical query processed successfully. Type: {query_type}")
        
        return jsonify({
            'result': response_text,
            'query_type': query_type,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error processing medical query: {str(e)}")
        
        # Check if database is healthy
        if not check_database_health():
            return jsonify({
                'error': "Database appears to be corrupted. Please reset and rebuild the database.",
                'db_status': 'unhealthy',
                'resolution': 'Use the reset function on the home page.'
            })
            
        return jsonify({'error': f"Error processing medical query: {str(e)}"})

# General query route with improved medical context
@app.route('/api/query', methods=['POST'])
def api_query():
    data = request.json
    query_text = data.get('query', '')
    
    if not query_text:
        return jsonify({'error': 'No query provided'})
    
    try:
        # Add a prefix to guide the response format with improved clinical focus
        enhanced_query = f"""
As a medical AI assistant using the medical literature database, please analyze this clinical query:

{query_text}

Your response should follow this clinical assessment format:
1. Clinical Interpretation: Analyze the key medical aspects of the query
2. Differential Diagnosis: List and explain potential conditions in order of likelihood
3. Management Considerations: Outline evidence-based treatment approaches
4. Prognosis: Discuss expected outcomes and recovery timelines
5. Clinical Resources: Identify specialists, tests, and resources needed

Base your analysis strictly on medical evidence from the database. If information is insufficient, clearly state what additional data would be needed for a complete assessment.
"""
        result = query_rag(enhanced_query)
        logger.info("General medical query processed successfully")
        return jsonify({'result': result})
    except Exception as e:
        logger.error(f"Error processing general query: {str(e)}")
        return jsonify({'error': str(e)})

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    db_status = check_database_health()
    
    # Check if documents folder exists and has files
    docs_exist = False
    doc_count = 0
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        pdf_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.pdf')]
        docs_exist = len(pdf_files) > 0
        doc_count = len(pdf_files)
        
    return jsonify({
        'status': 'healthy' if db_status else 'unhealthy',
        'database': 'connected' if db_status else 'disconnected',
        'documents': {
            'available': docs_exist,
            'count': doc_count
        },
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Ensure application directories exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
        
    # Log application start
    logger.info("Medical RAG application starting")
    
    # Check database health at startup
    db_status = check_database_health()
    logger.info(f"Initial database health check: {'Passed' if db_status else 'Failed'}")
    
    app.run(debug=True)