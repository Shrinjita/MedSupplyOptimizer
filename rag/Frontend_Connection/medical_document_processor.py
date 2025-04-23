import logging
import re
import os
import json
import hashlib
from typing import List, Dict, Any, Set, Tuple, Optional
from datetime import datetime
from pathlib import Path

# Enhanced document loading capabilities
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Configure logging with more detailed information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('medical_processor.log'),
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger('medical-processor')

# Cache directory for processed documents
CACHE_DIR = "document_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Expanded medical terminology patterns for enhanced metadata extraction
DISEASE_PATTERN = r"(cancer|carcinoma|tumor|diabetes|hypertension|asthma|arthritis|alzheimer|dementia|stroke|" \
                 r"copd|hepatitis|cirrhosis|emphysema|obesity|depression|anxiety|heart\s+(?:disease|failure)|" \
                 r"myocardial\s+infarction|pneumonia|atherosclerosis|ischemia|sepsis|fibrillation|" \
                 r"parkinson|epilepsy|multiple\s+sclerosis|huntington|als|meningitis|encephalitis|influenza|" \
                 r"hypothyroidism|hyperthyroidism|osteoporosis|leukemia|lymphoma|hiv|aids|tuberculosis|" \
                 r"glaucoma|cataracts|macular\s+degeneration)"

MEDICATION_PATTERN = r"(aspirin|ibuprofen|acetaminophen|paracetamol|metformin|insulin|lisinopril|" \
                    r"atorvastatin|simvastatin|omeprazole|albuterol|salbutamol|amlodipine|metoprolol|" \
                    r"prednisone|gabapentin|hydrochlorothiazide|levothyroxine|fluoxetine|sertraline|" \
                    r"citalopram|amoxicillin|azithromycin|ciprofloxacin|losartan|hydrocodone|oxycodone|" \
                    r"warfarin|clopidogrel|pantoprazole|escitalopram|bupropion|alprazolam|tramadol|" \
                    r"furosemide|montelukast)"

PROCEDURE_PATTERN = r"(surgery|biopsy|endoscopy|colonoscopy|mri|ct\s+scan|ultrasound|x-ray|ekg|ecg|" \
                   r"mammography|dialysis|radiotherapy|chemotherapy|catheterization|angioplasty|" \
                   r"cardioversion|intubation|lumbar\s+puncture|arthroscopy|bronchoscopy|thoracentesis|" \
                   r"paracentesis|amniocentesis|colposcopy|laparoscopy|cytoscopy|immunotherapy|" \
                   r"cardioversion|ablation|transplantation|amputation|appendectomy|cholecystectomy|" \
                   r"colostomy|hysterectomy|lobectomy|tracheostomy|coronary\s+bypass)"

# Lab test patterns for additional clinical context
LAB_TEST_PATTERN = r"(cbc|complete\s+blood\s+count|bmp|basic\s+metabolic\s+panel|cmp|" \
                 r"comprehensive\s+metabolic\s+panel|a1c|hemoglobin\s+a1c|tsh|thyroid|" \
                 r"lipid\s+panel|cholesterol|ldl|hdl|triglycerides|creatinine|bun|gfr|" \
                 r"alt|ast|alkaline\s+phosphatase|bilirubin|troponin|d-dimer|prothrombin\s+time|" \
                 r"inr|ptt|esr|crp|c-reactive\s+protein|ferritin|vitamin\s+d|b12|folate|" \
                 r"electrolytes|sodium|potassium|calcium|magnesium|phosphate|albumin|" \
                 r"urinalysis|urine\s+culture|blood\s+culture)"


class MedicalDocumentProcessor:
    """Enhanced processor for medical documents with comprehensive terminology extraction, 
    metadata enrichment, and advanced content analysis"""
    
    def __init__(self, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200,
                 use_cache: bool = True,
                 preserve_clinical_sections: bool = True):
        """
        Initialize the medical document processor with configurable parameters.
        
        Args:
            chunk_size: Size of document chunks for RAG processing
            chunk_overlap: Overlap between chunks to preserve context
            use_cache: Whether to cache processed documents
            preserve_clinical_sections: Keep clinical sections intact during chunking
        """
        self.use_cache = use_cache
        self.preserve_clinical_sections = preserve_clinical_sections
        
        # Configure separators to respect clinical document structure
        separators = [
            # Clinical document section headers (preserve these boundaries)
            "\n## ", "\n### ", "\nASSESSMENT:", "\nDIAGNOSIS:", "\nTREATMENT PLAN:", 
            "\nIMPRESSION:", "\nPLAN:", "\nHISTORY:", "\nPHYSICAL EXAM:",
            # Standard text separators (for general chunking)
            "\n\n", "\n", ". ", " ", ""
        ]
        
        # Medical text splitter with special handling for clinical sections
        if self.preserve_clinical_sections:
            # Use specialized separator pattern that preserves clinical sections
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=separators
            )
        else:
            # Standard separator pattern
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        
        # Create cache directory if it doesn't exist
        if self.use_cache and not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
    
    def _generate_cache_filename(self, file_path: str) -> str:
        """Generate a cache filename based on file path and modification time."""
        file_stats = os.stat(file_path)
        mod_time = file_stats.st_mtime
        file_hash = hashlib.md5(f"{file_path}:{mod_time}".encode()).hexdigest()
        return os.path.join(CACHE_DIR, f"{os.path.basename(file_path)}_{file_hash}.json")
    
    def _save_to_cache(self, file_path: str, chunks: List[Dict]) -> None:
        """Save processed chunks to cache."""
        if not self.use_cache:
            return
            
        cache_file = self._generate_cache_filename(file_path)
        try:
            # Convert Document objects to serializable format
            serializable_chunks = []
            for chunk in chunks:
                serializable_chunk = {
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata
                }
                serializable_chunks.append(serializable_chunk)
                
            with open(cache_file, 'w') as f:
                json.dump(serializable_chunks, f)
            logger.info(f"Saved processed document to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save to cache: {str(e)}")
    
    def _load_from_cache(self, file_path: str) -> Optional[List[Document]]:
        """Load processed chunks from cache if available and valid."""
        if not self.use_cache:
            return None
            
        cache_file = self._generate_cache_filename(file_path)
        if not os.path.exists(cache_file):
            return None
            
        try:
            with open(cache_file, 'r') as f:
                serialized_chunks = json.load(f)
                
            # Convert serialized chunks back to Document objects
            chunks = []
            for chunk_data in serialized_chunks:
                chunk = Document(
                    page_content=chunk_data["page_content"],
                    metadata=chunk_data["metadata"]
                )
                chunks.append(chunk)
                
            logger.info(f"Loaded {len(chunks)} chunks from cache: {cache_file}")
            return chunks
        except Exception as e:
            logger.warning(f"Failed to load from cache: {str(e)}. Will reprocess.")
            return None
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load and process a medical document with enhanced metadata"""
        try:
            logger.info(f"Loading medical document: {file_path}")
            
            # Check file extension to determine appropriate loader
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.pdf':
                loader = PyPDFLoader(file_path)
                pages = loader.load()
            else:
                # Fall back to unstructured loader for other file types
                loader = UnstructuredFileLoader(file_path)
                pages = loader.load()
            
            # Extract document type and clinical field
            doc_type = self.extract_document_type(pages)
            clinical_field = self.extract_clinical_field(pages)
            
            # Calculate document-level quality metrics
            evidence_level = self.assess_evidence_level(pages)
            recency_score = self.assess_recency(pages)
            reference_quality = self.assess_reference_quality(pages)
            
            processed_pages = []
            for i, page in enumerate(pages):
                # Extract medical terminology
                diseases = self.extract_patterns(page.page_content, DISEASE_PATTERN)
                medications = self.extract_patterns(page.page_content, MEDICATION_PATTERN)
                procedures = self.extract_patterns(page.page_content, PROCEDURE_PATTERN)
                lab_tests = self.extract_patterns(page.page_content, LAB_TEST_PATTERN)
                
                # Extract medical conditions with enhanced context
                medical_conditions = self.extract_medical_conditions(page.page_content)
                
                # Determine clinical importance based on terminology density
                clinical_importance = self.calculate_clinical_importance(
                    diseases, medications, procedures, lab_tests, page.page_content
                )
                
                # Extract clinical sections
                sections = self.extract_clinical_sections(page.page_content)
                
                # Enhance metadata
                page.metadata.update({
                    "page_number": i + 1,
                    "source": file_path,
                    "document_type": doc_type,
                    "clinical_field": clinical_field,
                    "diseases_mentioned": list(diseases),
                    "medications_mentioned": list(medications),
                    "procedures_mentioned": list(procedures),
                    "lab_tests": list(lab_tests),
                    "medical_conditions": medical_conditions,
                    "clinical_importance": clinical_importance,
                    "clinical_sections": sections,
                    "evidence_level": evidence_level,
                    "recency_score": recency_score,
                    "reference_quality": reference_quality,
                    "processing_date": datetime.now().isoformat()
                })
                
                processed_pages.append(page)
                
            logger.info(f"Successfully processed document with {len(processed_pages)} pages")
            return processed_pages
            
        except Exception as e:
            logger.error(f"Error loading medical document {file_path}: {str(e)}")
            raise
    
    def split_document(self, pages: List[Document]) -> List[Document]:
        """Split medical document into chunks while preserving clinical context"""
        try:
            logger.info(f"Splitting {len(pages)} pages into chunks")
            all_chunks = []
            
            for page in pages:
                # Special handling for clinical sections if enabled
                if self.preserve_clinical_sections and "clinical_sections" in page.metadata:
                    chunks = self._split_preserving_sections(page)
                else:
                    # Create chunks while preserving metadata
                    chunks = self.text_splitter.create_documents(
                        texts=[page.page_content],
                        metadatas=[page.metadata]
                    )
                
                # Enhance chunk metadata with position information
                for i, chunk in enumerate(chunks):
                    chunk.metadata["chunk_index"] = i
                    chunk.metadata["total_chunks"] = len(chunks)
                    chunk.metadata["chunk_id"] = f"{page.metadata.get('source', 'doc')}_{page.metadata.get('page_number', 0)}_{i}"
                    
                    # Analyze chunk content for specific important clinical information
                    chunk.metadata["contains_dosage"] = bool(re.search(r"\d+\s*mg|\d+\s*mcg|\d+\s*ml", chunk.page_content))
                    chunk.metadata["contains_warning"] = bool(re.search(r"warning|caution|alert|contraindication", chunk.page_content, re.I))
                    
                all_chunks.extend(chunks)
                
            logger.info(f"Created {len(all_chunks)} chunks from document")
            return all_chunks
            
        except Exception as e:
            logger.error(f"Error splitting document: {str(e)}")
            raise
    
    def _split_preserving_sections(self, page: Document) -> List[Document]:
        """Special splitting method that keeps clinical sections intact"""
        chunks = []
        sections = page.metadata.get("clinical_sections", {})
        
        if not sections:
            # Fall back to standard splitting if no sections found
            return self.text_splitter.create_documents(
                texts=[page.page_content],
                metadatas=[page.metadata]
            )
        
        # Process each section as a potential chunk
        for section_name, section_text in sections.items():
            # Skip empty sections
            if not section_text.strip():
                continue
                
            # If section is too large, split it further
            if len(section_text) > self.text_splitter._chunk_size:
                section_chunks = self.text_splitter.create_documents(
                    texts=[section_text],
                    metadatas=[{**page.metadata, "section": section_name}]
                )
                chunks.extend(section_chunks)
            else:
                # Keep the section as a single chunk
                chunk = Document(
                    page_content=section_text,
                    metadata={**page.metadata, "section": section_name}
                )
                chunks.append(chunk)
        
        return chunks
    
    def extract_patterns(self, text: str, pattern: str) -> set:
        """Extract medical terminology based on pattern"""
        matches = re.findall(pattern, text.lower())
        return set(matches)
    
    # Enhanced document metadata extraction with additional clinical context
    def extract_medical_conditions(self, text: str) -> List[Dict[str, Any]]:
        """Extract medical conditions with enhanced context and severity assessment"""
        conditions = []
        
        # Find potential medical conditions with surrounding context
        disease_matches = list(re.finditer(DISEASE_PATTERN, text.lower()))
        
        for match in disease_matches:
            disease = match.group(0)
            
            # Get surrounding context (50 chars before and after)
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end]
            
            # Look for severity indicators
            severity = "unknown"
            if re.search(r"severe|serious|critical|acute", context, re.IGNORECASE):
                severity = "severe"
            elif re.search(r"moderate|significant", context, re.IGNORECASE):
                severity = "moderate"
            elif re.search(r"mild|minimal|slight", context, re.IGNORECASE):
                severity = "mild"
            
            # Look for temporality (current vs. history)
            temporality = "current"
            if re.search(r"history|previous|past|prior|resolved", context, re.IGNORECASE):
                temporality = "history"
            
            conditions.append({
                "condition": disease,
                "context": context,
                "severity": severity,
                "temporality": temporality
            })
        
        return conditions
    
    def extract_document_type(self, pages: List) -> str:
        """Determine the type of medical document with enhanced detection"""
        full_text = " ".join([p.page_content.lower() for p in pages])
        
        # Define document type patterns with weighted terms
        type_patterns = {
            "research_study": {
                "terms": ["clinical trial", "randomized", "cohort", "placebo", "double-blind", 
                         "prospective", "retrospective", "crossover", "phase i", "phase ii", 
                         "phase iii", "control group", "experimental group", "p-value", 
                         "statistically significant", "inclusion criteria", "exclusion criteria"],
                "threshold": 2  # Minimum matches needed
            },
            "clinical_guideline": {
                "terms": ["guideline", "recommendation", "consensus", "practice", "standard of care",
                         "clinical practice", "best practice", "first line", "second line", 
                         "treatment algorithm", "management protocol", "evidence grade", "level of evidence"],
                "threshold": 2
            },
            "case_report": {
                "terms": ["case report", "patient presented", "medical history", "case study", 
                         "case series", "year-old patient", "chief complaint", "presenting with", 
                         "was admitted", "follow-up visit"],
                "threshold": 2
            },
            "literature_review": {
                "terms": ["review", "literature", "meta-analysis", "systematic review", "narrative review", 
                         "scoping review", "evidence synthesis", "critical appraisal"],
                "threshold": 2
            },
            "educational_material": {
                "terms": ["learning objectives", "continuing medical education", "cme", "medical education", 
                         "teaching points", "key concepts", "review questions"],
                "threshold": 2
            },
            "drug_monograph": {
                "terms": ["dosage", "indication", "contraindication", "adverse effects", "drug interaction",
                         "pharmacokinetics", "pharmacodynamics", "metabolism", "half-life", "excretion"],
                "threshold": 3
            }
        }
        
        # Count matches for each document type
        type_scores = {}
        for doc_type, pattern in type_patterns.items():
            count = sum(1 for term in pattern["terms"] if term in full_text)
            if count >= pattern["threshold"]:
                type_scores[doc_type] = count
        
        # If no specific type is detected with enough confidence
        if not type_scores:
            return "general_medical"
        
        # Return the type with the most matches
        return max(type_scores.items(), key=lambda x: x[1])[0]
    
    def extract_clinical_field(self, pages: List) -> str:
        """Determine the clinical field of the document"""
        full_text = " ".join([p.page_content.lower() for p in pages])
        
        fields = {
            "cardiology": ["heart", "cardiac", "cardiovascular", "ecg", "echocardiogram", "myocardial", 
                          "arrhythmia", "coronary", "hypertension", "angina", "pacemaker", "stent", 
                          "valve", "cath lab", "atrial", "ventricular", "fibrillation", "tachycardia"],
            
            "neurology": ["brain", "neural", "neuron", "seizure", "epilepsy", "cerebral", "stroke", 
                         "cognition", "eeg", "headache", "migraine", "tremor", "neuropathy", 
                         "encephalopathy", "meningitis", "multiple sclerosis", "parkinson"],
            
            "oncology": ["cancer", "tumor", "malignant", "metastasis", "chemotherapy", "radiation", 
                        "oncology", "carcinoma", "sarcoma", "lymphoma", "leukemia", "biopsy", 
                        "remission", "staging", "oncologist", "immunotherapy"],
            
            "gastroenterology": ["stomach", "intestinal", "liver", "hepatic", "pancreas", "colon", 
                                "esophagus", "ibd", "gastric", "bowel", "endoscopy", "colonoscopy", 
                                "gerd", "ulcer", "cirrhosis", "colitis", "crohn", "hepatitis"],
            
            "pulmonology": ["lung", "pulmonary", "respiratory", "copd", "asthma", "pneumonia", 
                           "bronchitis", "thoracic", "ventilation", "oxygen", "spirometry", "bronchoscopy", 
                           "emphysema", "intubation", "chest x-ray", "dyspnea"],
            
            "endocrinology": ["hormone", "thyroid", "diabetes", "insulin", "pituitary", "adrenal", 
                             "pancreatic", "glucose", "metabolism", "endocrine", "t3", "t4", "tsh", 
                             "estrogen", "testosterone", "cortisol", "hyperglycemia", "hypoglycemia"],
            
            "dermatology": ["skin", "dermal", "rash", "psoriasis", "eczema", "melanoma", "acne", 
                           "dermatitis", "lesion", "biopsy", "dermis", "epidermis", "hair", "nail", 
                           "dermatologist", "erythema", "pruritus"],
            
            "orthopedics": ["bone", "joint", "fracture", "arthritis", "osteoporosis", "skeletal", 
                           "tendon", "ligament", "orthopedic", "cartilage", "x-ray", "mri", "cast", 
                           "spine", "vertebra", "disc", "knee", "hip replacement", "physical therapy"],
            
            "obstetrics": ["pregnancy", "obstetric", "fetal", "maternal", "childbirth", "prenatal", 
                          "labor", "delivery", "cesarean", "ultrasound", "trimester", "amniotic", 
                          "placenta", "gestational", "preeclampsia"],
            
            "gynecology": ["uterus", "ovary", "cervical", "menstrual", "gynecologic", "menopause", 
                          "pap smear", "pelvic", "breast", "mammogram", "hysterectomy", "endometriosis", 
                          "contraception", "fertility", "hpv"],
            
            "pediatrics": ["child", "pediatric", "infant", "adolescent", "neonatal", "childhood", 
                          "growth", "vaccination", "developmental", "milestones", "pediatrician", 
                          "congenital", "birth", "puberty"],
            
            "psychiatry": ["mental", "psychiatric", "depression", "anxiety", "schizophrenia", 
                          "bipolar", "therapy", "counseling", "psychological", "mood", "cognitive", 
                          "behavioral", "ssri", "antidepressant", "psychosis", "trauma", "ptsd"],
            
            "infectious_disease": ["infection", "bacterial", "viral", "fungal", "antibiotic", 
                                 "vaccine", "immunity", "fever", "pathogen", "culture", "sensitivity", 
                                 "resistance", "antimicrobial", "sepsis", "quarantine", "epidemic", "pandemic"],
            
            "urology": ["kidney", "urinary", "bladder", "prostate", "renal", "urologic", 
                       "erectile", "urination", "catheter", "dialysis", "cystoscopy", 
                       "incontinence", "psa", "stone", "nephrology"],
            
            "ophthalmology": ["eye", "retina", "optic", "vision", "glaucoma", "cataract", 
                             "cornea", "blindness", "ophthalmologist", "lens", "visual acuity", 
                             "refraction", "pupil", "macula", "conjunctiva", "intraocular pressure"],
            
            "emergency_medicine": ["emergency", "trauma", "acute", "critical", "resuscitation", 
                                  "triage", "life-threatening", "ambulance", "intubation", "cpr", 
                                  "defibrillator", "shock", "overdose", "hemorrhage", "stabilization"]
        }
        
        # Count mentions of terms from each field with weighted scoring
        field_scores = {}
        for field, terms in fields.items():
            # Primary terms (beginning of list) get higher weight
            primary_terms = terms[:min(5, len(terms))]
            secondary_terms = terms[min(5, len(terms)):]
            
            # Calculate score with primary terms weighted more heavily
            primary_count = sum(2 for term in primary_terms if term in full_text)
            secondary_count = sum(1 for term in secondary_terms if term in full_text)
            total_score = primary_count + secondary_count
            
            if total_score > 0:
                field_scores[field] = total_score
        
        # If no specific field is detected, return general medicine
        if not field_scores:
            return "general_medicine"
        
        # Return the field with the highest score
        return max(field_scores.items(), key=lambda x: x[1])[0]
    
    def extract_clinical_sections(self, text: str) -> Dict[str, str]:
        """Extract clinical sections from text based on common medical document structure"""
        sections = {}
        
        # Define patterns for common clinical sections
        section_patterns = [
            (r"(?:HISTORY|HISTORY OF PRESENT ILLNESS|HPI|HISTORY OF PRESENTING ILLNESS):(.*?)(?=\n[A-Z][A-Z\s]+:|\Z)", "history"),
            (r"(?:PAST MEDICAL HISTORY|PMH):(.*?)(?=\n[A-Z][A-Z\s]+:|\Z)", "past_medical_history"),
            (r"(?:MEDICATIONS|CURRENT MEDICATIONS|MEDS):(.*?)(?=\n[A-Z][A-Z\s]+:|\Z)", "medications"),
            (r"(?:ALLERGIES|DRUG ALLERGIES):(.*?)(?=\n[A-Z][A-Z\s]+:|\Z)", "allergies"),
            (r"(?:PHYSICAL EXAMINATION|PHYSICAL EXAM|EXAMINATION|EXAM):(.*?)(?=\n[A-Z][A-Z\s]+:|\Z)", "physical_exam"),
            (r"(?:ASSESSMENT|IMPRESSION):(.*?)(?=\n[A-Z][A-Z\s]+:|\Z)", "assessment"),
            (r"(?:PLAN|TREATMENT PLAN):(.*?)(?=\n[A-Z][A-Z\s]+:|\Z)", "plan"),
            (r"(?:DIAGNOSIS|DIAGNOSES):(.*?)(?=\n[A-Z][A-Z\s]+:|\Z)", "diagnosis"),
            (r"(?:LABORATORY|LAB|LABS|LABORATORY RESULTS):(.*?)(?=\n[A-Z][A-Z\s]+:|\Z)", "laboratory"),
            (r"(?:IMAGING|RADIOLOGY|IMAGING RESULTS):(.*?)(?=\n[A-Z][A-Z\s]+:|\Z)", "imaging"),
            (r"(?:PROCEDURES|OPERATIONS|SURGERIES):(.*?)(?=\n[A-Z][A-Z\s]+:|\Z)", "procedures"),
            (r"(?:FOLLOW-UP|FOLLOWUP):(.*?)(?=\n[A-Z][A-Z\s]+:|\Z)", "followup")
        ]
        
        for pattern, section_name in section_patterns:
            matches = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                sections[section_name] = matches.group(1).strip()
        
        return sections
    
    def assess_evidence_level(self, pages: List[Document]) -> str:
        """Assess the evidence level of the document based on content analysis"""
        all_text = " ".join([page.page_content for page in pages])
        all_text = all_text.lower()
        
        # Check for randomized controlled trial indicators
        if re.search(r"randomized control|rct|double blind|placebo.{0,20}control", all_text):
            return "high"
        
        # Check for other study types
        elif re.search(r"cohort study|case.{0,5}control|observational study|meta.{0,5}analysis", all_text):
            return "medium"
        
        # Check for case reports or opinions
        elif re.search(r"case report|case series|expert opinion|clinical practice", all_text):
            return "low"
        
        return "unknown"
    
    def assess_recency(self, pages: List) -> Dict[str, Any]:
        """Assess how recent the medical information is"""
        full_text = " ".join([p.page_content.lower() for p in pages])
        current_year = datetime.now().year
        
        # Look for publication dates
        year_matches = re.findall(r"(?:published|copyright|©|\bpub\.|\bdate:)\s*(?:in\s*)?(\d{4})", full_text)
        years_mentioned = re.findall(r"\b(20\d{2}|19\d{2})\b", full_text)
        
        result = {
            "publication_year": None,
            "recency_score": 0.5,  # Default medium recency score
            "has_recent_references": False
        }
        
        # If explicit publication date found
        if year_matches:
            publication_year = int(max(year_matches))  # Use the most recent publication date
            result["publication_year"] = publication_year
            years_diff = current_year - publication_year
            
            # Score from 0 to 1, with 1 being current year
            if years_diff <= 0:
                result["recency_score"] = 1.0
            elif years_diff < 3:
                result["recency_score"] = 0.9
            elif years_diff < 5:
                result["recency_score"] = 0.7
            elif years_diff < 10:
                result["recency_score"] = 0.5
            elif years_diff < 15:
                result["recency_score"] = 0.3
            else:
                result["recency_score"] = 0.1
        
        # Check for recent references
        if years_mentioned:
            recent_years = [int(y) for y in years_mentioned if int(y) > (current_year - 5)]
            if recent_years:
                result["has_recent_references"] = True
                result["most_recent_reference"] = max(recent_years)
        
        return result
    
    def assess_reference_quality(self, pages: List) -> Dict[str, Any]:
        """Assess the quality of references in the document"""
        full_text = " ".join([p.page_content.lower() for p in pages])
        
        # Check for high-quality reference sources
        high_quality_journals = [
            "nejm", "new england journal", "lancet", "jama", "bmj", "british medical journal",
            "annals of internal medicine", "journal of clinical oncology", "circulation", 
            "gastroenterology", "hepatology", "blood", "journal of the american college of cardiology",
            "nature medicine", "cell", "science", "plos", "cochrane"
        ]
        
        medical_organizations = [
            "world health organization", "who", "cdc", "centers for disease control", 
            "national institutes of health", "nih", "fda", "food and drug administration",
            "american heart association", "american cancer society", "american diabetes association",
            "european medicines agency", "ema"
        ]
        
        result = {
            "has_references": bool(re.search(r"references|bibliography", full_text)),
            "has_high_quality_journals": any(journal in full_text for journal in high_quality_journals),
            "has_medical_organizations": any(org in full_text for org in medical_organizations),
            "reference_quality_score": 0.5  # Default medium quality
        }
        
        # Calculate quality score based on reference sources
        score = 0.5  # Start with neutral score
        
        if result["has_high_quality_journals"]:
            score += 0.3
            
        if result["has_medical_organizations"]:
            score += 0.2
            
        if bool(re.search(r"meta-analysis|systematic review|cochrane", full_text)):
            score += 0.2
            
        if not result["has_references"]:
            score = max(0.1, score - 0.3)
            
        # Cap score between 0.1 and 1.0
        result["reference_quality_score"] = min(1.0, max(0.1, score))
        
        return result
    
    def calculate_clinical_importance(self, 
                                     diseases: Set[str], 
                                     medications: Set[str], 
                                     procedures: Set[str],
                                     lab_tests: Set[str],
                                     content: str) -> float:
        """Calculate clinical importance based on medical terminology density and specificity"""
        try:
            # Base importance starts at 1.0 (neutral)
            importance = 1.0
            
            # Count tokens to calculate density
            tokens = content.split()
            total_tokens = len(tokens)
            if total_tokens == 0:
                return importance
            
            # Calculate density of medical terms with weighted importance
            # Lab tests and procedures typically more important than general disease mentions
            medical_term_weights = {
                "diseases": 1.0,
                "medications": 1.2,
                "procedures": 1.3,
                "lab_tests": 1.1
            }
            
            weighted_term_count = (
                len(diseases) * medical_term_weights["diseases"] +
                len(medications) * medical_term_weights["medications"] +
                len(procedures) * medical_term_weights["procedures"] +
                len(lab_tests) * medical_term_weights["lab_tests"]
            )
            
            # Calculate weighted density (terms per 100 words)
            term_density = weighted_term_count / (total_tokens / 100)
            
            # Adjust importance based on term density (logarithmic scale for better scaling)
            if term_density > 0:
                importance += 0.3 * min(2.0, term_density / 2)
            
            # Check for critical medical keywords indicating high importance
            critical_keywords = [
                "emergency", "urgent", "critical", "severe", "life-threatening",
                "mortality", "fatal", "death", "guideline", "standard of care",
                "protocol", "recommended", "evidence-based", "clinical trial",
                "first-line therapy", "gold standard", "contraindicated", "warning",
                "adverse event", "side effect", "interaction", "high-risk", "overdose"
            ]
            
            # Check for warning/caution phrases with higher weight
            warning_phrases = [
                "black box warning", "not recommended", "discontinue immediately",
                "seek medical attention", "medical emergency", "call 911",
                "contraindicated", "do not use", "fatal if", "life-threatening reaction"
            ]
            
            # Count critical keywords with regular expressions for better matching
            critical_count = sum(1 for keyword in critical_keywords 
                               if re.search(r'\b' + re.escape(keyword) + r'\b', content.lower()))
            
            warning_count = sum(2 for phrase in warning_phrases 
                              if re.search(r'\b' + re.escape(phrase) + r'\b', content.lower()))
            
            if critical_count > 0:
                importance += min(0.5, 0.1 * critical_count)
                
            if warning_count > 0:
                importance += min(0.6, 0.15 * warning_count)
            
            # Check for specific document structures that indicate high clinical value
            if re.search(r"(recommendation|guideline|protocol|standard\s+of\s+care).*?(class|level|grade)\s+[IA]", 
                        content, re.IGNORECASE):
                importance += 0.3
            
            # Check for references to studies, which can indicate evidence-based content
            study_references = len(re.findall(r"(study|trial|cohort|analysis|p\s*<\s*0\.0[0-9]|confidence interval|odds ratio)",
                                             content, re.IGNORECASE))
            if study_references > 0:
                importance += min(0.3, 0.05 * study_references)
                
            # Look for dosage information which is highly clinical
            dosage_patterns = len(re.findall(r"\d+\s*(?:mg|mcg|g|ml|units)/(?:day|dose|kg|m2|hr|h|hour|week)", content))
            if dosage_patterns > 0:
                importance += min(0.4, 0.1 * dosage_patterns)
            
            # Check for diagnostic criteria which are clinically important
            if re.search(r"diagnostic criteria|differential diagnosis|icd-10|icd-9|cpt code", content, re.IGNORECASE):
                importance += 0.25
            
            # Cap importance between 0.1 and 2.0
            importance = max(0.1, min(2.0, importance))
            
            return round(importance, 2)
            
        except Exception as e:
            logger.error(f"Error calculating clinical importance: {str(e)}")
            return 1.0  # Default to neutral importance on error

    def process_file(self, file_path: str) -> List[Document]:
        """Main entry point to process a medical document file of various formats"""
        try:
            logger.info(f"Starting processing of medical document: {file_path}")
            
            # Check if processed file exists in cache
            if self.use_cache:
                cached_chunks = self._load_from_cache(file_path)
                if cached_chunks:
                    return cached_chunks
            
            # Load and process document
            pages = self.load_document(file_path)
            
            # Split into chunks
            chunks = self.split_document(pages)
            
            # Save to cache for future use
            if self.use_cache:
                self._save_to_cache(file_path, chunks)
            
            logger.info(f"Completed processing: {file_path}")
            return chunks
        
        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {str(e)}")
            raise


# Helper function for easy use
def process_medical_document(file_path: str, chunk_size: int = 1000, use_cache: bool = True) -> List[Document]:
    """Convenience function to process a medical document without creating a class instance"""
    processor = MedicalDocumentProcessor(chunk_size=chunk_size, use_cache=use_cache)
    return processor.process_file(file_path)