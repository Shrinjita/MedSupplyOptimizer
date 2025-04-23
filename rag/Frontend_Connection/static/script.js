document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM fully loaded, initializing medical query system...");
    
    // Get references to critical buttons
    const submitButton = document.getElementById('submit-query');
    const queryOptions = document.querySelectorAll('.query-option');
    const queryTypeInput = document.getElementById('query-type');
    const medicalForm = document.getElementById('medical-form');
    const generalForm = document.getElementById('general-form');
    
    // Setup query type selection
    queryOptions.forEach(option => {
        option.addEventListener('click', function() {
            // Remove selected class from all options
            queryOptions.forEach(opt => opt.classList.remove('selected'));
            
            // Add selected class to clicked option
            this.classList.add('selected');
            
            // Update hidden input with the selected query type
            const selectedType = this.getAttribute('data-type');
            queryTypeInput.value = selectedType;
            
            // Show/hide appropriate form based on selected type
            if (selectedType === 'general') {
                if (medicalForm) medicalForm.style.display = 'none';
                if (generalForm) generalForm.style.display = 'block';
            } else {
                if (medicalForm) medicalForm.style.display = 'block';
                if (generalForm) generalForm.style.display = 'none';
                
                // Update medical form fields based on query type
                updateMedicalFormFields(selectedType);
            }
            
            console.log(`Query type changed to: ${selectedType}`);
        });
    });

    // Function to update medical form fields based on query type
    function updateMedicalFormFields(queryType) {
        const symptomsLabel = document.querySelector('label[for="symptoms-input"]');
        const detailsLabel = document.querySelector('label[for="patient-details-input"]');
        const symptomsInput = document.getElementById('symptoms-input');
        const patientDetailsInput = document.getElementById('patient-details-input');
        
        switch(queryType) {
            case 'disease':
                if (symptomsLabel) symptomsLabel.textContent = 'Symptoms:';
                if (detailsLabel) detailsLabel.textContent = 'Patient Details:';
                if (symptomsInput) symptomsInput.placeholder = 'Describe your symptoms in detail...';
                if (patientDetailsInput) patientDetailsInput.placeholder = 'Age, gender, medical history, etc.';
                break;
            case 'recovery':
                if (symptomsLabel) symptomsLabel.textContent = 'Current Condition:';
                if (detailsLabel) detailsLabel.textContent = 'Recovery Goals:';
                if (symptomsInput) symptomsInput.placeholder = 'Describe your current health condition...';
                if (patientDetailsInput) patientDetailsInput.placeholder = 'What recovery outcomes are you seeking?';
                break;
            case 'resources':
                if (symptomsLabel) symptomsLabel.textContent = 'Medical Need:';
                if (detailsLabel) detailsLabel.textContent = 'Location Details:';
                if (symptomsInput) symptomsInput.placeholder = 'What medical resources are you looking for?';
                if (patientDetailsInput) patientDetailsInput.placeholder = 'Your location and accessibility requirements...';
                break;
            default:
                if (symptomsLabel) symptomsLabel.textContent = 'Medical Query:';
                if (detailsLabel) detailsLabel.textContent = 'Additional Details:';
                if (symptomsInput) symptomsInput.placeholder = 'Enter your medical question...';
                if (patientDetailsInput) patientDetailsInput.placeholder = 'Any relevant patient information...';
        }
    }
    
    // Simple markdown parser since marked is not available
    const simpleMarkdown = {
        parse: function(text) {
            if (!text) return '';
            
            return text
                // Headers
                .replace(/^### (.*$)/gim, '<h3>$1</h3>')
                .replace(/^## (.*$)/gim, '<h2>$1</h2>')
                .replace(/^# (.*$)/gim, '<h1>$1</h1>')
                // Bold
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                // Italic
                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                // Lists
                .replace(/^\s*[\*\-]\s*(.*?)$/gm, '<li>$1</li>')
                .replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>')
                // Numbered lists
                .replace(/^\s*\d+\.\s*(.*?)$/gm, '<li>$1</li>')
                .replace(/(<li>.*<\/li>)/gs, '<ol>$1</ol>')
                // Paragraphs
                .replace(/\n\n/g, '</p><p>')
                // Line breaks
                .replace(/\n/g, '<br>');
        }
    };
    
    // Parse the response text into sections
    function parseResponseSections(text) {
        const sections = [];
        
        // Try to find sections based on markdown headers (## Section Name)
        const headerRegex = /##\s+(.*?)(?=\n)([\s\S]*?)(?=##\s+|$)/g;
        let match;
        
        while ((match = headerRegex.exec(text)) !== null) {
            sections.push({
                title: match[1].trim(),
                content: match[2].trim()
            });
        }
        
        // If no markdown headers found, try numbered lists (1. Section Name)
        if (sections.length === 0) {
            const sectionRegex = /(\d+\.\s*[\w\s]+)[\s\n]+([\s\S]+?)(?=\d+\.\s*[\w\s]+[\s\n]|$)/g;
            
            while ((match = sectionRegex.exec(text)) !== null) {
                sections.push({
                    title: match[1].trim().replace(/^\d+\.\s*/, ''),
                    content: match[2].trim()
                });
            }
        }
        
        // If still no sections found, return a single section with all content
        if (sections.length === 0) {
            sections.push({
                title: "Medical Response",
                content: text
            });
        }
        
        return sections;
    }
    
    // Get an appropriate icon for each section
    function getSectionIcon(sectionTitle) {
        const title = sectionTitle.toLowerCase();
        
        if (title.includes('clinical') || title.includes('assessment')) {
            return 'fa-stethoscope';
        } else if (title.includes('diagnosis')) {
            return 'fa-clipboard-list';
        } else if (title.includes('management') || title.includes('treatment') || title.includes('recommendation')) {
            return 'fa-pills';
        } else if (title.includes('follow') || title.includes('monitoring')) {
            return 'fa-calendar-check';
        } else if (title.includes('education') || title.includes('patient')) {
            return 'fa-user-md';
        } else if (title.includes('prognosis') || title.includes('outcome')) {
            return 'fa-chart-line';
        } else if (title.includes('resource')) {
            return 'fa-hospital';
        } else {
            return 'fa-file-medical';  // Default icon
        }
    }
    
    // Improved medical response formatter with better visuals and functionality
    function formatMedicalResponse(responseText, queryType = 'general') {
        try {
            // Remove references section if it exists
            responseText = responseText.replace(/\*\*References\*\*:[\s\S]*$/, '');
            
            // Parse the text into sections
            const sections = parseResponseSections(responseText);
            
            // Create a summary from the first few sentences of the response
            let summaryText = responseText.split(/\.\s+/).slice(0, 2).join('. ') + '.';
            if (summaryText.length > 300) {
                summaryText = summaryText.substring(0, 297) + '...';
            }
            
            // Create HTML with a header summary card
            let formattedHtml = `
                <div class="medical-response-container">
                    <div class="medical-summary-card">
                        <div class="summary-header">
                            <div class="summary-icon">
                                <i class="fas ${getQueryTypeIcon(queryType)}"></i>
                            </div>
                            <div class="summary-title">
                                <h2>${capitalizeFirstLetter(queryType)} Assessment</h2>
                                <div class="timestamp">${new Date().toLocaleDateString()} | ${new Date().toLocaleTimeString()}</div>
                            </div>
                        </div>
                        <div class="summary-content">
                            <p>${summaryText}</p>
                        </div>
                        <div class="action-buttons">
                            <button class="btn-print" onclick="window.print()"><i class="fas fa-print"></i> Print</button>
                            <button class="btn-save" onclick="saveToLocal('medical-report-${Date.now()}')"><i class="fas fa-save"></i> Save</button>
                            <button class="btn-read" onclick="readAloud('.medical-response-container')"><i class="fas fa-volume-up"></i> Read Aloud</button>
                        </div>
                    </div>
                    <div class="section-navigator">
                        <p>Jump to section:</p>
                        <div class="section-links">
                            ${sections.map((section, index) => 
                                `<a href="#section-${index}" class="section-link">${section.title}</a>`).join('')}
                        </div>
                    </div>
                    <div class="medical-response-sections">`;
            
            sections.forEach((section, index) => {
                const iconClass = getSectionIcon(section.title);
                const severityClass = getSeverityClass(section.content);
                
                formattedHtml += `
                    <div id="section-${index}" class="response-section ${severityClass}">
                        <div class="section-header">
                            <h3 class="section-title">
                                <i class="fas ${iconClass}"></i> 
                                ${section.title}
                            </h3>
                            <div class="section-controls">
                                <button class="btn-collapse" onclick="toggleSection(this)">
                                    <i class="fas fa-chevron-up"></i>
                                </button>
                            </div>
                        </div>
                        <div class="section-content">
                            ${formatSectionContent(section.content)}
                        </div>
                    </div>
                `;
            });
            
            formattedHtml += `
                    </div>
                    <div class="disclaimer">
                        <i class="fas fa-exclamation-circle"></i>
                        <p>This information is for educational purposes only and should not replace professional medical advice. 
                        Always consult with a qualified healthcare provider for diagnosis and treatment options.</p>
                    </div>
                </div>
            `;
            
            // Add the required JS functions for interactivity
            addInteractivityFunctions();
            
            return formattedHtml;
        } catch (error) {
            console.error("Error formatting medical response:", error);
            // Fallback to basic formatting
            return `<div class="medical-response-basic">${simpleMarkdown.parse(responseText)}</div>`;
        }
    }

    // Get appropriate icon based on query type
    function getQueryTypeIcon(queryType) {
        switch(queryType.toLowerCase()) {
            case 'disease':
                return 'fa-disease';
            case 'recovery':
                return 'fa-heartbeat';
            case 'resources':
                return 'fa-hospital';
            default:
                return 'fa-notes-medical';
        }
    }

    // Function to determine severity class based on content analysis
    function getSeverityClass(content) {
        const lowercasedContent = content.toLowerCase();
        
        if (lowercasedContent.includes('emergency') || 
            lowercasedContent.includes('immediate attention') ||
            lowercasedContent.includes('urgent care')) {
            return 'severity-high';
        } else if (lowercasedContent.includes('monitor closely') || 
                   lowercasedContent.includes('follow-up') ||
                   lowercasedContent.includes('consult with')) {
            return 'severity-medium';
        }
        
        return 'severity-normal';
    }

    // Helper utility function to capitalize first letter
    function capitalizeFirstLetter(string) {
        return string.charAt(0).toUpperCase() + string.slice(1);
    }

    // Format section content with enhanced formatting
    function formatSectionContent(content) {
        let formatted = content
            // Format list items
            .replace(/^\s*[•*-]\s+(.*?)$/gm, '<li>$1</li>')
            // Format important points with improved highlighting
            .replace(/(!{1,3}|IMPORTANT:?|NOTE:?|CAUTION:?|WARNING:?|KEY FINDING:?)\s+(.*?)(?:\r?\n|\r|$)/g, 
                    '<div class="highlighted-point"><i class="fas fa-exclamation-triangle"></i> $2</div>')
            // Highlight medical terms in parentheses
            .replace(/\(([^)]+)\)/g, '<span class="term-in-parens" title="Medical term">($1)</span>')
            // Format paragraph breaks
            .replace(/\n\n/g, '</p><p>')
            // Format line breaks
            .replace(/\n/g, '<br>')
            // Add tooltips for common medical abbreviations
            .replace(/\b(BP|HR|RR|SpO2|T|Hb|WBC|BUN|Cr|Na|K|Cl|HCO3|FEV1|FVC|BMI)\b/g, 
                    '<abbr title="Medical abbreviation" class="medical-abbr">$1</abbr>');
        
        // Wrap content in paragraph tags if it doesn't already contain list items
        if (!formatted.includes('<li>')) {
            formatted = `<p>${formatted}</p>`;
        } else {
            // If it contains list items, wrap them in a ul
            formatted = formatted.replace(/(<li>.*?<\/li>)+/g, '<ul>$&</ul>');
        }
        
        return formatted;
    }

    // Add the interactivity functions to the document if they don't exist
    function addInteractivityFunctions() {
        if (!window.toggleSection) {
            window.toggleSection = function(button) {
                const section = button.closest('.response-section');
                const content = section.querySelector('.section-content');
                const icon = button.querySelector('i');
                
                if (content.style.display === 'none') {
                    content.style.display = 'block';
                    icon.classList.remove('fa-chevron-down');
                    icon.classList.add('fa-chevron-up');
                } else {
                    content.style.display = 'none';
                    icon.classList.remove('fa-chevron-up');
                    icon.classList.add('fa-chevron-down');
                }
            };
        }
        
        if (!window.saveToLocal) {
            window.saveToLocal = function(filename) {
                const content = document.querySelector('.medical-response-container').innerHTML;
                const blob = new Blob([`<html><head><title>Medical Report</title>
                    <style>${document.getElementById('medical-highlight-styles').textContent}</style></head>
                    <body>${content}</body></html>`], 
                    {type: 'text/html'});
                const url = URL.createObjectURL(blob);
                
                const a = document.createElement('a');
                a.href = url;
                a.download = `${filename}.html`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            };
        }
        
        if (!window.readAloud) {
            window.readAloud = function(selector) {
                const textToRead = document.querySelector(selector).textContent.replace(/\s+/g, ' ').trim();
                
                if ('speechSynthesis' in window) {
                    const speech = new SpeechSynthesisUtterance();
                    speech.text = textToRead;
                    speech.volume = 1;
                    speech.rate = 0.9;
                    speech.pitch = 1;
                    window.speechSynthesis.speak(speech);
                } else {
                    alert("Text-to-speech functionality is not supported in your browser.");
                }
            };
        }
    }

    // Enhanced medical term highlighting
    function applyMedicalHighlighting() {
        console.log("Applying enhanced medical highlighting");
        
        // Get all elements that need highlighting
        const medicalContent = document.querySelectorAll('.section-content p, .section-content li');
        
        // Extended list of medical terms to highlight
        const medicalTerms = [
            // Diagnoses
            'diagnosis', 'condition', 'disease', 'disorder', 'syndrome', 'infection', 'inflammation',
            // Symptoms
            'symptom', 'pain', 'discomfort', 'acute', 'chronic', 'fatigue', 'fever', 'nausea',
            // Treatments
            'treatment', 'therapy', 'medication', 'prescription', 'dosage', 'regimen', 'intervention',
            // Clinical terms
            'clinical', 'prognosis', 'etiology', 'pathology', 'assessment', 'evaluation',
            // Body systems
            'cardiac', 'respiratory', 'neurological', 'gastrointestinal', 'musculoskeletal',
            // Procedures
            'surgery', 'procedure', 'test', 'scan', 'imaging', 'biopsy', 'screening',
            // Healthcare
            'hospital', 'clinic', 'physician', 'specialist', 'referral', 'consultation'
        ];
        
        // Create regex pattern for all terms (case insensitive, word boundaries)
        const medicalTermPattern = new RegExp(`\\b(${medicalTerms.join('|')})\\b`, 'gi');
        
        // Highlight medical terms
        medicalContent.forEach(el => {
            // Don't process elements that are already styled
            if (el.classList.contains('processed-highlight')) return;
            
            // Replace medical terms with highlighted spans
            el.innerHTML = el.innerHTML.replace(medicalTermPattern, 
                '<span class="medical-term-highlight" title="Medical terminology">$1</span>');
            
            // Mark as processed
            el.classList.add('processed-highlight');
        });
        
        // Enhance highlighted points
        document.querySelectorAll('.highlighted-point').forEach(point => {
            // Add pulsing effect to important points
            point.classList.add('pulse-highlight');
        });
        
        // Add enhanced CSS if it doesn't exist
        if (!document.getElementById('medical-highlight-styles')) {
            const style = document.createElement('style');
            style.id = 'medical-highlight-styles';
            style.textContent = `
                .medical-response-container {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    max-width: 900px;
                    margin: 0 auto;
                    color: #333;
                }
                
                .medical-summary-card {
                    background: linear-gradient(135deg, #5885af, #274472);
                    color: white;
                    border-radius: 10px;
                    padding: 20px;
                    margin-bottom: 30px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                }
                
                .summary-header {
                    display: flex;
                    align-items: center;
                    margin-bottom: 15px;
                }
                
                .summary-icon {
                    font-size: 2.5em;
                    margin-right: 15px;
                }
                
                .summary-title h2 {
                    margin: 0;
                    font-size: 1.6em;
                }
                
                .timestamp {
                    opacity: 0.8;
                    font-size: 0.85em;
                    margin-top: 5px;
                }
                
                .summary-content {
                    margin-bottom: 20px;
                    font-size: 1.1em;
                    line-height: 1.4;
                }
                
                .action-buttons {
                    display: flex;
                    gap: 10px;
                }
                
                .action-buttons button {
                    background: rgba(255,255,255,0.2);
                    border: none;
                    border-radius: 5px;
                    padding: 8px 15px;
                    color: white;
                    cursor: pointer;
                    font-size: 0.9em;
                    transition: background 0.3s;
                    display: flex;
                    align-items: center;
                    gap: 5px;
                }
                
                .action-buttons button:hover {
                    background: rgba(255,255,255,0.3);
                }
                
                .section-navigator {
                    display: flex;
                    align-items: center;
                    margin-bottom: 20px;
                    flex-wrap: wrap;
                    gap: 10px;
                }
                
                .section-navigator p {
                    margin: 0;
                    font-weight: 600;
                }
                
                .section-links {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                }
                
                .section-link {
                    background: #f1f5f9;
                    padding: 5px 12px;
                    border-radius: 20px;
                    text-decoration: none;
                    color: #1e3a8a;
                    font-size: 0.9em;
                    transition: all 0.2s;
                }
                
                .section-link:hover {
                    background: #dbeafe;
                    transform: translateY(-2px);
                }
                
                .response-section {
                    margin-bottom: 20px;
                    padding: 0;
                    border-radius: 8px;
                    background-color: white;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                    overflow: hidden;
                    border-left: 5px solid #64748b;
                }
                
                .severity-high {
                    border-left: 5px solid #ef4444;
                }
                
                .severity-medium {
                    border-left: 5px solid #f59e0b;
                }
                
                .severity-normal {
                    border-left: 5px solid #22c55e;
                }
                
                .section-header {
                    padding: 15px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    background: #f8fafc;
                }
                
                .section-title {
                    margin: 0;
                    color: #1e3a8a;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    font-size: 1.2em;
                }
                
                .section-controls .btn-collapse {
                    background: none;
                    border: none;
                    cursor: pointer;
                    color: #64748b;
                    width: 30px;
                    height: 30px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                
                .section-controls .btn-collapse:hover {
                    background: #e2e8f0;
                }
                
                .section-content {
                    padding: 20px;
                    line-height: 1.6;
                }
                
                .section-content p {
                    margin-top: 0;
                }
                
                .medical-term-highlight {
                    background-color: rgba(255, 217, 102, 0.3);
                    border-radius: 3px;
                    padding: 0 3px;
                    font-weight: 500;
                    position: relative;
                    cursor: help;
                }
                
                .medical-term-highlight:hover {
                    background-color: rgba(255, 217, 102, 0.5);
                }
                
                .medical-abbr {
                    border-bottom: 1px dotted #666;
                    cursor: help;
                }
                
                .term-in-parens {
                    font-style: italic;
                    color: #4b5563;
                }
                
                .pulse-highlight {
                    animation: pulse-bg 2s ease-in-out;
                    border-left: 3px solid #4caf50;
                    padding-left: 10px;
                    background-color: rgba(76, 175, 80, 0.1);
                    margin: 8px 0;
                    padding: 8px 12px;
                    border-radius: 4px;
                }
                
                .highlighted-point {
                    position: relative;
                    margin: 15px 0;
                    padding: 12px 15px 12px 40px;
                    background-color: rgba(66, 133, 244, 0.1);
                    border-left: 3px solid #4285f4;
                    border-radius: 4px;
                }
                
                .highlighted-point i {
                    position: absolute;
                    left: 12px;
                    top: 12px;
                    color: #4285f4;
                }
                
                @keyframes pulse-bg {
                    0% { background-color: rgba(76, 175, 80, 0); }
                    50% { background-color: rgba(76, 175, 80, 0.2); }
                    100% { background-color: rgba(76, 175, 80, 0.1); }
                }
                
                .disclaimer {
                    margin-top: 30px;
                    padding: 15px;
                    background-color: #f3f4f6;
                    border-radius: 6px;
                    font-size: 0.9em;
                    color: #4b5563;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }
                
                .disclaimer i {
                    font-size: 1.5em;
                    color: #6b7280;
                }
                
                .disclaimer p {
                    margin: 0;
                }
                
                .loading-indicator {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    padding: 40px;
                }
                
                .spinner {
                    width: 50px;
                    height: 50px;
                    border: 4px solid rgba(0,0,0,0.1);
                    border-radius: 50%;
                    border-top-color: #1976d2;
                    animation: spin 0.8s linear infinite;
                    margin-bottom: 20px;
                }
                
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                
                .error-message {
                    padding: 20px;
                    background-color: #fee2e2;
                    border-left: 4px solid #ef4444;
                    border-radius: 4px;
                    color: #b91c1c;
                }
                
                @media (max-width: 768px) {
                    .medical-response-container {
                        padding: 0 15px;
                    }
                    
                    .summary-header {
                        flex-direction: column;
                        text-align: center;
                    }
                    
                    .summary-icon {
                        margin-right: 0;
                        margin-bottom: 10px;
                    }
                    
                    .action-buttons {
                        justify-content: center;
                        flex-wrap: wrap;
                    }
                    
                    .section-navigator {
                        flex-direction: column;
                        align-items: flex-start;
                    }
                    
                    .section-links {
                        width: 100%;
                    }
                }
                
                /* For printing */
                @media print {
                    body {
                        background: white !important;
                        color: black !important;
                    }
                    
                    .medical-response-container {
                        max-width: 100% !important;
                    }
                    
                    .action-buttons, .btn-collapse {
                        display: none !important;
                    }
                    
                    .response-section {
                        break-inside: avoid;
                        box-shadow: none !important;
                        border: 1px solid #ddd;
                    }
                    
                    .section-content {
                        display: block !important;
                    }
                }
            `;
            document.head.appendChild(style);
        }
        
        console.log("Enhanced medical highlighting applied");
    }
    
    // Handle medical query submission
    if (submitButton) {
        console.log("Found medical query submit button");
        
        submitButton.addEventListener('click', function(e) {
            e.preventDefault();
            console.log("Medical query button clicked");
            
            const queryType = document.getElementById('query-type') ? 
                document.getElementById('query-type').value : 'general';
            const symptomsInput = document.getElementById('symptoms-input');
            const patientDetailsInput = document.getElementById('patient-details-input');
            
            // Get form values
            const symptoms = symptomsInput ? symptomsInput.value.trim() : '';
            const patientDetails = patientDetailsInput ? patientDetailsInput.value.trim() : '';
            
            console.log(`Processing ${queryType} query with symptoms: "${symptoms}" and details: "${patientDetails}"`);
            
            // Show loading indicator
            const resultContainer = document.getElementById('result-container');
            if (resultContainer) {
                resultContainer.innerHTML = '<div class="loading-indicator"><div class="spinner"></div><p>Analyzing medical information...</p></div>';
                resultContainer.style.display = 'block';
            }
            
            // Send request to backend
            fetch('/api/medical-query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    queryType: queryType,
                    symptoms: symptoms,
                    patientDetails: patientDetails
                })
            })
            .then(response => response.json())
            .then(data => {
                console.log("Response received:", data);
                
                if (resultContainer) {
                    if (data.error) {
                        resultContainer.innerHTML = `<div class="error-message">${data.error}</div>`;
                    } else {
                        // Format and display the response with highlighting
                        resultContainer.innerHTML = formatMedicalResponse(data.result, queryType);
                        
                        // Apply highlighting after render
                        applyMedicalHighlighting();
                    }
                }
            })
            .catch(error => {
                console.error("Error:", error);
                if (resultContainer) {
                    resultContainer.innerHTML = `<div class="error-message">Error: ${error.message}</div>`;
                }
            });
        });
    } else {
        console.warn("Medical query submit button not found!");
    }
    
    // Also set up the general query form submit button if it exists
    const generalSubmitButton = document.getElementById('submit-general-query');
    if (generalSubmitButton) {
        generalSubmitButton.addEventListener('click', function(e) {
            e.preventDefault();
            const queryInput = document.getElementById('query-input');
            const query = queryInput ? queryInput.value.trim() : '';
            
            if (!query) {
                alert('Please enter a medical question');
                return;
            }
            
            // Show loading
            const resultContainer = document.getElementById('result-container');
            if (resultContainer) {
                resultContainer.innerHTML = '<div class="loading-indicator"><div class="spinner"></div><p>Processing medical query...</p></div>';
                resultContainer.style.display = 'block';
            }
            
            // Send request
            fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query
                })
            })
            .then(response => response.json())
            .then(data => {
                if (resultContainer) {
                    if (data.error) {
                        resultContainer.innerHTML = `<div class="error-message">${data.error}</div>`;
                    } else {
                        // Format and display the response with highlighting
                        resultContainer.innerHTML = formatMedicalResponse(data.result, 'general');
                        
                        // Apply highlighting after render
                        applyMedicalHighlighting();
                    }
                }
            })
            .catch(error => {
                if (resultContainer) {
                    resultContainer.innerHTML = `<div class="error-message">Error: ${error.message}</div>`;
                }
            });
        });
    }
});