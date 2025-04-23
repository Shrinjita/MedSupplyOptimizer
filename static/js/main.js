document.addEventListener('DOMContentLoaded', function() {
    // Add bounce effect to buttons
    const buttons = document.querySelectorAll('.btn');
    buttons.forEach(button => {
        button.addEventListener('click', function(e) {
            this.classList.add('bounce-element');
            setTimeout(() => {
                this.classList.remove('bounce-element');
            }, 1000);
        });
    });

    // File path input and browse button integration
    const filePathInput = document.getElementById('file-path-input');
    const browseButton = document.getElementById('browse-button');
    
    if (browseButton && filePathInput) {
        browseButton.addEventListener('click', function() {
            fetch('/browse_file')
                .then(response => response.json())
                .then(data => {
                    if (data.file_path) {
                        filePathInput.value = data.file_path;
                    } else if (data.error) {
                        console.error('Error browsing for file:', data.error);
                    }
                })
                .catch(error => console.error('Error:', error));
        });
    }

    // Biometric verification animation
    const verifyButtons = document.querySelectorAll('.verify-button');
    verifyButtons.forEach(button => {
        button.addEventListener('click', function() {
            const bioType = this.dataset.bioType;
            const statusElement = document.getElementById(`${bioType}-status`);
            
            if (statusElement) {
                statusElement.innerHTML = '<div class="loader"></div>';
                
                // Simulate verification process
                setTimeout(() => {
                    statusElement.innerHTML = '<i class="fas fa-check-circle text-success"></i> Verified';
                    statusElement.classList.add('bounce-element');
                }, 2000);
            }
        });
    });

    // Flash message animation
    const flashMessages = document.querySelectorAll('.alert');
    flashMessages.forEach(message => {
        // Automatically remove flash messages after 5 seconds
        setTimeout(() => {
            message.style.opacity = '0';
            setTimeout(() => {
                message.remove();
            }, 500);
        }, 5000);
    });
});