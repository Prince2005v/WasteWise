document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const submitBtn = document.getElementById('submitBtn');
    const uploadForm = document.getElementById('uploadForm');

    // File input change handler
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop handlers
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('drop', handleDrop);

    // Form submit handler
    uploadForm.addEventListener('submit', handleFormSubmit);

    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            displayFileInfo(file);
        }
    }

    function handleDragOver(e) {
        e.preventDefault();
        dropZone.classList.add('dragover');
    }

    function handleDragLeave(e) {
        e.preventDefault();
        dropZone.classList.remove('dragover');
    }

    function handleDrop(e) {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            if (isValidImageFile(file)) {
                fileInput.files = files;
                displayFileInfo(file);
            } else {
                showAlert('Please select a valid image file (PNG, JPG, JPEG, GIF, WebP)', 'danger');
            }
        }
    }

    function displayFileInfo(file) {
        fileName.textContent = `${file.name} (${formatFileSize(file.size)})`;
        fileInfo.classList.remove('d-none');
        fileInfo.classList.add('fade-in');
        submitBtn.disabled = false;
        
        // Preview image if possible
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = function(e) {
                // You could add image preview here if desired
            };
            reader.readAsDataURL(file);
        }
    }

    function clearFile() {
        fileInput.value = '';
        fileInfo.classList.add('d-none');
        submitBtn.disabled = true;
    }

    function isValidImageFile(file) {
        const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/webp'];
        return validTypes.includes(file.type);
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    function handleFormSubmit(e) {
        const file = fileInput.files[0];
        if (!file) {
            e.preventDefault();
            showAlert('Please select a file first', 'danger');
            return;
        }

        if (!isValidImageFile(file)) {
            e.preventDefault();
            showAlert('Please select a valid image file', 'danger');
            return;
        }

        if (file.size > 16 * 1024 * 1024) { // 16MB
            e.preventDefault();
            showAlert('File size must be less than 16MB', 'danger');
            return;
        }

        // Show loading state
        submitBtn.innerHTML = '<span class="loading-spinner me-2"></span>Classifying...';
        submitBtn.disabled = true;
    }

    function showAlert(message, type = 'info') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        const container = document.querySelector('.container');
        const firstChild = container.firstElementChild;
        container.insertBefore(alertDiv, firstChild);

        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (alertDiv && alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }

    // Global function for clearing file (called from HTML)
    window.clearFile = clearFile;
});

// API usage example (for developers)
function classifyImageViaAPI(imageFile) {
    const formData = new FormData();
    formData.append('file', imageFile);

    return fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log('Classification result:', data.prediction);
            console.log('Disposal tips:', data.disposal_tips);
            return data;
        } else {
            throw new Error(data.error || 'Classification failed');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        throw error;
    });
}

// Make API function globally available for testing
window.classifyImageViaAPI = classifyImageViaAPI;
