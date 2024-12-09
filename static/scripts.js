document.addEventListener('DOMContentLoaded', () => {
    const documentFilesInput = document.getElementById('document-files');
    const uploadButton = document.getElementById('upload-button');
    const documentSelect = document.getElementById('document-select');
    const documentViewer = document.getElementById('document-viewer');
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');

    let uploadedFiles = {};

    uploadButton.addEventListener('click', () => {
        const files = documentFilesInput.files;
        if (files.length === 0) {
            alert('Please select at least one document to upload.');
            return;
        }

        const formData = new FormData();
        for (const file of files) {
            formData.append('files', file);
        }

        fetch('/upload', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert('Error uploading documents: ' + data.error);
                return;
            }

            // Clear existing uploadedFiles
            uploadedFiles = {};

            // Store the document IDs and original names
            data.files.forEach(fileInfo => {
                const originalName = fileInfo.original_filename;
                const storedName = fileInfo.stored_filename;  // This is the document_id
                uploadedFiles[storedName] = {
                    originalName: originalName
                };
            });

            alert('Documents uploaded successfully!');
            populateDocumentSelect();
            displayDocument();
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });

    function populateDocumentSelect() {
        documentSelect.innerHTML = '';
        const defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.disabled = true;
        defaultOption.selected = true;
        defaultOption.textContent = 'Select a document';
        documentSelect.appendChild(defaultOption);

        for (const documentId in uploadedFiles) {
            const option = document.createElement('option');
            option.value = documentId;  // Use stored_filename as the value
            option.textContent = uploadedFiles[documentId].originalName;
            documentSelect.appendChild(option);
        }
    }

    function displayDocument() {
        const selectedDocumentId = documentSelect.value;
        if (selectedDocumentId) {
            documentViewer.src = `/view_document/${selectedDocumentId}`;
        } else {
            documentViewer.src = '';
        }
    }

    documentSelect.addEventListener('change', displayDocument);

    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const userMessage = chatInput.value.trim();
        if (userMessage === '') return;

        const selectedDocumentId = documentSelect.value;
        if (!selectedDocumentId) {
            alert('Please select a document to chat with.');
            return;
        }

        addChatMessage('user', userMessage);
        chatInput.value = '';

        // Send the user message and document ID to the backend
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: userMessage, document_id: selectedDocumentId })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                addChatMessage('bot', 'Error: ' + data.error);
            } else {
                addChatMessage('bot', data.answer);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            addChatMessage('bot', 'An error occurred. Please try again.');
        });
    });

    function addChatMessage(sender, message) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('chat-message', sender);
        const messageContent = document.createElement('p');
        messageContent.textContent = message;
        messageElement.appendChild(messageContent);
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});