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
            uploadedFiles[file.name] = URL.createObjectURL(file);
        }

        fetch('/upload', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
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

        for (const fileName in uploadedFiles) {
            const option = document.createElement('option');
            option.value = fileName;
            option.textContent = fileName;
            documentSelect.appendChild(option);
        }
    }

    function displayDocument() {
        const selectedFile = documentSelect.value;
        if (selectedFile) {
            documentViewer.src = uploadedFiles[selectedFile];
        } else {
            documentViewer.src = '';
        }
    }

    documentSelect.addEventListener('change', displayDocument);

    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const userMessage = chatInput.value.trim();
        if (userMessage === '') return;

        addChatMessage('user', userMessage);
        chatInput.value = '';

        // Simulate bot response
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: userMessage, document: documentSelect.value })
        })
        .then(response => response.json())
        .then(data => {
            addChatMessage('bot', data.reply);
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
