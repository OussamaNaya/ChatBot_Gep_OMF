document.addEventListener('DOMContentLoaded', function () {
    const chatForm = document.getElementById('chatForm');
    const chatInput = document.getElementById('chatInput');
    const chatMessages = document.getElementById('chatMessages');
    const chatIcon = document.getElementById("chatIcon");
    const chatContainer = document.getElementById("chatContainer");

    const InitialMessage = "Comment puis-je t'aider ?";
    addBotMessage(InitialMessage);

    chatIcon.addEventListener("click", function () {
        if (chatContainer.classList.contains("hidden")) {
            chatContainer.classList.remove("hidden");
        } else {
            chatContainer.classList.add("hidden");
        }

        if (chatContainer.classList.contains("hidden")) {
            chatIcon.src = "/img/chat_open.jpeg";
        } else {
            chatIcon.src = "/img/logo.gif";
        }
    });

    document.getElementById('chatSendButton').addEventListener('click', function (event) {
        event.preventDefault(); // Empêche l'envoi classique
        const message = chatInput.value.trim();

        if (message === '') return;

        // Affiche immédiatement le message de l'utilisateur
        addUserMessage(message);
        chatInput.value = '';

        // Appel vers l'action Home.Chat via fetch
        fetch('/Home/Chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded', // ou application/json si tu modifies le backend
            },
            body: `message=${encodeURIComponent(message)}`
        })
        .then(response => response.text())
        .then(data => {
            addBotMessage(data);
        })
        .catch(error => {
            console.error('Erreur lors de l\'envoi du message :', error);
            addBotMessage("Une erreur est survenue. Veuillez réessayer.");
        });
    });

    function addUserMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message user-message';
        messageDiv.innerHTML = `
            <div class="message-content">${text}</div>
            <img src="/img/profile.png" class="chat-avatar" alt="User">
        `;
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    function addBotMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message bot-message';
        messageDiv.innerHTML = `
            <img src="/img/Botcolor.png" class="chat-avatar" alt="Bot">
            <div class="message-content">${text}</div>
        `;
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});
