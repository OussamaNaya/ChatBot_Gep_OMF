# ChatBot GEP OMF

## 📋 Description
Application de chatbot intelligente développée avec une architecture RAG (Retrieval-Augmented Generation) et une interface web ASP.NET Core MVC. Ce projet combine l'intelligence artificielle avancée avec une interface utilisateur moderne pour fournir des réponses précises et contextuelles.

## 🏗️ Architecture du Projet

### Structure des Dossiers
```
ChatBot_GPT_OMF/
├── QA RAG/           # Module d'Intelligence Artificielle
└── QA_ASP/           # Application Web ASP.NET Core
```

## 🤖 Module IA - QA RAG

### Fonctionnalités
- **Architecture RAG** : Retrieval-Augmented Generation pour des réponses contextuelles
- **Modèle Llama 3** : Utilisation du modèle Llama 3 en local via HuggingFace
- **API Together AI** : Intégration avec l'API Together AI pour une performance optimisée
- **Base de données vectorielle** : ChromaDB pour le stockage et la recherche de documents
- **Interface Gradio** : Interface de test et de développement

### Technologies Utilisées
- Python
- HuggingFace Transformers
- ChromaDB
- Gradio
- Together AI API
- Llama 3 (Meta)

### Fichiers Principaux
- `Service_QA_RAG.py` : Service principal de questions-réponses
- `app.py` : Application Flask/FastAPI
- `Preparing_Data.py` : Préparation et traitement des données
- `requirements.txt` : Dépendances Python

## 🌐 Application Web - QA_ASP

### Fonctionnalités
- **Interface utilisateur moderne** : Design responsive avec Bootstrap
- **Page d'accueil** : Présentation du chatbot avec icône interactive
- **Page QA** : Formulaire de questions-réponses
- **Architecture MVC** : Séparation claire des responsabilités

### Technologies Utilisées
- ASP.NET Core MVC (.NET 9.0)
- Bootstrap 5
- jQuery
- CSS3/HTML5
- JavaScript

### Structure MVC
- **Controllers** : `HomeController.cs`, `QAController.cs`
- **Views** : Pages Razor pour l'interface utilisateur
- **Models** : Modèles de données
- **wwwroot** : Ressources statiques (CSS, JS, images)

## 🚀 Installation et Configuration

### Prérequis
- Python 3.8+
- .NET 9.0 SDK
- Git

### Installation du Module IA
```bash
cd "QA RAG"
pip install -r requirements.txt
```

### Installation de l'Application Web
```bash
cd QA_ASP
dotnet restore
dotnet build
```

## 💻 Utilisation

### Lancement du Service IA
```bash
cd "QA RAG"
python app.py
```

### Lancement de l'Application Web
```bash
cd QA_ASP
dotnet run
```

L'application sera accessible à l'adresse : `https://localhost:5001`

## 📱 Interface Utilisateur

### Page d'Accueil
- Présentation du chatbot
- Icône interactive pour lancer les conversations
- Design moderne et responsive
<img width="952" alt="Capture1" src="https://github.com/user-attachments/assets/12b6e6fe-840a-4237-a024-1b3a4d9be5f1" />
---
<img width="423" alt="Capture2" src="https://github.com/user-attachments/assets/a32be11c-3e2a-49d5-b1df-7caa5a0b3356" />

---
<img width="397" alt="Capture3" src="https://github.com/user-attachments/assets/5c87c9a7-5544-49fe-98ee-f6f76d072fa6" />




### Page QA
- Formulaire de saisie des questions
- Affichage des réponses du chatbot
- Historique des conversations
<img width="628" alt="Capture4" src="https://github.com/user-attachments/assets/17e2c965-eef2-40bd-9ffe-46e2d5173e5b" />
---
<img width="531" alt="Capture5" src="https://github.com/user-attachments/assets/e7f7e86b-7f21-4f27-9434-9410acbfae8e" />



## 🔧 Configuration

### Variables d'Environnement
- `TOGETHER_API_KEY` : Clé API pour Together AI
- `MODEL_PATH` : Chemin vers le modèle Llama 3 local

### Configuration de la Base de Données
La base de données ChromaDB est configurée dans le dossier `content/chromadb/`

## 🤝 Contribution

1. Fork le projet
2. Créer une branche pour votre fonctionnalité (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 👨‍💻 Auteur

**Oussama Naya**
- GitHub: [@OussamaNaya](https://github.com/OussamaNaya)

## 🙏 Remerciements

- Meta pour le modèle Llama 3
- HuggingFace pour les outils de ML
- Together AI pour leur API
- La communauté open source

---

⭐ **Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile !**
