import chromadb
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from together import Together


# Initialisation globale de ChromaDB
db = None

def init_db(db_path="./content/chromadb"):
    global db
    if db is None:
        try:
            os.makedirs(db_path, exist_ok=True)
            db = chromadb.PersistentClient(path=db_path)
            print(f"✅ ChromaDB initialisé avec succès à l'emplacement : {db_path}")
        except Exception as e:
            print(f"❌ Erreur lors de l'initialisation de ChromaDB : {e}")
            db = None
    print(f"🔍 Valeur de db après initialisation : {db}")  # Ajout de debug
    return db

def add_document(role: str, question: str, answer: str):
    """Ajoute une question-réponse dans la collection associée au rôle spécifié"""
    db_instance = init_db()  # On utilise l'instance globale

    if db_instance is None:
        print("⚠️ Impossible d'ajouter des documents : ChromaDB n'est pas disponible.")
        return

    roles = ["Simple", "Dev", "Admin"]
    collections = {role: db_instance.get_or_create_collection(name=f"qa_{role}") for role in roles}

    if role not in collections:
        print("❌ Rôle invalide!")
        return

    collection = collections[role]
    collection.add(
        documents=[question],
        metadatas=[{"answer": answer}],
        ids=[f"{role}_{question}"]  # Identifiant unique basé sur le rôle et la question
    )
    print(f"✅ Ajouté avec succès dans la collection {role}.")

def ask_autre_role(role: str, question: str):

    seuil = 0.5
    message_question_existante_autre_role = "Cette question associée à un autre rôle."
    message_hors_contexte = "Cette question ne correspond à aucun des rôles définis."

    db_instance = init_db()  # On utilise l'instance globale

    if db_instance is None:
        print("⚠️ Impossible d'ajouter des documents : ChromaDB n'est pas disponible.")
        return

    roles = ["Simple", "Dev", "Admin"]
    collections = {role: db_instance.get_or_create_collection(name=f"qa_{role}") for role in roles}

    for rol in collections:
        if rol != role:
            collection = collections[rol]
            results = collection.query(query_texts=[question], n_results=1)

            if results and results.get("documents"):
                distance = results["distances"][0][0] if results["distances"] else None
                if distance is not None and distance <= seuil:
                    return True
    return False  # Ajoute un return par défaut pour éviter des erreurs

def ask_question(role: str, question: str):
    """Recherche la réponse en fonction du rôle donné"""

    seuil = 0.5
    message_question_existante_autre_role = "Cette question associée à un autre rôle."
    message_hors_contexte = "Cette question ne correspond à aucun des rôles définis."

    db_instance = init_db()  # On utilise l'instance globale

    if db_instance is None:
        print("⚠️ Impossible d'ajouter des documents : ChromaDB n'est pas disponible.")
        return

    roles = ["Simple", "Dev", "Admin"]
    collections = {role: db_instance.get_or_create_collection(name=f"qa_{role}") for role in roles}

    if role not in collections:
        return "Rôle invalide!"

    collection = collections[role]
    results = collection.query(query_texts=[question], n_results=1)

    if results and results.get("documents"):
        distance = results["distances"][0][0] if results["distances"] else None

        if distance <= seuil:
          answer = results["metadatas"][0][0]["answer"] if results["metadatas"] else "Réponse non disponible"
          #return f"Réponse: {answer}, Pertinence (distance): {distance:.4f}"
          return answer

        else :
          temp = ask_autre_role(role, question)
          if temp:
            return message_question_existante_autre_role
          else:
            return message_hors_contexte
          #return "Seuil elvee !"

        #return f"Réponse: {answer}, Pertinence (distance): {distance:.4f}"

    return "Aucune réponse trouvée"

def Load_LLaMA2_From_HuggingFace():
    from huggingface_hub import login
    login()


    # Charger le modèle sans quantification 4 bits, mais avec exécution sur CPU
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Charger le modèle directement sur CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": "cpu"}  # Forcer l'exécution sur CPU
    )

    return model, tokenizer

def Cloner_LLaMA_2():
    from huggingface_hub import login
    login()


    # Charger le modèle sans quantification 4 bits, mais avec exécution sur CPU
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Charger le modèle directement sur CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": "cpu"}  # Forcer l'exécution sur CPU
    )

    # Définir un chemin pour la sauvegarde locale
    save_path = "./llama-2-7b-chat-quantized"

    # Sauvegarder le modèle et le tokenizer
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"✅ Modèle et tokenizer sauvegardés localement dans : {save_path}")

def Cloner_LLaMA_3():
    # Connexion à Hugging Face
    login()

    # Nom du modèle
    model_name = "meta-llama/Llama-3.3-70B-Instruct"

    # Charger le tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Charger le modèle directement sur CPU sans quantification
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"  # Forcer l'exécution sur CPU
    )

    # Définir un chemin pour la sauvegarde locale
    save_path = "./Llama-3-70B-Standard"

    # Sauvegarder le modèle et le tokenizer
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"✅ Modèle et tokenizer sauvegardés localement dans : {save_path}")

def charger_model(path: str):
    """Charge un modèle en 4-bit et son tokenizer depuis un dossier local"""
    
    # Vérification GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Configuration 4-bit pour bitsandbytes
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    # Charger le tokenizer et le modèle
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map=device,  # Change "auto" par "cuda" si possible
        quantization_config=bnb_config
    )

    print(f"✅ Modèle et tokenizer chargés depuis : {path} sur {device.upper()}")

    return model, tokenizer
 
def charger_model_llama3(path: str):
    """Charge un modèle sans quantification (en 32 bits) et son tokenizer depuis un dossier local"""

    # Vérification GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Charger le tokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)

    # Charger le modèle en 32 bits (pas de quantification)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map=device,  # Utiliser "auto" si tu veux que Hugging Face gère la distribution
        torch_dtype=torch.float32  # Charger en 32 bits pour éviter les problèmes de quantification
    )

    print(f"✅ Modèle et tokenizer chargés depuis : {path} sur {device.upper()}")

    return model, tokenizer

def query_Model(model, tokenizer, question, reponse):

    # Charger le modèle sur le GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('==> Device is : ', device)

    # Format the input prompt
    #formatted_prompt = f"Question: {question}\n\nRespons: {response}"
    formatted_prompt = f"""
                              Voici une question posée par un utilisateur et la réponse récupérée à partir de ChromaDB.

                              🔹 **Instructions claires :**
                              - **Si la réponse récupérée est valide**, reformule-la en français avec plus de contexte et d'explications. Ajoute des émojis dynamiquement pour illustrer les concepts (ex : 🌍 pour un lieu, 🧠 pour une explication, ✅ pour une réponse correcte). Termine la réponse par ✅.
                              - **Si la réponse est exactement "Cette question ne correspond à aucun des rôles définis.", alors affiche exactement :**
                                **"Cette question ne correspond à aucun des rôles définis. ❌"**
                              - **Si la réponse est exactement " Cette question associée à un autre rôle. ", alors affiche exactement :**
                                **"Cette question est associée à un autre rôle. 🔄"**
                              - **Ne génère aucun autre texte en dehors des trois cas définis ci-dessus.**

                              **⚠️ Attention :** Tu dois suivre ces règles strictement et ne pas ajouter d'explications supplémentaires si la réponse ne correspond pas aux données disponibles.

                              **Longueur maximale de la réponse générée : 500 caractères.**

                              Question : {question}
                              Réponse récupérée : {reponse}
                              💡 Réponse en français :
                        """

    # Tokeniser l'entrée et envoyer sur le GPU
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    # Générer la réponse
    outputs = model.generate(**inputs, max_length=800) # Input length of input_ids is X

    # Décoder et afficher la réponse
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def Ask_Llama3_With_TogetherAI(question, reponse):
    client = Together(api_key = '5e4309c119d07b71ec696104dd005ff457cde97c91e5ea31fda07096b3723cc1')

    #formatted_prompt = f"Question: {question}\n\nRespons: {response}"
    prompt = f"""
                              Voici une question posée par un utilisateur et la réponse récupérée à partir de ChromaDB.

                              🔹 **Instructions claires :**
                              - **Si la réponse récupérée est valide**, reformule-la en français avec plus de contexte et d'explications. Ajoute des émojis dynamiquement pour illustrer les concepts (ex : 🌍 pour un lieu, 🧠 pour une explication, ✅ pour une réponse correcte). Termine la réponse par ✅.
                              - **Si la réponse est exactement "Cette question ne correspond à aucun des rôles définis.", alors affiche exactement :**
                                **"Cette question ne correspond à aucun des rôles définis. ❌"**
                              - **Si la réponse est exactement " Cette question associée à un autre rôle. ", alors affiche exactement :**
                                **"Cette question est associée à un autre rôle. 🔄"**
                              - **Ne génère aucun autre texte en dehors des trois cas définis ci-dessus.**

                              **⚠️ Attention :** Tu dois suivre ces règles strictement et ne pas ajouter d'explications supplémentaires si la réponse ne correspond pas aux données disponibles.

                              **Longueur maximale de la réponse générée : 500 caractères.**

                              Question : {question}
                              Réponse récupérée : {reponse}
                              💡 Réponse en français :
                        """
    
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[
          {
            "role": "user",
            "content": prompt
          }
        ]
    )

    return response.choices[0].message.content

def query_llama(question, reponse):

    # Charger le modèle sur le GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('==> Device is : ', device)

    # Format the input prompt
    #formatted_prompt = f"Question: {question}\n\nRespons: {response}"
    formatted_prompt = f"""
                              Voici une question posée par un utilisateur et la réponse récupérée à partir de ChromaDB.

                              🔹 **Instructions claires :**
                              - **Si la réponse récupérée est valide**, reformule-la en français avec plus de contexte et d'explications. Ajoute des émojis dynamiquement pour illustrer les concepts (ex : 🌍 pour un lieu, 🧠 pour une explication, ✅ pour une réponse correcte). Termine la réponse par ✅.
                              - **Si la réponse est exactement "Cette question ne correspond à aucun des rôles définis.", alors affiche exactement :**
                                **"Cette question ne correspond à aucun des rôles définis. ❌"**
                              - **Si la réponse est exactement " Cette question associée à un autre rôle. ", alors affiche exactement :**
                                **"Cette question est associée à un autre rôle. 🔄"**
                              - **Ne génère aucun autre texte en dehors des trois cas définis ci-dessus.**

                              **⚠️ Attention :** Tu dois suivre ces règles strictement et ne pas ajouter d'explications supplémentaires si la réponse ne correspond pas aux données disponibles.

                              **Longueur maximale de la réponse générée : 500 caractères.**

                              Question : {question}
                              Réponse récupérée : {reponse}
                              💡 Réponse en français :
                        """

    model_path = "./llama-2-7b-chat-quantized"
    model, tokenizer = charger_model(model_path)


    # Tokeniser l'entrée et envoyer sur le GPU
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    # Générer la réponse
    outputs = model.generate(**inputs, max_length=800) # Input length of input_ids is X

    # Décoder et afficher la réponse
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def query(query, role):
  message_question_existante_autre_role = "Cette question associée à un autre rôle."
  message_hors_contexte = "Cette question ne correspond à aucun des rôles définis."

  print('query : ', query)
  print('role : ', role)
  response = ask_question(role, query)
  print('response ask_question : ', response)

  if response == message_question_existante_autre_role:
    return response + ' 🔄'
  elif response==message_hors_contexte:
    return response + ' ❌'
  else:
    #test_answer = query_Model(model, tokenizer, query, response)
    test_answer = Ask_Llama3_With_TogetherAI(query, response)

    # Extraction de la réponse en français après "💡 Réponse en français :"
    start_index = test_answer.find("💡 Réponse en français :") + len("💡 Réponse en français :")
    filtered_response = test_answer[start_index:].strip()

    # Enlever "Correcte !" à la fin si nécessaire
    filtered_response = filtered_response.replace("Correct", "").strip()

    return filtered_response
  

# Fonction pour afficher l'utilisation actuelle de la mémoire GPU
def afficher_mem_gpu():
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated() / 1024**3  # en Go
        reserved_memory = torch.cuda.memory_reserved() / 1024**3  # en Go
        print(f"Mémoire allouée: {allocated_memory:.2f} Go")
        print(f"Mémoire réservée: {reserved_memory:.2f} Go")
    else:
        print("Aucun GPU disponible")

# Fonction pour libérer la mémoire GPU
def liberer_mem_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Libère la mémoire non utilisée
        print("Mémoire GPU libérée.")
    else:
        print("Aucun GPU disponible")



# -- Main --
if __name__ == "__main__":


    ## 1. Cree la base de donnees
    #db = init_db()

    ## 2. Cree les Collections
    ##roles = ["Simple", "Dev", "Admin"]
    #collections = {role: db.get_or_create_collection(name=f"qa_{role}") for role in roles}
    #print(collections)

    ## 3. Add/ Ask questions
    ## 4. Insertion des donnees
        # Simple Utilisateur
    #add_document("Simple", "Quelle est la capitale de la France ?", "Paris")
    #add_document("Simple", "Quel est le plus grand océan du monde ?", "Océan Pacifique")
    #add_document("Simple", "Combien de continents y a-t-il sur Terre ?", "7")

        # Dev Utilisateur
    #add_document("Dev", "Qu'est-ce qu'une API ?", "Interface de programmation permettant aux applications de communiquer entre elles.")
    #add_document("Dev", "Quelle est la différence entre un tableau et une liste en Python ?", "Un tableau a une taille fixe, alors qu'une liste est dynamique.")
    #add_document("Dev", "Qu'est-ce qu'un framework ?", "Un ensemble d'outils et de bibliothèques permettant de développer des applications rapidement.")

        # Admin Utilisateur
    #add_document("Admin", "Comment installer Apache sur un serveur Linux ?", "Utiliser la commande 'sudo apt install apache2' sur une machine Ubuntu.")
    #add_document("Admin", "Que faire en cas de 'Disk Full' sur un serveur Linux ?", "Libérer de l'espace disque en supprimant des fichiers inutiles.")
    #add_document("Admin", "Comment vérifier les journaux système sur un serveur Linux ?", "Utiliser la commande 'journalctl' pour consulter les journaux.")

        # Questions Communes
    # Question 1
    #add_document("Simple", "Qu'est-ce que le cloud ?", "Un espace de stockage en ligne pour conserver des fichiers accessibles partout.")
    #add_document("Dev", "Qu'est-ce que le cloud ?", "Une infrastructure qui permet d'héberger et d'exécuter des applications à distance.")
    #add_document("Admin", "Qu'est-ce que le cloud ?", "Un ensemble de serveurs distants offrant des services de stockage, de calcul et de réseau.")



    # Test
    '''
    list_simple = [
        ("Quelle est la capitale de la France ?", "Paris"),
        ("Quel est le plus grand océan du monde ?", "Océan Pacifique"),
        ("Combien de continents y a-t-il sur Terre ?", "7"),
    ]

    for question in list_simple:
        #print(question[0])
        reponse_reel = question[1]
        reponse_genere = ask_question("Simple", question[0])

        print('Reel reponse : ', reponse_reel)
        print('Reponse genere par le system : ', reponse_genere)
        print(50*'-')
    '''

    ## 5. Charger LLaMA
    #Cloner_LLaMA()
  
    ## 6. Charger le modèle sauvegardé
    #model_path = "./llama-2-7b-chat-quantized"
    #model, tokenizer = charger_model(model_path)

    # Vérifier si tout s'est bien passé
    #print(model)

    ## 7. Teste Llama_query
    #test_question = "Quelle est la capitale de la France ?"
    #response = ask_question("Simple", test_question)
    #print('response = ', response)

    #test_answer = query_llama(test_question, response)  # Utilisation de Mistral au lieu de DeepSeek

    #print("✅ LLaMa-2-7b-chat a généré une réponse :", test_answer)
    #print('Reponse du LLaMa : ', test_answer)

    ## 8. Teste Query
    #response = query("Quelle est la capitale de la France ?", "Simple")
    #print(response)

    #print("Torch détecte CUDA :", torch.cuda.is_available())  # Doit afficher True
    #print("Version de Torch :", torch.__version__)
    #print("Nom du GPU :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Aucun GPU détecté")

    #print(torch.cuda.is_available())  # Doit être True si CUDA est disponible
    #print(torch.cuda.get_device_name(0))  # Affiche le nom du GPU


    # Cloner LLama 3
    #Cloner_LLaMA_3()

    # Afficher l'utilisation de la mémoire GPU avant le modèle
    print('Afficher utilisation de la mémoire GPU : \n',afficher_mem_gpu())

    # Ton code pour charger et traiter le modèle ici...

    # Libérer la mémoire GPU
    print('Libérer la mémoire GPU : \n', liberer_mem_gpu())

    # Afficher l'utilisation de la mémoire GPU après avoir libéré
    print('Afficher utilisation de la memoire GPU apres avoir libere : \n',afficher_mem_gpu())

    # Charger  LLama 3
    #model_path = "./Llama-3-70B-Quantized"   
    #model, tokenizer = charger_model_llama3(model_path)

    ## 6. Charger le modèle sauvegardé
    model_path = "./llama-2-7b-chat-quantized"
    model, tokenizer = charger_model(model_path)

    # Vérifier si tout s'est bien passé
    print(model)



