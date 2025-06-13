from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import Service_QA_RAG  # Assure-toi que ce module existe et est correctement importé
import time

'''
FastAPI, c'est un framework web en Python utilisé pour créer des API rapides et performantes. 
Il est basé sur Starlette pour la gestion des requêtes et Pydantic pour la validation des données.
'''

app = FastAPI()

# 🚀 Ajout du middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔥 Remplace par ["http://localhost:5500"] si nécessaire
    allow_credentials=True,
    allow_methods=["*"],  # ✅ Autorise toutes les méthodes (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # ✅ Autorise tous les headers
)

# model, tokenizer = Service_QA_RAG.Load_LLaMA2_From_HuggingFace()
# if model != None and tokenizer != None:
#     print("Le model charger avec succees!")
# else:
#     print("Problem dans le chargement du model !")

class QueryRequest(BaseModel):
    question: str
    role: str

'''
@app.post("/query")
async def query_rag(request: QueryRequest):
    print(f"Requête reçue : {request.role} - {request.question}")  # Log pour debug
    #response = Service_QA_RAG.ask_question(request.role, request.question)
    response = Service_QA_RAG.query(request.question, request.role)
    print(f"Réponse envoyée : {response}")  # Log pour debug
    return {"answer": response}
'''

@app.get("/")
async def hello():
    return "Hello !"

@app.post("/query")
async def query_rag(request: QueryRequest):
    start_time = time.time()  # Démarrage du chronomètre

    print(f"Requête reçue : {request.role} - {request.question}")  # Log pour debug

    # Traitement de la requête
    #response = Service_QA_RAG.query(model, tokenizer, request.question, request.role)
    response = Service_QA_RAG.query(request.question, request.role)

    end_time = time.time()  # Fin du chronomètre
    duration = end_time - start_time  # Durée d'exécution

    print(f"Réponse envoyée : {response}")  # Log pour debug
    print(f"Durée d'exécution : {duration:.4f} secondes")  # Affichage de la durée

    return {"answer": response, "execution_time": f"{duration:.4f} secondes"}

# uvicorn app:app --reload