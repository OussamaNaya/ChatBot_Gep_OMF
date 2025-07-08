from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch


# Fonction pour charger le modèle et le tokenizer depuis Hugging Face
def load_model():
    # Connexion à Hugging Face
    login()

    # Charger le tokenizer et le modèle
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf")

    # Définir un chemin pour la sauvegarde locale
    save_path = "./Llama-2-13b-chat-hf"

    # Sauvegarder le modèle et le tokenizer localement
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"✅ Modèle et tokenizer sauvegardés localement dans : {save_path}")

    return model, tokenizer

# Fonction pour charger le modèle et le tokenizer depuis un chemin local
def load_model_from_local():
    # Chemin où le modèle et le tokenizer ont été sauvegardés
    model_path = "./Llama-2-13b-chat-hf"
    
    # Charger le modèle et le tokenizer depuis le chemin local
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    print(f"✅ Modèle et tokenizer chargés localement depuis : {model_path}")
    
    return model, tokenizer

# Fonction pour charger le modèle quantifié en 4 bits et le tokenizer depuis un chemin local
def load_model_from_local_4bit():
    # Chemin où le modèle quantifié en 4 bits et le tokenizer ont été sauvegardés
    model_path = "./unsloth-llama-2-13b-bnb-4bit"
    
    # Charger le tokenizer depuis le chemin local
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Charger le modèle quantifié en 4 bits depuis le chemin local
    # Utilisation de 'load_in_4bit' pour spécifier l'utilisation de la quantification 4 bits
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,  # Réduit l'usage mémoire
        torch_dtype=torch.float16,  # Assurez-vous que le modèle utilise la précision FP16
        device_map="auto"  # Charge le modèle automatiquement sur le GPU si disponible
    )

    print(f"✅ Modèle quantifié en 4 bits et tokenizer chargés localement depuis : {model_path}")
    
    return model, tokenizer

# --- Tester le modèle chat ---
def test_model(model, tokenizer):
    # Choisir le périphérique (GPU si disponible, sinon CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Envoyer le modèle sur le périphérique choisi (GPU ou CPU)
    model.to(device)

    # Prompt structuré dans le format LLaMA 2 Chat
    prompt = """<s>[INST] Quels sont les avantages de l'intelligence artificielle dans le domaine médical ? [/INST]"""

    # Tokenisation
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Génération
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id
    )

    # Décodage
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Affichage de la réponse générée
    print("\n=== Réponse du modèle ===")
    print(response)

# Main pour charger le modèle et tester
if __name__ == "__main__":
    # Charger le modèle et le tokenizer depuis le répertoire local
    #model, tokenizer = load_model_from_local()
    model, tokenizer = load_model_from_local_4bit()

    # Tester le modèle en générant une réponse
    test_model(model, tokenizer)