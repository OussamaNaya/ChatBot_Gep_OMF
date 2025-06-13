from transformers import AutoTokenizer, AutoModelForCausalLM


# Fonction pour charger le modèle et le tokenizer
def load_model():
    # Load model directly
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.3-70B-Instruct-bnb-4bit")
    model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.3-70B-Instruct-bnb-4bit")
    return model, tokenizer

# Fonction pour tester le modèle en générant une réponse
def test_model(model, tokenizer, prompt):
    response=''
    return response

# Main pour charger le modèle et effectuer un test
if __name__ == "__main__":
    # Charger le modèle
    model, tokenizer = load_model()

    print(model)
    
    # Exemple de prompt pour tester
    #prompt = "Explique-moi comment fonctionne le deep learning."
    
    # Générer une réponse avec le modèle
    #response = test_model(model, tokenizer, prompt)
    
    # Afficher la réponse générée
    #print("Réponse générée :")
    #print(response)
