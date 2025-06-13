from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import BitsAndBytesConfig

# Vérification de la version de torch et de la disponibilité du GPU
print(torch.__version__)
print(torch.cuda.is_available())  # Vérifie si un GPU est disponible

# Charger le modèle et le tokenizer
model_path = "./Llama-3-70B-Standard"
print("Chargement du tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Vérification de la mémoire GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Chargement du modèle sur {device.upper()}...")

# Configuration de la quantification en 4-bit avec offloading vers CPU
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Pour utiliser la quantification en 4-bit
    bnb_4bit_compute_dtype=torch.float16,  # Utilisation de FP16 pour la quantification
    llm_int8_enable_fp32_cpu_offload=True,  # Activer l'offloading vers le CPU pour la gestion de la mémoire
    bnb_4bit_quant_type="nf4"  # Utilisation de "nf4" pour 4-bit (ou "fp4" pour 8-bit)
)

# Charger le modèle avec la quantification en 4-bit et offloading
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",  # Auto-allocation sur CPU ou GPU
    quantization_config=bnb_config  # Appliquer la quantification
)

# Exemple de texte à générer
input_text = "Once upon a time,"
print(f"Texte d'entrée : {input_text}")

# Tokenizer l'entrée
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
print(f"Tokens générés : {inputs['input_ids']}")

# Déplacer les tensors sur le bon appareil (GPU/CPU)
inputs = {key: value.to(device) for key, value in inputs.items()}

# Ajouter le pad_token_id si ce n'est pas déjà fait
model.config.pad_token_id = tokenizer.eos_token_id  # Définir explicitement le pad_token_id comme eos_token_id

# Générer une réponse
with torch.no_grad():
    print("Génération en cours...")
    outputs = model.generate(inputs["input_ids"], max_length=50, attention_mask=inputs["attention_mask"], num_beams=1)

# Décoder la sortie
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Texte généré : {generated_text}")
