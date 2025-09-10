from flask import Flask, request, jsonify
import pickle
import numpy as np
import tensorflow as tf
from transformers import TFCamembertModel, CamembertTokenizer
from tensorflow.keras.utils import register_keras_serializable

# Charger CamemBERT
camembert = TFCamembertModel.from_pretrained("camembert-base")
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

@register_keras_serializable()
def extract_cls(inputs):
    input_ids, attention_mask = inputs
    outputs = camembert(input_ids=input_ids, attention_mask=attention_mask)
    cls_token = outputs.last_hidden_state[:, 0, :]
    return cls_token

# Chargement du modèle entraîné
model = tf.keras.models.load_model("email_classifier_model.keras", custom_objects={"extract_cls": extract_cls})

# Chargement du label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Création de l'application Flask
app = Flask(__name__)
MAX_LEN = 30

# Fonction de prétraitement
def preprocess_text(text):
    import re, unicodedata
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('french'))

    def enlever_accents(texte):
        return ''.join(c for c in unicodedata.normalize('NFD', texte) if unicodedata.category(c) != 'Mn')

    texte = text.lower()
    texte = enlever_accents(texte)
    texte = re.sub(r'\d+', '', texte)
    texte = re.sub(r'[^\w\s]', '', texte)
    texte = re.sub(r'\s+', ' ', texte).strip()
    mots = texte.split()
    mots = [mot for mot in mots if mot not in stop_words]
    return " ".join(mots)

# Route d'accueil
@app.route('/')
def home():
    return 'Bienvenue sur l’API de classification des e-mails.'

# Route de prédiction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('message', '')
    cleaned_text = preprocess_text(text)
    encoding = tokenizer(
        [cleaned_text],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="tf"
    )
    preds = model.predict({
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"]
    })
    class_idx = np.argmax(preds, axis=1)[0]
    category = label_encoder.inverse_transform([class_idx])[0]
    return jsonify({"category": category})

# Lancement du serveur
import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  # Prend la variable PORT sinon 8080 par défaut
    app.run(host='0.0.0.0', port=port)
