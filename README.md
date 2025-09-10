# Email Classification avec CamemBERT


## 📌 Description

Système de classification automatique d'emails en français utilisant le modèle CamemBERT (BERT spécialisé français). Ce projet implémente un pipeline complet de Machine Learning, de la preprocessing des données jusqu'au déploiement via API REST et Docker.

## 🎯 Objectif

Classifier automatiquement des emails clients en 4 catégories :
- **Réclamation** : Demandes de remboursement, plaintes
- **Problème technique** : Dysfonctionnements, bugs
- **Question produit** : Informations sur les produits
- **Autre** : Demandes diverses

## 🚀 Résultats

- **Précision** : 82%
- **Dataset** : 1,751 emails français
- **Modèle** : CamemBERT + Dense layers
- **Déploiement** : API Flask + Docker

## 🛠️ Technologies

- **Machine Learning** : TensorFlow, Keras, scikit-learn
- **NLP** : HuggingFace Transformers, CamemBERT
- **Preprocessing** : NLTK, pandas, numpy
- **API** : Flask
- **Déploiement** : Docker
- **Langue** : Python 3.10

## 📊 Architecture

1. **Collecte** : Dataset d'emails français étiquetés
2. **Preprocessing** : Nettoyage, suppression stop words, normalisation
3. **Modélisation** : Fine-tuning CamemBERT sur données métier
4. **Entraînement** : TensorFlow/Keras avec validation
5. **API** : Endpoint Flask pour classification temps réel
6. **Déploiement** : Container Docker prêt cloud

## 🔧 Installation

### Prérequis
- Python 3.10+
- Docker (optionnel)

### Installation locale
```bash
git clone https://github.com/medsaa0/email-classifier-api.git
cd email-classifier-api
pip install -r requirements.txt
```

### Télécharger les modèles
```bash
# Les modèles sont volumineux, téléchargez-les depuis les releases
# ou entraînez le modèle avec le notebook fourni
```

## 🚀 Utilisation

### API Flask
```bash
python app.py
```

L'API sera disponible sur `http://localhost:8080`

### Test de l'API
```bash
curl -X POST http://localhost:8080/predict \
-H "Content-Type: application/json" \
-d '{"message": "Mon produit ne fonctionne plus, je veux un remboursement"}'
```

Réponse :
```json
{"category": "réclamation"}
```

### Docker
```bash
docker build -t email-classifier .
docker run -p 8080:8080 email-classifier
```

## 📁 Structure du projet

```
├── app.py                 # API Flask
├── requirements.txt       # Dépendances Python
├── Dockerfile            # Configuration Docker
├── Etapes du projet/      # Jupyter notebooks d'exploration
├── Data/                  # data
└── models/              # Modèles entraînés
```

## 🔍 Détails techniques

### Preprocessing
- Suppression des accents avec `unicodedata`
- Nettoyage regex (ponctuation, chiffres)
- Stop words français via NLTK
- Tokenisation avec CamembertTokenizer

### Modèle
- **Base** : CamemBERT pré-entraîné (`camembert-base`)
- **Architecture** : Transformer + Dense layers
- **Optimiseur** : Adam (lr=2e-5)
- **Loss** : Categorical crossentropy
- **Métriques** : Accuracy

### API
- **Framework** : Flask
- **Endpoint** : POST `/predict`
- **Format** : JSON
- **Port** : 8080

## 📈 Performance

| Métrique | Valeur |
|----------|---------|
| Précision globale | 82% |
| Échantillons test | 350 emails |
| Temps d'inférence | ~200ms |

## 🐳 Déploiement

Le projet est containerisé avec Docker pour un déploiement facile :

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["python", "app.py"]
```

## 📚 Notebooks

- `Etapes_Du_Projet.ipynb` : Pipeline complet d'entraînement
- Exploration des données, preprocessing, entraînement et évaluation

## 🤝 Contribution

1. Fork le projet
2. Créez une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add AmazingFeature'`)
4. Push sur la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

## 📄 Licence

Distribué sous licence MIT. Voir `LICENSE` pour plus d'informations.

## 👤 Auteur

**Saidi Mohammed**
- LinkedIn : https://www.linkedin.com/in/mohammed-saidi-3426912b2/
- Email : saidimohammed0301@gmail.com

## 🙏 Remerciements

- [CamemBERT](https://camembert-model.fr/) pour le modèle français
- [HuggingFace](https://huggingface.co/) pour les Transformers
- [TensorFlow](https://tensorflow.org/) pour le framework ML
