# Image de base Python légère
FROM python:3.10-slim

# Définit le répertoire de travail dans le conteneur
WORKDIR /app

# Copie les fichiers requirements.txt et installe les dépendances
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copie tout le reste du code
COPY . .

# Expose le port 8080
EXPOSE 8080

# Commande pour lancer ton app Flask sur le port 8080
CMD ["python", "app.py"]
