# 🚀 Guide d'Installation et d'Exécution Locale

Ce guide explique comment configurer et lancer l'interface utilisateur (Gradio) du projet RAG localement.

## 1. Prérequis

Assurez-vous d'avoir installé Python (>= 3.10) et Git sur votre machine.

## 2. Cloner le dépôt et installer les dépendances

`ash
git clone https://github.com/VOTRE_PROFIL/VOTRE_DEPOT.git
cd VOTRE_DEPOT
pip install -r requirements.txt
`

## 3. Configuration des variables d'environnement

L'application a besoin d'accéder à l'API Mistral (pour la génération) et potentiellement HuggingFace (pour certains modèles).
Créez un fichier .env à la racine du projet ou exportez les variables dans votre terminal :

`ash
# Sur Windows (PowerShell)
$env:MISTRAL_API_KEY="votre_cle_api_mistral"

# Sur Linux / macOS
export MISTRAL_API_KEY="votre_cle_api_mistral"
`

## 4. Préparation de l'Index (Vectorstore)

L'interface a besoin de l'index vectoriel pré-calculé pour fonctionner.
Le dossier ag_project/data/ est ignoré par Git en raison de sa taille. Vous avez deux options :

**Option A (Recommandée) : Télécharger l'index pré-calculé**
1. Téléchargez l'archive ectorstore.zip depuis les *Releases* GitHub du projet.
2. Extrayez son contenu dans ag_project/data/processed/vectorstore/. Vous devriez avoir les fichiers aiss_index.bin et chunks_metadata.json.

**Option B : Reconstruire l'index localement**
Si vous souhaitez tout reconstruire à partir de zéro, exécutez le pipeline complet :
`ash
python etape1_collecte.py
python etape2_nettoyage.py
python etape4_indexation.py
`
*(Attention, cette opération prend du temps et nécessite de l'espace disque pour télécharger la documentation complète).*

## 5. Lancer l'interface

Une fois l'index en place et la clé API configurée, lancez simplement :

`ash
python ui.py
`

L'application web démarrera localement. Ouvrez l'URL affichée dans votre navigateur (généralement http://127.0.0.1:7860).
