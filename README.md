# 🖼️ Multimodal Image Search Engine

Un moteur de recherche d'images multimodal utilisant Elasticsearch et l'apprentissage profond pour effectuer des recherches par similarité visuelle et par tags textuels.

## 📋 Table des matières

- [Aperçu](#aperçu)
- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation](#utilisation)
- [Structure du projet](#structure-du-projet)
- [Technologies utilisées](#technologies-utilisées)
- [Fonctionnement technique](#fonctionnement-technique)

## 🎯 Aperçu

Ce projet implémente un moteur de recherche d'images avancé qui combine:
- **Recherche vectorielle**: Utilise VGG16 pour extraire des embeddings d'images
- **Recherche textuelle**: Filtrage par tags avec correspondance exacte et floue
- **Recherche hybride**: Combinaison des deux approches pour des résultats optimaux

## ✨ Fonctionnalités

- 🔍 **Recherche par image**: Téléchargez une image pour trouver des images visuellement similaires
- 🏷️ **Filtrage par tags**: Recherche d'images par mots-clés ou tags
- 🎭 **Mode hybride**: Combinez recherche visuelle et textuelle
- ⚡ **Performance optimisée**: Utilisation d'Elasticsearch pour des recherches rapides
- 📊 **Interface intuitive**: Application Streamlit simple et élégante
- 🎚️ **Résultats configurables**: Ajustez le nombre de résultats affichés (Top K)

## 🏗️ Architecture

```
┌─────────────────┐
│   Interface     │
│   Streamlit     │
└────────┬────────┘
         │
    ┌────▼────┐
    │  app.py │
    └────┬────┘
         │
    ┌────┴──────────────────┐
    │                       │
┌───▼──────────┐    ┌──────▼──────────┐
│FeatureExtract│    │ ElasticSearch   │
│   (VGG16)    │    │     Client      │
└──────────────┘    └─────────────────┘
         │                  │
         └──────┬───────────┘
                │
        ┌───────▼────────┐
        │  Elasticsearch │
        │   Index Store  │
        └────────────────┘
```

## 📦 Prérequis

- Python 3.8+
- Elasticsearch 7.x ou 8.x
- Au moins 4 GB de RAM
- GPU recommandé (optionnel, pour l'extraction de features)

## 🚀 Installation

### 1. Cloner le repository

```bash
git clone https://github.com/votre-username/Multimodal-Image-Search-Engine.git
cd Multimodal-Image-Search-Engine
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

**Dépendances principales:**
```
streamlit
tensorflow
keras
elasticsearch
Pillow
numpy
```

### 4. Installer et démarrer Elasticsearch

**Option A - Docker (recommandé):**
```bash
docker run -d --name elasticsearch \
  -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  elasticsearch:8.11.0
```

**Option B - Installation locale:**
Téléchargez Elasticsearch depuis [elastic.co](https://www.elastic.co/downloads/elasticsearch)

## ⚙️ Configuration

Modifiez le fichier `app/config.py` selon votre environnement:

```python
# Répertoire contenant vos images
IMAGES_DIR = r"D:\images"  # Changez ce chemin

# Configuration Elasticsearch
INDEX_NAME = 'my-image-embeddings'
ES_HOSTS = [{'host': 'localhost', 'port': 9200, 'scheme': 'http'}]

# Paramètres de recherche
TOP_N_DEFAULT = 5
```

### Création de l'index Elasticsearch

Avant la première utilisation, créez l'index avec le mapping approprié:

```python
from elasticsearch import Elasticsearch

es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

mapping = {
    "mappings": {
        "properties": {
            "image_id": {"type": "keyword"},
            "image_name": {"type": "text"},
            "relative_path": {"type": "keyword"},
            "tags": {"type": "text"},
            "image_embedding": {
                "type": "dense_vector",
                "dims": 4096  # Dimension des embeddings VGG16 fc1
            }
        }
    }
}

es.indices.create(index='my-image-embeddings', body=mapping)
```

## 🎮 Utilisation

### Démarrer l'application

```bash
cd app
streamlit run app.py
```

L'application sera accessible à l'adresse: `http://localhost:8501`

### Modes de recherche

#### 1. Recherche par image seule
1. Téléchargez une image via l'interface
2. Cliquez sur "Lancer la recherche"
3. Les images similaires s'affichent par ordre de pertinence

#### 2. Recherche par tags seule
1. Entrez des mots-clés dans le champ "Filtre de Tags/Texte"
2. Cliquez sur "Lancer la recherche"
3. Les images correspondantes aux tags s'affichent

#### 3. Recherche hybride
1. Téléchargez une image ET entrez des tags
2. La recherche combine similarité visuelle et filtrage textuel
3. Résultats optimaux pour des recherches spécifiques

## 📁 Structure du projet

```
Multimodal-Image-Search-Engine/
│
├── app/
│   ├── app.py                 # Application Streamlit principale
│   ├── config.py             # Configuration globale
│   ├── elastic_client.py     # Client et requêtes Elasticsearch
│   └── feature_extractor.py  # Extraction de features VGG16
│
├── README.md                 # Ce fichier
└── requirements.txt          # Dépendances Python
```

## 🛠️ Technologies utilisées

| Technologie | Usage |
|------------|-------|
| **Streamlit** | Interface utilisateur web |
| **TensorFlow/Keras** | Deep learning et VGG16 |
| **Elasticsearch** | Stockage et recherche vectorielle |
| **VGG16** | Extraction de features visuelles |
| **NumPy** | Calculs numériques |
| **Pillow** | Traitement d'images |

## 🔬 Fonctionnement technique

### Extraction de features

- **Modèle**: VGG16 pré-entraîné sur ImageNet
- **Couche**: fc1 (4096 dimensions)
- **Normalisation**: Vecteurs normalisés L2

```python
# Preprocessing
Image → Resize(224x224) → RGB → Normalize → VGG16(fc1) → L2 Norm
```

### Calcul de similarité

La similarité cosinus est utilisée pour comparer les embeddings:

```
Score = cosineSimilarity(query_vector, image_embedding) + 1.0
```

Le `+1.0` transforme le score de [-1, 1] à [0, 2] pour faciliter l'interprétation.

### Stratégie de recherche

1. **Vectorielle pure**: Script score avec similarité cosinus
2. **Textuelle pure**: Match + Fuzzy search sur les tags
3. **Hybride**: Script score avec filtre textuel préalable

## 📝 Notes importantes

- Les images doivent être indexées dans Elasticsearch avant la recherche
- Le chemin `IMAGES_DIR` doit contenir les images référencées dans l'index
- La première exécution télécharge les poids VGG16 (~500MB)
- Les embeddings sont mis en cache pour améliorer les performances
