# MissingDataLab  - Simulation et imputation de données manquantes

MissingDataLab est un outil Python conçu pour simuler des données manquantes selon différents mécanismes (MCAR, MAR, MNAR) et appliquer diverses méthodes d'imputation pour les compléter. 
Le projet permet également d'évaluer la qualité des imputations à travers des métriques et des visualisations détaillées.

## Fonctionnalités

- Simulation de données manquantes selon les mécanismes :
  - MCAR (Missing Completely at Random)
  - MAR (Missing at Random)
  - MNAR (Missing Not at Random)

- Méthodes d'imputation supportées :
  - Imputations simples : mean, median, mode, constante
  - Imputations avancées : KNN, SoftImputer, ICE, ACP, MissForest

- Évaluation de l'imputation :
  - Calcul d'indicateurs de performance (MSE, accuracy)
  - Mesures d'efficacité (temps d'exécution, utilisation mémoire)
  - Analyse de la variance des imputations

- Visualisation :
  - Graphiques détaillés des résultats d'imputation
  - Comparaison des méthodes d'imputation selon divers critères

## Structure du projet

```tree
MissingDataLab/
├── config/
│   └── config.yaml               # Fichier de configuration du projet
├── data/
│   ├── original/                 # Données originales (sans valeurs manquantes)
│   │   └── data_original.csv
│   └── result/                   # Résultats (CSV et graphiques)
│       ├── result_graph.png
│       ├── result_score.csv
│       └── variance_originale.csv
├── src/
│   ├── ampute/                   # Scripts pour générer les données manquantes
│   │   ├── ampute_utils.py
│   │   ├── generate.py
│   │   └── math_utils.py
│   ├── impute/                   # Scripts d'imputation
│   │   ├── advance_impute.py
│   │   ├── generate.py
│   │   └── simple_impute.py
│   └── utils/                    # Outils et fonctions auxiliaires
│       ├── data_utils.py
│       ├── graph_utils.py
│       ├── metrics_utils.py
│       └── path_utils.py
├── main.py                       # Script principal pour exécuter le projet
└── requirements.txt              # Fichier de dépendances venv
└── environment.yml               # Fichier de dépendances conda
└── README.md                     # README Markdown
└── README.txt                    # Ce fichier
```

## Installation

### 1. Cloner le dépôt :

```bash
   git clone https://github.com/your-username/MissingDataLab.git
   cd MissingDataLab
```

### 2. Créer et activer un environnement virtuel :

   #### a. Avec venv

```bash
      python -m venv venv
      source venv/bin/activate  # Sur Linux/Mac
      .\venv\Scripts\activate   # Sur Windows
      pip install -r requirements.txt
```

   #### b. Avec Conda

```bash
      conda env create -f environment.yml
      conda activate missingdatalab
```


## Configuration

Le fichier `config/config.yaml` permet de personnaliser les paramètres du projet :

- `data_origin` : chemin vers les données d'entrée
- `missing_data` : paramètres de génération des données manquantes (proportions, mécanismes, colonnes)
- `imputation_methods` : méthodes d'imputation pour chaque colonne
- `evaluation` : nombre de répétitions pour l'amputation et l'imputation

## Utilisation

### 1. Lancer le script principal

```bash
   python main.py
```

### 2. Résultats générés

   - CSV des résultats dans `data/result/`
   - Graphiques comparatifs enregistrés sous `data/result/results_graph.png`

### Métriques et évaluation

Les performances d'imputation sont évaluées à l'aide de :

- Accuracy pour les variables catégorielles
- MSE (Mean Squared Error) pour les variables numériques
- Temps d'exécution et utilisation mémoire
- Analyse de la variance des valeurs imputées

### Visualisation des résultats

Les graphiques générés montrent l'impact des différentes méthodes d'imputation selon :

- Les mécanismes des données manquantes (MCAR, MAR, MNAR)
- Les proportions de valeurs manquantes (ex : 10%, 50%, 80%)
- Les performances des méthodes (Accuracy/MSE, temps, mémoire, variance)

# Remerciements

Merci à tous les contributeurs et utilisateurs pour leur soutien et leurs retours.