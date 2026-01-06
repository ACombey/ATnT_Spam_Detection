# SMS Spam Classification — Deep Learning (Transfer Learning)

## Description

Ce projet a pour objectif de détecter automatiquement les SMS indésirables (spam) à partir d’un dataset public.  
Nous utilisons le **transfer learning** avec **DistilBERT**, un modèle pré-entraîné sur des milliards de textes, afin d’obtenir de très bonnes performances même avec un dataset limité.

Le notebook propose également une comparaison avec un modèle classique de **régression logistique**.

---

## Dataset

Le dataset utilisé contient des SMS étiquetés comme `ham` (message légitime) ou `spam` (message indésirable) :

- Source : [Spam SMS Dataset](https://full-stack-bigdata-datasets.s3.eu-west-3.amazonaws.com/Deep%20Learning/project/spam.csv)
- Colonnes principales :
  - `v1` : label (`ham` ou `spam`)
  - `v2` : contenu du message

---

## Démarche

1. **Prétraitement des données**
   - Nettoyage du texte : suppression de la ponctuation et mise en minuscules
   - Tokenisation et lemmatisation avec **spaCy**
   - Suppression des stopwords

2. **Vectorisation**
   - Pour le modèle classique : encodage des tokens avec un vocabulaire limité
   - Pour DistilBERT : tokenization via `DistilBertTokenizerFast`

3. **Modèles entraînés**
   - **Logistic Regression** sur TF-IDF
   - **DistilBERT fine-tuned** pour classification binaire

4. **Évaluation**
   - Métriques : Accuracy, Precision, Recall, F1-score
   - Visualisation possible via matrice de confusion

5. **Optimisation**
   - Ajustement du nombre d’epochs (3 → 5)
   - Possibilité de threshold tuning pour améliorer le rappel

---

## Résultats

### DistilBERT (5 epochs)

| Metric    | Score      |
|----------|-----------|
| Accuracy | 99.28 %   |
| Precision| 97.90 %   |
| Recall   | 96.55 %   |
| F1-score | 97.22 %   |

- Seulement 3 spams ont été manqués sur l’ensemble de test
- Excellent compromis entre précision et rappel

### Logistic Regression (TF-IDF)

| Metric    | Score      |
|----------|-----------|
| Accuracy | ~96 %     |
| Precision| ~95 %     |
| Recall   | ~93 %     |
| F1-score | ~94 %     |

> Le modèle basé sur DistilBERT surpasse nettement le modèle classique.

---

## Application métier

Ce modèle peut être intégré dans un **filtre SMS** ou un **système anti-spam** pour protéger les utilisateurs :  

- Bloquer la grande majorité des messages indésirables
- Préserver les messages légitimes (ham)
- Possibilité d’ajuster le seuil de classification selon les besoins métier

---

## Prérequis

- Python 3.8+
- Librairies Python :
  ```bash
  pip install torch transformers datasets pandas numpy scikit-learn spacy
  python -m spacy download en_core_web_sm
