# Détection de Spam SMS avec DistilBERT

## Description du projet
Ce projet vise à détecter automatiquement les SMS **spam** ou **ham** (messages légitimes) grâce au **transfer learning** avec le modèle **DistilBERT**.  
Le modèle est entraîné sur un dataset public de SMS, après un prétraitement minimal (nettoyage et tokenisation).

L'objectif est de construire un modèle efficace avec peu de données, capable d'identifier la majorité des spams tout en limitant les faux positifs.


## Résultats obtenus

### Tableau synthétique des métriques sur le test set

| Métrique      | Valeur  |
|---------------|---------|
| Accuracy      | 0.993   |
| Precision     | 0.9726  |
| Recall        | 0.9530  |
| F1-score      | 0.9627  |

- **Accuracy** : 99,3 % → très bon classement global.  
- **Precision** : 97,3 % → peu de faux positifs (ham classé spam).  
- **Recall** : 95,3 % → majorité des spams détectés.  
- **F1-score** : 96,3 % → bon compromis entre précision et rappel.  

### Commentaires
- La perte décroît rapidement et l’accuracy atteint presque 100 % dès la troisième époque sur le train set.  
- Quelques spams restent non détectés (3 à 6 sur 145), ce qui pourrait être amélioré en ajustant le seuil de classification ou en combinant avec des règles simples.  

---

## Conclusion
- Le transfert de connaissances via DistilBERT permet d’obtenir un modèle performant même avec un dataset de taille modeste.  
- Les résultats sont suffisants pour filtrer la grande majorité des spams tout en préservant les messages légitimes.  
- Des améliorations possibles incluent :  
  - Ajuster le seuil de détection des spams.  
  - Tester d’autres modèles (BERT, ALBERT, RoBERTa).  
  - Ajouter des règles supplémentaires pour capturer les rares spams manqués.  

---

## Installation et exécution

```bash
pip install torch transformers datasets scikit-learn matplotlib
