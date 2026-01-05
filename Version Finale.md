# NOM:BLALET SALIMA
# Groupe:CAC 1
# TITRE: Prédiction de l’état de santé à partir de données physiologiques à l’aide du machine learning
<img src="PHOTO_BLALET.jpg" style="height:464px;margin-right:432px"/>

# Rapport de projet : Classification d’une maladie à partir de paramètres vitaux

## 1. Contexte métier et objectif
# 1.1.Descriptif de l'analyse:
Dans le domaine médical, l’évaluation de l’état d’un patient repose souvent sur l’interprétation de signaux vitaux (température, pouls, saturation en oxygène, glycémie, tension artérielle).  Cette interprétation peut être complexe et source d’erreurs, surtout lorsque le volume de patients augmente ou que les symptômes sont peu spécifiques.[2][3][1]

# L’objectif de ce projet :
est de construire un modèle de classification supervisée qui prédise automatiquement si un patient est malade ou sain à partir de ses paramètres physiologiques, en accordant une importance particulière au rappel de la classe malade.[3][1][2]

***

## 2. Chargement des données et description du jeu de données

### 2.1 Import des bibliothèques

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

plt.style.use('ggplot')
sns.set_theme(style="whitegrid")
```


### 2.2 Chargement du dataset

```python
data = pd.read_csv('/content/maladie_observations.csv')
data.sample(5)
data.shape
```


Le dataset contient 5 725 lignes et 6 colonnes : `temperature`, `pouls`, `oxygene`, `glycemie`, `tension`, `label`.[2][3]

### 2.3 Distribution de la variable cible

```python
data.groupby('label')['label'].value_counts()
```


La classe 0 (sain) compte 2 551 individus et la classe 1 (malade) 3 174 individus.[2]

***

## 3. Analyse exploratoire des données (EDA)

### 3.1 Valeurs manquantes et premiers profils

```python
data.isna().sum()
data.describe()
data.temperature.describe()
```


On observe des valeurs manquantes dans `temperature`, `pouls` et `oxygene`, et un maximum aberrant de température autour de 522,5 °C.[3][2]

### 3.2 Correction des outliers de température

```python
# Médiane de température pour remplacer les valeurs aberrantes
median_temp = data[data['temperature'] < 100]['temperature'].median()

# Remplacement des valeurs supérieures à 100 par la médiane
data['temperature'] = data['temperature'].apply(
    lambda x: median_temp if x > 100 else x
)

# Vérification après correction
data.temperature.describe()
```


Après correction, la température max revient vers 40 °C, avec une moyenne et un écart‑type cohérents.[2][3]

### 3.3 Imputation simple (version exploratoire sur température)

```python
mean_temp = data['temperature'].mean()
data['temperature'] = data['temperature'].fillna(mean_temp)
data.temperature.describe()
```


Cette étape montre l’effet d’une imputation par la moyenne sur la distribution de `temperature`.[2]

### 3.4 Corrélations entre variables

```python
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr().abs(), cmap='magma', annot=True)
plt.title('Matrice de corrélation (valeurs absolues)')
plt.show()
```


La heatmap met en évidence des corrélations modérées mais pas de colinéarité extrême.[2]

***

## 4. Nettoyage et préparation des données pour le modèle

### 4.1 Séparation des variables explicatives et de la cible

```python
X = data.drop('label', axis=1)
Y = data['label']
```


### 4.2 Split train/test

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=75
)
```


### 4.3 Imputation des valeurs manquantes (bonne pratique sans data leakage)

```python
# Colonnes avec valeurs manquantes
missing_cols = ['temperature', 'pouls', 'oxygene']

imputer = SimpleImputer(strategy='mean')

# Fit uniquement sur le train
imputer.fit(X_train[missing_cols])

# Transformation de train et test
X_train[missing_cols] = imputer.transform(X_train[missing_cols])
X_test[missing_cols] = imputer.transform(X_test[missing_cols])
```


Cette approche respecte la règle de ne pas utiliser le test set pour calculer les paramètres d’imputation.[1][2]

***

## 5. Construction du modèle de classification

### 5.1 Entraînement d’une régression logistique

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
```


Le modèle est adapté à une cible binaire (`label`) et fournit des probabilités de maladie.[1][3][2]

### 5.2 Prédictions sur le jeu de test

```python
y_pred = model.predict(X_test)
```


***

## 6. Évaluation des performances

### 6.1 Accuracy

```python
acc = accuracy_score(Y_test, y_pred)
print("Accuracy du modèle :", acc)
```


L’accuracy obtenue est proche de 0,996, ce qui traduit un taux d’erreur très faible.[2]

### 6.2 Matrice de confusion

```python
cm = confusion_matrix(Y_test, y_pred)
print(cm)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='summer')
plt.title('Matrice de confusion')
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')
plt.show()
```


La matrice de confusion permet d’identifier explicitement les faux positifs et faux négatifs.[1][2]

### 6.3 Rapport de classification (précision, rappel, F1)

```python
print(classification_report(Y_test, y_pred, target_names=['sain', 'malade']))
```


Ce rapport fournit précision, rappel et F1‑score pour chaque classe, conformément aux recommandations de la correction en contexte médical.[1]

***

## 7. Visualisations complémentaires

### 7.1 Température moyenne par classe

```python
plt.figure(figsize=(5, 4))
sns.barplot(data=data, x='label', y='temperature', palette=['red', 'blue'])
plt.title('Température moyenne selon le label')
plt.xlabel('label (0 = sain, 1 = malade)')
plt.ylabel('Température')
plt.show()
```


### 7.2 Oxygène moyen par classe

```python
plt.figure(figsize=(5, 4))
sns.barplot(data=data, x='label', y='oxygene', palette=['violet', 'cyan'])
plt.title('Saturation en oxygène moyenne selon le label')
plt.xlabel('label (0 = sain, 1 = malade)')
plt.ylabel('Oxygène')
plt.show()
```


Ces graphiques mettent en évidence des différences de profils vitaux entre patients sains et malades.[3][2]

***

## 8. Conclusion

Ce projet montre comment passer d’un tableau de mesures physiologiques bruité (valeurs manquantes et outliers) à un modèle de classification performant et interprétable.  Les différentes étapes – EDA, nettoyage, imputation, split train/test, entraînement et évaluation – sont implémentées en Python en respectant les bonnes pratiques méthodologiques, en particulier la gestion du data leakage et l’usage d’indicateurs adaptés au risque clinique.[1][3][2]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/128828236/c52bfd1a-4be0-4942-9476-77b5fa4e24d0/Correction-Projet-1.md)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/128828236/f01fe3c2-0ae9-45de-99f9-3b64c5be35c0/CODE_PYTON-1.ipynb)
[3](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/128828236/0e895021-caf8-4f76-9d96-b5f7c734fa8f/maladie_observations.csv)
