import pandas as pd

# Charger le dataset
df = pd.read_csv("data/dataset.csv")

# Récupérer les colonnes des symptômes
symptom_columns = [col for col in df.columns if col.startswith("Symptom_")]

# Nettoyer les symptômes
for col in symptom_columns:
    df[col] = df[col].fillna("").str.strip().str.lower()

# Afficher les maladies
print("Nombre de maladies :", df["Disease"].nunique())

# Afficher quelques symptômes
all_symptoms = set()

for col in symptom_columns:
    all_symptoms.update(df[col].unique())

all_symptoms.discard("")

print("Nombre total de symptômes :", len(all_symptoms))
print("\nPremiers symptômes :")
print(sorted(all_symptoms)[:20])

# Vérifier les valeurs manquantes restantes
print("\nValeurs manquantes :")
print(df[symptom_columns].isnull().sum().sum())