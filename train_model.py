
import pandas as pd
import pickle
import os

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("Loading dataset...")

df = pd.read_csv("data/dataset.csv")

symptom_columns = [
    col for col in df.columns
    if col.startswith("Symptom_")
]

# Clean symptoms
for col in symptom_columns:
    df[col] = (
        df[col]
        .fillna("")
        .str.strip()
        .str.lower()
    )

print("Original dataset:", len(df))


# ==========================================
# 2. REMOVE DUPLICATE COMBINATIONS
# ==========================================

df["symptom_combination"] = df[symptom_columns].apply(
    lambda row: tuple(sorted(s for s in row if s != "")),
    axis=1
)

df_unique = df.drop_duplicates(
    subset="symptom_combination"
).copy()

print("Dataset without duplicates:", len(df_unique))


# ==========================================
# 3. PREPARE X AND Y
# ==========================================

X_symptoms = df_unique["symptom_combination"].tolist()

mlb = MultiLabelBinarizer()

X = mlb.fit_transform(X_symptoms)

y = df_unique["Disease"].str.strip()

print("Number of symptoms:", len(mlb.classes_))
print("X shape:", X.shape)
print("Number of diseases:", y.nunique())


# ==========================================
# 4. SPLIT DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 5. DEFINE MODELS
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}


# ==========================================
# 6. TRAIN AND EVALUATE MODELS
# ==========================================

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy:", round(accuracy, 4))

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )


# ==========================================
# 7. TRAIN FINAL MODEL
# ==========================================

print("\nTraining final Random Forest model...")

final_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Train on all unique observations
final_model.fit(X, y)


# ==========================================
# 8. SAVE MODEL AND ENCODER
# ==========================================

os.makedirs("models", exist_ok=True)

with open("models/fitted_model.pkl", "wb") as f:
    pickle.dump(final_model, f)

with open("models/symptom_encoder.pkl", "wb") as f:
    pickle.dump(mlb, f)


# ==========================================
# 9. FINISH
# ==========================================

print("\n================================")
print("MODEL TRAINING COMPLETED!")
print("================================")

print("Saved: models/fitted_model.pkl")
print("Saved: models/symptom_encoder.pkl")
