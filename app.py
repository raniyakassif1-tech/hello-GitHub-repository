
import os
import json
import pickle
import random

import pandas as pd
import numpy as np
import nltk

from flask import Flask, render_template, request, jsonify
from nnet import NeuralNet
import torch
from nltk_utils import bag_of_words


# ==========================================
# 1. INITIALIZATION
# ==========================================

app = Flask(__name__)

device = torch.device("cpu")

# ==========================================
# 2. LOAD NLP MODEL (SYMPTOM RECOGNITION)
# ==========================================

FILE = "models/data.pth"

model_data = torch.load(FILE, map_location=device, weights_only=False)

input_size = model_data["input_size"]
hidden_size = model_data["hidden_size"]
output_size = model_data["output_size"]

all_words = model_data["all_words"]
tags = model_data["tags"]
model_state = model_data["model_state"]

nlp_model = NeuralNet(
    input_size,
    hidden_size,
    output_size
).to(device)

nlp_model.load_state_dict(model_state)
nlp_model.eval()


# ==========================================
# 3. LOAD NEW DISEASE PREDICTION MODEL
# ==========================================

with open("models/fitted_model.pkl", "rb") as f:
    prediction_model = pickle.load(f)

with open("models/symptom_encoder.pkl", "rb") as f:
    symptom_encoder = pickle.load(f)

symptoms_list = list(symptom_encoder.classes_)


# ==========================================
# 4. LOAD MEDICAL DATA
# ==========================================

diseases_description = pd.read_csv(
    "data/symptom_Description.csv"
)

diseases_description["Disease"] = (
    diseases_description["Disease"]
    .str.strip()
    .str.lower()
)

disease_precaution = pd.read_csv(
    "data/symptom_precaution.csv"
)

disease_precaution["Disease"] = (
    disease_precaution["Disease"]
    .str.strip()
    .str.lower()
)

symptom_severity = pd.read_csv(
    "data/Symptom-severity.csv"
)

symptom_severity = symptom_severity.map(
    lambda s: s.lower().strip().replace(" ", "")
    if isinstance(s, str) else s
)

# ==========================================
# 5. USER SYMPTOMS
# ==========================================

user_symptoms = set()


# ==========================================
# 6. NLP SYMPTOM RECOGNITION
# ==========================================

def get_symptom(sentence):

    sentence = nltk.word_tokenize(sentence)

    X = bag_of_words(sentence, all_words)

    X = X.reshape(1, X.shape[0])

    X = torch.from_numpy(X)

    output = nlp_model(X)

    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)

    prob = probs[0][predicted.item()].item()

    return tag, prob


# ==========================================
# 7. HOME PAGE
# ==========================================

@app.route("/")
def index():

    user_symptoms.clear()

    with open(
        "static/assets/files/ds_symptoms.txt",
        "r",
        encoding="utf-8"
    ) as file:

        all_symptoms = file.readlines()

    data = [
        s.replace("'", "")
         .replace("_", " ")
         .replace(",\n", "")
        for s in all_symptoms
    ]

    return render_template(
        "index.html",
        data=json.dumps(data)
    )


# ==========================================
# 8. CHATBOT PREDICTION
# ==========================================

@app.route("/symptom", methods=["GET", "POST"])
def predict_symptom():

    global user_symptoms

    try:

        sentence = request.json["sentence"]

        if sentence.replace(".", "").replace("!", "").lower().strip() == "done":

            if not user_symptoms:

                response_sentence = (
                    "Please enter some symptoms first."
                )

            else:

                # Convert symptoms to binary vector
                input_vector = symptom_encoder.transform(
                    [list(user_symptoms)]
                )

                # Predict disease
                disease = prediction_model.predict(
                    input_vector
                )[0]

                disease_key = disease.strip().lower()

                # Get description
                description_row = diseases_description.loc[
                    diseases_description["Disease"] == disease_key,
                    "Description"
                ]

                description = (
                    description_row.iloc[0]
                    if not description_row.empty
                    else "Description unavailable."
                )

                # Get precautions
                precaution_row = disease_precaution[
                    disease_precaution["Disease"] == disease_key
                ]

                if not precaution_row.empty:

                    row = precaution_row.iloc[0]

                    precautions = ", ".join(
                        str(row[col])
                        for col in [
                            "Precaution_1",
                            "Precaution_2",
                            "Precaution_3",
                            "Precaution_4"
                        ]
                        if col in row.index and pd.notna(row[col])
                    )

                else:
                    precautions = "Please consult a healthcare professional."

                response_sentence = (
                    f"Possible condition suggested by the model: {disease}."
                    f"<br><br><i>Description: {description}</i>"
                    f"<br><br><b>Precautions: {precautions}</b>"
                    "<br><br>This is an educational prediction, not a medical diagnosis. "
                    "Please consult a qualified healthcare professional."
                )

                user_symptoms.clear()

        else:

            symptom, prob = get_symptom(sentence)

            if prob > 0.5:

                response_sentence = (
                    f"I'm {(prob * 100):.2f}% sure this is {symptom}."
                )

                user_symptoms.add(
                    symptom.strip().lower()
                )

            else:

                response_sentence = (
                    "I'm sorry, but I don't understand you."
                )

        return jsonify(response_sentence.replace("_", " "))

    except Exception as e:

        print("Chatbot error:", str(e))

        return jsonify(
            "An error occurred. Please try again."
        ), 500


# ==========================================
# 9. RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(debug=True)
