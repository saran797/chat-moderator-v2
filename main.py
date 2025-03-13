import torch
import joblib
import os
import numpy as np
from flask import Flask, request, jsonify
from transformers import AutoTokenizer
from model import BiLSTMModel  # Import your BiLSTM model

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define model parameters
INPUT_DIM = 768  # RoBERTa embeddings size
HIDDEN_DIM = 128  # LSTM hidden size
NUM_LAYERS = 2  # Number of BiLSTM layers
OUTPUT_DIM = 256  # Output feature size

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Initialize model
lstm_model = BiLSTMModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM)
lstm_model.load_state_dict(torch.load(f"{MODEL_DIR}/bilstm_model_v2.pth", map_location=device))
lstm_model.to(device)
lstm_model.eval()  # Set to evaluation mode

# Load Random Forest Model
rf_model = joblib.load(f"models/random_forest_model.pkl")

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Function to extract features
def get_features(text):
    tokens = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    tokens = {key: value.to(device) for key, value in tokens.items()}

    with torch.no_grad():
        outputs = lstm_model(**tokens)
        lstm_out = outputs[0][:, -1, :]
        features = lstm_out.cpu().numpy()
    
    return features

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    features = get_features(text)
    prediction = rf_model.predict(features)
    
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
