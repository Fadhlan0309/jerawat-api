from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import io

app = Flask(__name__)

# Muat model dan scaler
with open('elm_model.pkl', 'rb') as model_file:
    loaded_elm = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil file gambar dari request
    file = request.files['image']
    img = Image.open(file.stream)

    # Proses gambar untuk mengekstrak fitur (contoh: GLCM)
    features = extract_features(img)  # Implementasikan fungsi ini sesuai kebutuhan

    # Normalisasi fitur
    manual_input_df = pd.DataFrame([features])
    manual_input_normalized = loaded_scaler.transform(manual_input_df)

    # Prediksi dengan model
    prediction = loaded_elm.predict(manual_input_normalized)
    predicted_class = prediction[0]  # Ambil nilai prediksi
    
    return jsonify({'predicted_class': int(predicted_class)})

def extract_features(image):
    # Implementasikan ekstraksi fitur dari gambar
    # Kembalikan fitur sebagai list atau array
    return [0.5, 0.5, 0.5, 0.5, 0.5]  # Contoh placeholder

if __name__ == '__main__':
    app.run(debug=True)