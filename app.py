from flask import Flask, request, jsonify
import numpy as np
import librosa
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("model.h5")

labels = ['belly_pain', 'burping', 'cold_hot', 'discomfort', 'hungry', 'lonely', 'scared', 'tired', 'other']

def extract_mel_spectrogram(file_path, max_len=128):
    y, sr = librosa.load(file_path, sr=22050)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
    if mel_norm.shape[1] < max_len:
        pad_width = max_len - mel_norm.shape[1]
        mel_norm = np.pad(mel_norm, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_norm = mel_norm[:, :max_len]
    return mel_norm

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    file_path = "temp.wav"
    file.save(file_path)

    spectrogram = extract_mel_spectrogram(file_path)
    spectrogram = np.expand_dims(spectrogram, axis=(0, -1))  # (1, 128, 128, 1)

    prediction = model.predict(spectrogram)[0]
    predicted_label = labels[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return jsonify({
        "prediction": predicted_label,
        "confidence": round(confidence, 4)
    })