from flask import Flask, request, render_template, jsonify
import os
import numpy as np
import librosa
import joblib
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

app = Flask(__name__)

# Load the trained model and label encoder
model = joblib.load('baby_cry_classifier.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Function to extract features from an audio file
def extract_features(audio, sample_rate):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

# Function to predict the class of an audio file
def predict_class(audio_file):
    try:
        # Load the audio file with librosa
        audio, sample_rate = librosa.load(audio_file, sr=None, res_type='kaiser_fast')
        features = extract_features(audio, sample_rate)
        features = np.expand_dims(features, axis=0)  # Reshape for the model
        
        # Predict the probabilities and the class
        prediction_probabilities = model.predict_proba(features)[0]
        predicted_class_index = np.argmax(prediction_probabilities)
        predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
        probability = prediction_probabilities[predicted_class_index] * 100  # Convert to percentage
        
        return predicted_class, probability
    except Exception as e:
        return f"Error processing the file: {str(e)}", 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save_audio', methods=['POST'])
def save_audio():
    if 'audio_data' in request.files:
        audio = request.files['audio_data']
        filename = datetime.now().strftime("%Y%m%d%H%M%S") + ".wav"
        file_path = os.path.join('uploads', filename)
        audio.save(file_path)
        predicted_class, prediction_probability = predict_class(file_path)
        return jsonify({"status": "success", "predicted_class": predicted_class, "probability": prediction_probability})
    return jsonify({"status": "error"})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
