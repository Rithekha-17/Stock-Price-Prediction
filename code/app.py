from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)

# Load model, scaler, and metrics
model = pickle.load(open('bagging_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
metrics = pickle.load(open('metrics.pkl', 'rb'))

@app.route('/')
def home():
    return 'Flask API is running successfully!'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[float(data['stock2']), float(data['stock3']), float(data['stock4']), float(data['stock5'])]])
    
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    return jsonify({
        'prediction': float(prediction),
        'mse': round(metrics['mse'], 4),
        'r2': round(metrics['r2'], 4),
        'model': 'Bagging Regressor'  # ✅ added model name to the response
    })

if __name__ == '__main__':
    app.run(debug=True)
