from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Define location list
locations = [
    '7th Avenue', '9th Avenue', 'Ali Pur', 'Alipur Farash', 'Arsalan Town', 'B-17', 'Bahria Town',
    'Bani Gala', 'Bhara kahu', 'Burma Town', 'C-18', 'CBR Town', 'Chak Shahzad', 'Chatha Bakhtawar', 'D-12',
    'D-17', 'DHA Defence', 'E-11', 'E-18', 'E-7', 'Emaar Canyon Views', 'F-10', 'F-11', 'F-15', 'F-17',
    'F-6', 'F-7', 'F-8', 'FECHS', 'Faisal Town - F-18', 'G-10', 'G-11', 'G-13', 'G-14', 'G-15', 'G-6',
    'G-7', 'G-8', 'G-9', 'Garden Town', 'Ghauri Town', 'Golra Mor', 'Green Avenue', 'Gulberg',
    'Gulshan-e-Khudadad', 'H-13', 'I-10', 'I-13', 'I-14', 'I-8', 'I-9', 'Islamabad - Murree Expressway',
    'Islamabad Expressway', 'Islamabad Highway', 'Jhang Syedan', 'Jhangi Syedan', 'Kashmir Highway',
    'Khanna Pul', 'Koral Chowk', 'Koral Town', 'Korang Town', 'Kuri Road', 'Lehtarar Road', 'Madina Town',
    'Margalla Town', 'Marwa Town', 'Meherban Colony', 'National Police Foundation',
    'National Police Foundation O-9', 'Naval Anchorage', 'Other', 'PWD Housing Scheme', 'PWD Road',
    'Pakistan Town', 'Park Road', 'Park View City', 'River Garden', 'Shah Allah Ditta', 'Shaheen Town',
    'Shahpur', 'Shehzad Town', 'Soan Garden', 'Swan Garden', 'Taramrri', 'Tarlai', 'Tarnol', 'Thanda Pani',
    'Zaraj Housing Scheme'
]

@app.route('/')
def index():
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        baths = int(request.form['baths'])
        purpose = int(request.form['purpose'])
        bedrooms = int(request.form['bedrooms'])
        area_type = int(request.form['area_type'])  # 1 for Marla, 0 for Kanal
        area_input = float(request.form['area'])
        location_index = int(request.form['location'])
    except ValueError:
        return jsonify({'error': 'Invalid input, please check your values.'})

    # Convert area to marlas based on area_type
    area_marla = area_input if area_type == 1 else area_input * 20

    # Create one-hot encoded location
    location_onehot = np.zeros(len(locations))
    location_onehot[location_index] = 1

    # Combine all features
    features = [baths, purpose, bedrooms, area_type, area_marla] + location_onehot.tolist()

    # Create a DataFrame to match scaler's expected input format
    feature_columns = ['baths', 'purpose', 'bedrooms', 'Area Type', 'area_marla'] + \
                      [f"location_{loc}" for loc in locations]
    features_df = pd.DataFrame([features], columns=feature_columns)

    # Standardize the features
    features_scaled = scaler.transform(features_df)

    # Predict log price
    predicted_log_price = model.predict(features_scaled)[0]
    predicted_price = np.exp(predicted_log_price)

    # Handle invalid predictions
    if np.isnan(predicted_price) or predicted_price <= 0:
        return jsonify({'error': 'Unable to predict price. Invalid data.'})

    # Format the predicted price
    if predicted_price >= 100000:
        predicted_price = f"{predicted_price / 100000:.2f} Lakh"
    else:
        predicted_price = f"{int(predicted_price)}"

    return jsonify({'predicted_price': predicted_price})

if __name__ == '__main__':
    app.run(debug=True)
