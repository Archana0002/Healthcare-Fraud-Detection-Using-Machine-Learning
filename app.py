import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load your pre-trained ML model (pickled)
model = joblib.load(r'C:\Users\ARCHANA\OneDrive\Desktop\appu\app\Model\random_forest.pkl')

def preprocess_input(user_input):
    # Convert input to DataFrame for consistency
    input_df = pd.DataFrame([user_input])

    # Convert date strings to datetime objects
    for date_col in ['ClaimStartDt', 'ClaimEndDt', 'AdmissionDt', 'DischargeDt', 'DOB', 'DOD']:
        input_df[date_col] = pd.to_datetime(input_df[date_col], format='%d-%m-%Y', errors='coerce')

    # Feature Engineering: Add derived features
    input_df['ClaimDuration'] = (input_df['ClaimEndDt'] - input_df['ClaimStartDt']).dt.days
    input_df['AdmissionYear'] = input_df['AdmissionDt'].dt.year
    input_df['DischargeYear'] = input_df['DischargeDt'].dt.year
    input_df['AgeAtAdmission'] = (input_df['AdmissionDt'] - input_df['DOB']).dt.days // 365

    # Drop the original datetime columns if not needed
    input_df = input_df.drop(columns=['ClaimStartDt', 'ClaimEndDt', 'AdmissionDt', 'DischargeDt', 'DOB', 'DOD'])

    # Encode categorical variables using LabelEncoder
    categorical_cols = ['BeneID', 'ClaimID', 'Provider', 'RenalDiseaseIndicator']
    for col in categorical_cols:
        le = LabelEncoder()
        input_df[col] = le.fit_transform(input_df[col])

    # Ensure the DataFrame is in the correct order of features for the model
    feature_order = [
        'BeneID', 'ClaimID', 'Provider', 'InscClaimAmtReimbursed', 'DeductibleAmtPaid',
        'Gender', 'Race', 'RenalDiseaseIndicator', 'State', 'County',
        'NoOfMonths_PartACov', 'NoOfMonths_PartBCov', 
        'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt',
        'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'Age', 
        'ClaimDuration', 'AdmissionYear', 'DischargeYear', 'AgeAtAdmission'
    ]

    input_df = input_df[feature_order]
    return input_df

def predict_fraud(preprocessed_input):
    """Make predictions using the trained model."""
    prediction = model.predict(preprocessed_input)
    return prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_claim', methods=['POST'])
def get_prediction():
    try:
        # Get user input from the form
        user_input = {key: request.form[key] for key in request.form}

        # Preprocess and predict
        preprocessed_data = preprocess_input(user_input)
        prediction = predict_fraud(preprocessed_data)

        # Determine result and assign a CSS class
        result = "Not Fraud" if prediction[0] == 0 else "Fraud"
        result_class = "not-fraud" if result == "Not Fraud" else "fraud"

        # Render result page with the prediction result and CSS class
        return render_template('result.html', prediction_result=result, result_class=result_class)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
