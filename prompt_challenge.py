import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Load the trained model and preprocessing objects
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
bmi_encoder = joblib.load('bmi_encoder.pkl')
age_encoder = joblib.load('age_encoder.pkl')
feature_columns = joblib.load('feature_columns.pkl')

def calculate_bmi(weight, height):
    return weight / ((height/100) ** 2)

def get_bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Normal'
    elif bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

def get_age_bucket(age):
    if age <= 25:
        return '18-25'
    elif age <= 35:
        return '26-35'
    elif age <= 45:
        return '36-45'
    elif age <= 55:
        return '46-55'
    else:
        return '56+'

def prepare_input_data(input_dict):
    # Create DataFrame with all features in correct order
    df = pd.DataFrame({col: [0] for col in feature_columns})
    
    # Update with actual values
    for key, value in input_dict.items():
        if key in df.columns:
            df[key] = value
    
    return df

def main():
    st.title('Health Insurance Premium Predictor')
    
    # Input form
    st.header('Enter Your Information')
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input('Age', min_value=18, max_value=100, value=30)
        height = st.number_input('Height (cm)', min_value=100, max_value=250, value=170)
        weight = st.number_input('Weight (kg)', min_value=30, max_value=200, value=70)
        diabetes = st.checkbox('Diabetes')
        blood_pressure = st.checkbox('Blood Pressure Problems')
    
    with col2:
        transplants = st.checkbox('Any Transplants')
        chronic_diseases = st.checkbox('Any Chronic Diseases')
        allergies = st.checkbox('Known Allergies')
        cancer_history = st.checkbox('History of Cancer in Family')
        surgeries = st.number_input('Number of Major Surgeries', min_value=0, max_value=10)
    
    if st.button('Calculate Premium'):
        # Calculate derived features
        bmi = calculate_bmi(weight, height)
        bmi_category = get_bmi_category(bmi)
        age_bucket = get_age_bucket(age)
        
        # Display derived features
        st.subheader('Calculated Metrics')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('BMI', f'{bmi:.2f}')
        with col2:
            st.metric('BMI Category', bmi_category)
        with col3:
            st.metric('Age Bucket', age_bucket)
        
        # Prepare input dictionary
        input_dict = {
            'Age': age,
            'Height': height,
            'Weight': weight,
            'BMI': bmi,
            'BMI_Category_encoded': bmi_encoder.transform([bmi_category])[0],
            'Age_Bucket_encoded': age_encoder.transform([age_bucket])[0],
            'Diabetes': int(diabetes),
            'BloodPressureProblems': int(blood_pressure),
            'AnyTransplants': int(transplants),
            'AnyChronicDiseases': int(chronic_diseases),
            'KnownAllergies': int(allergies),
            'HistoryOfCancerInFamily': int(cancer_history),
            'NumberOfMajorSurgeries': surgeries
        }
        
        # Create input DataFrame with correct feature order
        input_data = prepare_input_data(input_dict)
        
        # Scale numerical features
        numerical_features = ['Age', 'Height', 'Weight', 'BMI']
        input_data[numerical_features] = scaler.transform(input_data[numerical_features])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display prediction
        st.subheader('Premium Prediction')
        st.success(f'Estimated Annual Premium: ${prediction:,.2f}')
        
        # Display feature importance
        st.subheader('Feature Importance Analysis')
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h')
        fig.update_layout(title='Feature Importance')
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()
