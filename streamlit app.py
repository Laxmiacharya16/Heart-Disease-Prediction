import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the scaler and the model
scaler_path = 'scaler.pkl'
model_path = 'bestrf_model.pkl'

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Title of the app with custom styling
st.markdown("""
    <style>
    .main {
        background-color: #000000; /* Black background */
        color: #FFFFFF; /* White text color */
    }
    .title {
        color: #FF0000; /* Red color for the title */
        font-weight: bold; /* Bold text */
    }
    .description, .info {
        color: #FFFFFF; /* White text for description and additional info */
    }
    </style>
    <div class="main">
        <h2 class="title">Welcome to the Heart Disease Prediction App</h2>
        <p class="description">This application predicts the likelihood of heart disease based on user input.</p>
        <p class="info">Fill in the details below and click the <strong>Predict</strong> button to get results.</p>
    </div>
""", unsafe_allow_html=True)

# Function to get user input
def user_input_features():
    age = st.text_input('Age', '50')
    sex = st.text_input('Sex (0 = Female, 1 = Male)', '1')
    cp = st.text_input('Chest Pain Type (0, 1, 2, 3)', '0')
    trestbps = st.text_input('Resting Blood Pressure', '120')
    chol = st.text_input('Serum Cholesterol', '200')
    fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (0 = No, 1 = Yes)', '0')
    restecg = st.text_input('Resting Electrocardiographic Results (0, 1, 2)', '0')
    thalach = st.text_input('Maximum Heart Rate Achieved', '150')
    exang = st.text_input('Exercise Induced Angina (0 = No, 1 = Yes)', '0')
    oldpeak = st.text_input('ST Depression Induced by Exercise', '1.0')
    slope = st.text_input('Slope of the Peak Exercise ST Segment (0, 1, 2)', '0')
    ca = st.text_input('Number of Major Vessels Colored by Fluoroscopy (0, 1, 2, 3, 4)', '0')
    thal = st.text_input('Thalassemia (0, 1, 2, 3)', '0')

    data = {
        'age': [int(age)],
        'sex': [int(sex)],
        'cp': [int(cp)],
        'trestbps': [int(trestbps)],
        'chol': [int(chol)],
        'fbs': [int(fbs)],
        'restecg': [int(restecg)],
        'thalach': [int(thalach)],
        'exang': [int(exang)],
        'oldpeak': [float(oldpeak)],
        'slope': [int(slope)],
        'ca': [int(ca)],
        'thal': [int(thal)]
    }
    features = pd.DataFrame(data)
    return features

input_df = user_input_features()

# Add a stylish button
if st.button('Predict'):
    # Preprocess the input data
    input_scaled = scaler.transform(input_df)

    # Make predictions
    prediction = model.predict(input_scaled)

    # Display results
    st.markdown('---')  # Horizontal line for separation

    st.markdown(f"""
        <div class="main">
            <h3 class="prediction">Prediction</h3>
            The model predicts: **{np.array(['No', 'Yes'])[prediction][0]}**
        </div>
    """, unsafe_allow_html=True)

    st.markdown('---')  # Horizontal line for separation

    # Provide some additional information
    st.markdown("""
        <div class="main">
            <h3 class="about">About</h3>
            This model uses various features to predict heart disease. The features used include age, sex, chest pain type, and other health metrics. 
            If you have any questions or feedback, feel free to reach out!
        </div>
    """, unsafe_allow_html=True)
