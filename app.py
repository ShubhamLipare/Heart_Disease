import streamlit as st
import requests
import json

def main():
    st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
    st.title("Heart Disease Prediction App")
    st.write("Enter the required details below to predict the likelihood of heart disease.")
    
    # Input fields
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=200)
    chol = st.number_input("Cholesterol Level", min_value=100, max_value=600)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia Type", [0, 1, 2, 3])
    
    # Prepare input data
    input_data = {
        "features": [
            age, 
            1 if sex == "Male" else 0, 
            cp, 
            trestbps, 
            chol, 
            fbs, 
            restecg, 
            thalach, 
            exang, 
            oldpeak, 
            slope, 
            ca, 
            thal
        ]
    }
    
    if st.button("Predict Heart Disease"):        
        try:
            response = requests.post("https://heart-disease-api-0g3d.onrender.com/predict", json=input_data) #http://localhost:8000 
            
            if response.status_code == 200:
                result = response.json()["prediction"]
                st.success(f"Prediction: {'Heart Disease Detected' if result == 1 else 'No Heart Disease Detected'}")
            else:
                st.error(f"Error: {response.status_code}, {response.text}")
        
        except requests.exceptions.ConnectionError:
            st.error("Error: Could not connect to FastAPI server. Make sure it's running.")

if __name__ == "__main__":
    main()
