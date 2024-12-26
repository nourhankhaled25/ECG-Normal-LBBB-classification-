import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import joblib  
import script 
#  streamlit run deployment.py 
knn_model = joblib.load("knn_model.pkl")
svm_model = joblib.load("svm_model.pkl")
rf_model = joblib.load("rf_model.pkl")


st.set_page_config(page_title="ECG Signal Prediction", page_icon="🫀", layout="wide")
st.title("ECG Signal Prediction")
st.markdown("""
    **This app helps in the classification of ECG signals into Normal or LBBB (Left Bundle Branch Block).**
            
    1. **Preprocessing**: Signal cleaning through mean removal, bandpass filtering, and normalization.
    2. **Feature Extraction**: Using Daubechies wavelet transform for capturing key features.
    3. **Classification**: Using machine learning models (KNN, SVM, and Random Forest).
""")


uploaded_file = st.file_uploader("Upload your ECG Test File", type=["txt", "csv"])
if uploaded_file:
    test_data = pd.read_csv(uploaded_file, sep="|", header=None)
    st.success(f"Successfully loaded {len(test_data)} signals.")
    
    
    if st.checkbox("Preview Uploaded Data"):
        st.dataframe(test_data)

    signal_number = st.number_input(
        "Enter Signal Number (Row Index):",
        min_value=0,
        max_value=len(test_data) - 1,
        step=1,
        value=0,
    )

    model_choice = st.selectbox(
        "Select Classification Model",
        ("KNN (K-Nearest Neighbors)", "SVM (Support Vector Machine)", "Random Forest Classifier")
    )

    if st.button("Predict Signal"):
        selected_signal = test_data.iloc[signal_number].dropna().tolist()
        
        st.write(f"### Selected Signal (Row {signal_number}):")
        st.line_chart(selected_signal)  

        # Preprocess
        preprocessed_data = script.preprocess([selected_signal]) 

        # feature extraction
        coeffs = script.wavedec(preprocessed_data[0])
        approximation_coeffs = coeffs[0] 
        
        # statistics
        features = script.compute_statistics(approximation_coeffs)
        features = np.array(features).reshape(1, -1) 

        # predictions
        if model_choice == "KNN (K-Nearest Neighbors)":
            prediction = knn_model.predict(features)[0]
        elif model_choice == "SVM (Support Vector Machine)":
            prediction = svm_model.predict(features)[0]
        else:
            prediction = rf_model.predict(features)[0]

    
        st.subheader(f"Prediction from {model_choice}:")

        if prediction == "Normal":
            st.success("✅ No heart disease detected. The ECG signal appears to be normal.")  
    
        elif prediction == "LBBB":
            st.warning("⚠️ Left Bundle Branch Block (LBBB) detected.")
            