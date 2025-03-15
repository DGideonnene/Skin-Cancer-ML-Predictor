import numpy as np
import pandas as pd
import joblib
import streamlit as st

# Load Model
model = joblib.load('skin_cancer_prediction_model.pkl')

def performance_prediction(input_data):
    column_names = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                    'smoothness_mean', 'compactness_mean', 'concavity_mean',
                    'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                    'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
                    'fractal_dimension_se', 'radius_worst', 'texture_worst',
                    'perimeter_worst', 'area_worst', 'smoothness_worst',
                    'compactness_worst', 'concavity_worst', 'concave_points_worst',
                    'symmetry_worst', 'fractal_dimension_worst']

    input_data = np.array(input_data).reshape(1, -1)
    input_df = pd.DataFrame(input_data, columns=column_names)
    pred = model.predict(input_df)

    if pred[0] == 1:
        return "Malignant: The tumor is cancerous and may invade nearby tissues. Seek medical attention immediately."
    else:
        return "Benign: The tumor is non-cancerous. Regular checkups are recommended."

# Streamlit UI
def main():
    st.set_page_config(page_title="Skin Cancer Prediction", layout="wide")
    st.markdown("""<style>
        .main {background-color: #f4f4f9;}
        h1 {color: #007bff; text-align: center;}
        .stButton>button {width: 100%; background-color: #007bff; color: white;}
    </style>""", unsafe_allow_html=True)

    st.title("🔬 Skin Cancer Prediction")
    st.write("Enter the following measurements for a medical diagnosis.")

    # Create columns for a clean UI
    col1, col2, col3 = st.columns(3)
    inputs = []
    fields = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
              'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean',
              'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
              'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
              'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
              'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst',
              'symmetry_worst', 'fractal_dimension_worst']

    for i, field in enumerate(fields):
        with [col1, col2, col3][i % 3]:
            value = st.text_input(f"{field.replace('_', ' ').capitalize()}", key=field)
            inputs.append(value)

    prediction = ""

    if st.button("Predict Diagnosis"):
        try:
            input_data = [float(value) for value in inputs]
            prediction = performance_prediction(input_data)
        except ValueError:
            prediction = "Please ensure all fields are filled with valid numeric values."

    if prediction:
        st.subheader("Diagnosis Result:")
        st.write(prediction)

if __name__ == '__main__':
    main()
