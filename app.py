import streamlit as st
import joblib
import numpy as np

# Load the trained model and encoder
model = joblib.load("iris_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# Streamlit UI
st.title("🌸 Iris Flower Species Predictor")
st.write("Enter the flower's measurements and get the predicted species!")

# Input fields for user data
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

# Predict button
if st.button("Predict"):
    # Prepare input data
    new_sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Make prediction
    predicted_class = model.predict(new_sample)
    species_name = encoder.inverse_transform(predicted_class)

    # Display result
    st.success(f"🌿 Predicted Species: **{species_name[0]}**")