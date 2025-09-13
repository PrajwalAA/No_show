import streamlit as st
import pandas as pd
import joblib
import os

# --- Configuration & Utility Functions ---

# Use st.cache_data to cache the loaded model and encoders.
# This prevents them from being reloaded every time the app is updated.
@st.cache_data
def load_resources():
    """Loads the pre-trained model and label encoders."""
    if os.path.exists("mango_model.pkl") and os.path.exists("mango_label_encoders.pkl"):
        try:
            model = joblib.load("mango_model.pkl")
            encoders = joblib.load("mango_label_encoders.pkl")
            return model, encoders
        except Exception as e:
            st.error(f"Error loading model or encoders: {e}")
            return None, None
    else:
        st.error("Error: 'mango_model.pkl' or 'mango_label_encoders.pkl' not found.")
        st.info("Please ensure these files are in the same directory as this script.")
        return None, None

# --- Main Application ---

st.set_page_config(
    page_title="Mango Status Predictor",
    page_icon="🥭",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🥭 Mango Status Predictor")
st.markdown("Enter the attributes of the mango to predict its ripeness status.")

# Load the model and encoders. The app will halt if they are not found.
loaded_model, loaded_label_encoders = load_resources()

if loaded_model is None or loaded_label_encoders is None:
    st.stop() # Stop the app execution if files are missing.

# In a real-world scenario, you would have access to the feature names from the training script.
# For this example, we'll assume the following columns and their types.
# You must adjust these to match your actual training data.
FEATURE_COLS = ['Size', 'Color', 'Sweetness', 'Aroma', 'Weight']
CAT_COLS = ['Color', 'Aroma']
NUM_COLS = [col for col in FEATURE_COLS if col not in CAT_COLS]

# The target variable's label encoder is also needed for a meaningful output.
# We'll assume the target variable was named 'Status'.
TARGET_ENCODER = loaded_label_encoders.get('Status')

# --- User Input Form ---

with st.form("input_form"):
    st.header("Enter Mango Details")
    user_input_data = {}

    # Gather categorical inputs
    for col in CAT_COLS:
        if col in loaded_label_encoders:
            le = loaded_label_encoders[col]
            options = le.classes_
            user_input_data[col] = st.selectbox(f"Select a value for '{col}'", options)
        else:
            st.warning(f"Label encoder for '{col}' not found. Please check your `mango_label_encoders.pkl` file.")
            user_input_data[col] = st.text_input(f"Enter value for '{col}' (no encoder found)")

    # Gather numerical inputs
    for col in NUM_COLS:
        user_input_data[col] = st.number_input(f"Enter value for '{col}'", value=1.0)
    
    # Prediction button
    submitted = st.form_submit_button("Predict Status")

# --- Prediction Logic ---
if submitted:
    try:
        # Convert input to DataFrame, ensuring columns match the model's training data.
        # The order of columns in the DataFrame must be the same as the training data.
        user_df = pd.DataFrame([user_input_data])[FEATURE_COLS]

        # Encode categorical columns using the loaded encoders.
        for col in CAT_COLS:
            le = loaded_label_encoders[col]
            # Use le.transform on the single value in the DataFrame column
            user_df[col] = le.transform(user_df[col])

        # Ensure all columns are of numeric format for the model.
        user_df = user_df.apply(pd.to_numeric, errors='coerce')
        user_df.dropna(inplace=True)
        if user_df.empty:
            st.error("Invalid input detected. Please check your values.")
        else:
            # Make prediction
            prediction_encoded = loaded_model.predict(user_df)

            # Get the human-readable prediction label.
            if TARGET_ENCODER:
                prediction_label = TARGET_ENCODER.inverse_transform(prediction_encoded)[0]
                st.success(f"The predicted status is: **{prediction_label}**")
            else:
                st.warning("Target label encoder not found. Displaying raw prediction.")
                st.info(f"The predicted status is: {prediction_encoded[0]}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure your input values are correct and the model files are compatible.")
