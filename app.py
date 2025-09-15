import streamlit as st
import pandas as pd
import joblib
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="No-Show Prediction App",
    page_icon="🗓️",
    layout="centered"
)

# --- Function to load the model and encoders ---
# The st.cache_resource decorator ensures this function runs only once.
@st.cache_resource
def load_resources():
    model_path = "mango_model.pkl"
    encoders_path = "mango_label_encoders.pkl"
    data_path = "new.xlsx"

    if not os.path.exists(model_path):
        st.error(f"Error: The model file '{model_path}' was not found.")
        st.stop()
    if not os.path.exists(encoders_path):
        st.error(f"Error: The encoders file '{encoders_path}' was not found.")
        st.stop()
    if not os.path.exists(data_path):
        st.error(f"Error: The data file '{data_path}' was not found.")
        st.stop()

    try:
        loaded_model = joblib.load(model_path)
        loaded_label_encoders = joblib.load(encoders_path)
        # Load the data just to get the list of columns
        df = pd.read_excel(data_path)
        cat_cols = [col for col in df.columns if col in loaded_label_encoders]
        feature_cols = df.drop(columns=["status", "Unnamed: 0"], errors='ignore').columns.tolist()
        
        return loaded_model, loaded_label_encoders, cat_cols, feature_cols
    
    except Exception as e:
        st.error(f"An error occurred while loading files: {e}")
        st.stop()

# Load the resources
model, label_encoders, cat_cols, feature_cols = load_resources()

# --- Main App Title and Description ---
st.title("🗓️ No-Show Prediction App")
st.write("This application predicts whether a person will attend their appointment based on various factors.")
st.markdown("---")

# --- User Input Section ---
st.header("Enter Details for Prediction")

# Dictionary to store user inputs
user_input_data = {}

# Create input widgets for each feature
for col in feature_cols:
    if col in cat_cols:
        # Get options from the loaded LabelEncoder
        le = label_encoders[col]
        options = le.classes_
        
        # Use a selectbox for categorical features
        selected_option = st.selectbox(
            f"Select value for **'{col}'**:",
            options=options,
            key=col
        )
        user_input_data[col] = selected_option
    else:
        # Use a number input for numerical features
        try:
            # We can use the original dataframe to get a hint of the data type
            df_temp = pd.read_excel("new.xlsx")
            default_value = df_temp[col].mean()
            if df_temp[col].dtype == 'int64':
                 user_input_data[col] = st.number_input(
                    f"Enter value for **'{col}'**:",
                    value=int(default_value),
                    step=1,
                    key=col
                )
            else:
                user_input_data[col] = st.number_input(
                    f"Enter value for **'{col}'**:",
                    value=default_value,
                    key=col
                )
        except:
            user_input_data[col] = st.number_input(
                f"Enter value for **'{col}'**:",
                value=0.0,
                key=col
            )


# --- Prediction Button ---
if st.button("Predict No-Show Status", type="primary"):
    # Convert user input to a DataFrame
    user_df = pd.DataFrame([user_input_data])
    
    # Ensure column order matches the training data
    user_df = user_df[feature_cols]

    # Encode categorical columns using the loaded encoders
    for col in cat_cols:
        if col in user_df.columns:
            le = label_encoders[col]
            try:
                user_df[col] = le.transform(user_df[col])
            except ValueError as e:
                st.error(f"Error encoding column '{col}': {e}. Please select a valid option from the list.")
                st.stop()
    
    # Convert all columns to numeric for prediction
    user_df = user_df.apply(pd.to_numeric)
    
    # Make the prediction
    try:
        prediction_numeric = model.predict(user_df)
        
        # Decode the numerical prediction back to the original string
        status_le = label_encoders["status"]
        prediction_status = status_le.inverse_transform(prediction_numeric)
        
        st.markdown("---")
        st.subheader("Prediction Result")
        st.success(f"The predicted status is: **{prediction_status[0]}**")
        st.balloons()
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
