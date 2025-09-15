import streamlit as st
import pandas as pd
import joblib
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="No-Show Prediction App",
    page_icon="🗓️",
    layout="wide"  # Use wide layout for side-by-side inputs
)

# --- Function to load the model and encoders ---
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
        df = pd.read_excel(data_path)
        cat_cols = [col for col in df.columns if col in loaded_label_encoders]
        feature_cols = df.drop(columns=["status", "Unnamed: 0"], errors='ignore').columns.tolist()
        return loaded_model, loaded_label_encoders, cat_cols, feature_cols
    except Exception as e:
        st.error(f"An error occurred while loading files: {e}")
        st.stop()

# --- Load resources ---
model, label_encoders, cat_cols, feature_cols = load_resources()

# --- App Title ---
st.title("🗓️ No-Show Prediction App")
st.write("Predict whether a person will attend their appointment based on various factors.")
st.markdown("---")

# --- User Input Section ---
st.header("Enter Details for Prediction")
user_input_data = {}

# Determine columns per row (for side-by-side layout)
cols_per_row = 3
# Corrected calculation without using the 'math' library
num_features = len(feature_cols)
num_rows = num_features // cols_per_row + (1 if num_features % cols_per_row > 0 else 0)

for row_idx in range(num_rows):
    # Select the subset of features for this row
    start_idx = row_idx * cols_per_row
    end_idx = start_idx + cols_per_row
    row_features = feature_cols[start_idx:end_idx]
    
    cols = st.columns(len(row_features))
    
    for i, col in enumerate(row_features):
        with cols[i]:
            if col in cat_cols:
                le = label_encoders[col]
                options = le.classes_
                user_input_data[col] = st.selectbox(f"**{col}**", options=options, key=col)
            else:
                try:
                    df_temp = pd.read_excel("new.xlsx")
                    default_value = df_temp[col].mean()
                    if df_temp[col].dtype == 'int64':
                        user_input_data[col] = st.number_input(f"**{col}**", value=int(default_value), step=1, key=col)
                    else:
                        user_input_data[col] = st.number_input(f"**{col}**", value=float(default_value), key=col)
                except:
                    user_input_data[col] = st.number_input(f"**{col}**", value=0.0, key=col)

# --- Prediction Button ---
if st.button("Predict No-Show Status", type="primary"):
    try:
        user_df = pd.DataFrame([user_input_data])
        user_df = user_df[feature_cols]
        
        # Encode categorical columns
        for col in cat_cols:
            if col in user_df.columns:
                le = label_encoders[col]
                try:
                    user_df[col] = le.transform(user_df[col])
                except ValueError as e:
                    st.error(f"Error encoding column '{col}': {e}. Please select a valid option from the list.")
                    st.stop()
        
        # Ensure all data is numeric
        user_df = user_df.apply(pd.to_numeric)
        
        # Make prediction
        prediction_numeric = model.predict(user_df)
        status_le = label_encoders["status"]
        prediction_status = status_le.inverse_transform(prediction_numeric)
        
        st.markdown("---")
        st.subheader("Prediction Result")
        st.success(f"The predicted status is: **{prediction_status[0]}**")
        st.balloons()
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
