import pandas as pd
import joblib 
import os

# --- 1. Load the saved model and LabelEncoders ---
print("\n--- Loading the saved model and LabelEncoders ---")
model_path = "mango_model.pkl"
encoders_path = "mango_label_encoders.pkl"
data_path = "new.xlsx"

if not os.path.exists(model_path) or not os.path.exists(encoders_path) or not os.path.exists(data_path):
    print("Error: Saved model, encoder, or data files not found.")
    exit()

try:
    loaded_model = joblib.load(model_path)
    loaded_label_encoders = joblib.load(encoders_path)
    print("✅ Model and LabelEncoders loaded successfully.")
    
    # Load the data file to get the list of features and their types
    df = pd.read_excel(data_path)
    # Get the list of columns used for training (features)
    feature_cols = df.drop(columns=["status", "Unnamed: 0"], errors='ignore').columns.tolist()
    # Get the list of categorical columns from the loaded encoders
    cat_cols = list(loaded_label_encoders.keys())

except Exception as e:
    print(f"Error loading files: {e}")
    exit()

# --- 2. Test with new user input ---
print("\n--- Testing with User Input ---")
user_input_data = {}

# Loop through each feature to get user input
for col in feature_cols:
    if col in cat_cols:
        # Check if the column exists in the encoders (could be the target 'status')
        if col in loaded_label_encoders:
            le = loaded_label_encoders[col]
            options = le.classes_
            print(f"\nSelect a value for '{col}':")
            for i, option in enumerate(options):
                print(f"[{i}] {option}")
            while True:
                try:
                    choice_index = int(input("Enter the number corresponding to your choice: "))
                    if 0 <= choice_index < len(options):
                        value = options[choice_index]
                        break
                    else:
                        print("Invalid choice. Please enter a valid number.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            user_input_data[col] = [value]
    else:
        # Handle numerical input
        while True:
            try:
                value = float(input(f"Enter value for '{col}': "))
                user_input_data[col] = [value]
                break
            except ValueError:
                print("Invalid input. Please enter a number.")


# --- 3. Prepare data and make prediction ---
# Convert input to DataFrame
user_df = pd.DataFrame(user_input_data)

# Encode categorical columns
for col in cat_cols:
    if col in user_df.columns:
        le = loaded_label_encoders[col]
        user_df[col] = le.transform(user_df[col])

# Make the prediction
try:
    prediction_numeric = loaded_model.predict(user_df)

    # Decode the numerical prediction back to the original string
    status_le = loaded_label_encoders["status"]
    prediction_status = status_le.inverse_transform(prediction_numeric)

    print("\n--- Prediction for your input ---")
    print(f"The predicted status is: {prediction_status[0]}")
    
except Exception as e:
    print(f"\nAn error occurred during prediction: {e}")
