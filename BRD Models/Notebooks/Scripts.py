import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import os

# Define Model Architecture
class DQKD(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQKD, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Load Model
def load_model(model_path, input_dim, output_dim):
    model = DQKD(input_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Preprocessing Function
def preprocess_input(input_data, selected_features, scaler):
    """
    Preprocesses the input data to match the training data format.

    Args:
        input_data (dict): A dictionary containing input features.
        selected_features (list): List of selected features used during training.
        scaler (StandardScaler): Fitted StandardScaler object.

    Returns:
        torch.Tensor: A preprocessed tensor ready for model inference.
    """
    df = pd.DataFrame([input_data])  # Put data into a DataFrame
    df = df.rename(columns={col: col.lower() for col in df.columns})  # Lowercase column names

    # Ensure only selected features are present
    df = df[selected_features]

    # Convert to numeric, impute with the median
    for col in selected_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(df.median())  # Impute missing values

    # Scale using the fitted scaler
    scaled_data = scaler.transform(df)

    # Convert to tensor
    tensor_data = torch.tensor(scaled_data, dtype=torch.float32)
    return tensor_data

# Prediction Function
def predict(model, input_tensor):
    """
    Makes a prediction using the loaded model.

    Args:
        model (nn.Module): Loaded PyTorch model.
        input_tensor (torch.Tensor): Preprocessed input tensor.

    Returns:
        int: Predicted class label.
    """
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()

def main():
    # Configuration
    MODEL_PATH = 'brd_model.pth'  # Path to your saved model
    SCALER_PATH = 'scaler.pkl' # Path to the scaler object
    SELECTED_FEATURES = [
        'temp', 'nasal', 'eye', 'ears', 'cough', 'weight', 'total_steps', 'total_mi',
        'lying_bouts', 'lying_daily', 'milk_intake', 'milk_percent', 'starter_intake',
        'speed', 'speed_percent', 'brix', 'fecal', 'ultrasound', 'pneuday', 'breath'
    ]

    # Load the scaler
    scaler = load_scaler(SCALER_PATH)

    # Load the Model
    num_classes = 7 # Replace with correct value
    model = load_model(MODEL_PATH, input_dim=len(SELECTED_FEATURES), output_dim=num_classes)

    # Example Input Data (replace with actual data)
    input_data = {
        'temp': 38.5,
        'nasal': 0.0,
        'eye': 0.0,
        'ears': 0.0,
        'cough': 0.0,
        'weight': 50.0,
        'total_steps': 500.0,
        'total_mi': 2500.0,
        'lying_bouts': 15.0,
        'lying_daily': 17.0,
        'milk_intake': 8.0,
        'milk_percent': 80.0,
        'starter_intake': 200.0,
        'speed': 700.0,
        'speed_percent': 100.0,
        'brix': 0.09,
        'fecal': 1.0,
        'ultrasound': 0.0,
        'pneuday': -2.0,
        'breath': 0.0
    }

    # Preprocess Input
    input_tensor = preprocess_input(input_data, SELECTED_FEATURES, scaler)

    # Make Prediction
    prediction = predict(model, input_tensor)

    # Output Prediction
    print(f"Predicted BRD Risk Class: {prediction}")

# Helper Function to Load Scaler
import joblib
def load_scaler(scaler_path):
    """Loads the StandardScaler object."""
    return joblib.load(scaler_path)

if __name__ == "__main__":
    main()
