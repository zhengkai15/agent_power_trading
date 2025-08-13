import os
import pandas as pd
import numpy as np
import joblib
from src.model.price_predictor import PricePredictor
from src.utils.logging import log

def run_price_prediction(data_df, sequence_length=24):
    """
    Runs price prediction on new data.
    Args:
        data_df (pd.DataFrame): The dataframe containing features for inference.
        sequence_length (int): The length of the input sequence for the LSTM model.
    Returns:
        np.ndarray: Predicted prices.
    """
    log.info("Starting price prediction inference...")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    model_path = os.path.join(base_dir, 'models', 'price_predictor.pth')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')

    if not os.path.exists(model_path):
        log.error(f"Error: Price predictor model not found at {model_path}. Please train the model first.")
        return None
    if not os.path.exists(scaler_path):
        log.error(f"Error: Scaler not found at {scaler_path}. Please train the model first.")
        return None

    # 1. Load scaler
    log.info(f"Loading scaler from {scaler_path}...")
    scaler = joblib.load(scaler_path)

    # 2. Prepare inference data
    log.info("Preparing inference data...")
    # Drop target columns if they exist in inference data
    X_inference = data_df.drop(columns=['price', 'day_ahead_price'], errors='ignore').values
    X_inference_scaled = scaler.transform(X_inference)

    # Create sequences for LSTM
    def create_sequences_for_lstm(X_data, seq_length):
        X_seq = []
        for i in range(len(X_data) - seq_length + 1): # +1 to include the last possible sequence
            X_seq.append(X_data[i:(i + seq_length)])
        return np.array(X_seq)

    # For inference, we need to predict for each time step, so we create sequences
    # that end at each point we want to predict.
    # If we want to predict for the entire data_df, we need sequences up to the last element.
    # The sequence_length should be consistent with training.
    # For simplicity, let's assume we predict for the last point of each sequence.
    # If we want to predict for all points, we need to adjust the sequence creation.
    # For now, let's assume we are predicting for the next step given a sequence.
    # So, if data_df has N points, we can form N - sequence_length sequences.
    
    # To predict for all time steps in data_df, we need to create sequences ending at each step.
    # This means for a data_df of length N, we will have N-seq_length+1 predictions.
    # The first prediction will be for data_df.iloc[seq_length-1]
    
    # Let's adjust create_sequences_for_lstm to return sequences for all possible predictions
    # For a sequence of length L, to predict for time t, we need data from t-L to t-1.
    # So, if we have data up to time T, and we want to predict for T, we need sequence ending at T-1.
    # The current setup in train_predictor.py predicts y[i+seq_length] from X[i:i+seq_length]
    # This means X_seq[k] predicts y_seq[k] which corresponds to X_scaled[k+seq_length]
    # So, for inference, if we want to predict for data_df.iloc[t], we need X_scaled[t-sequence_length:t]
    
    # Let's simplify for now: assume we want to predict for the entire X_inference_scaled
    # We will create sequences for all possible prediction points.
    # The output will be for the time points corresponding to the end of each sequence.
    
    # If X_inference_scaled has N rows, and sequence_length is L, we can form N-L+1 sequences.
    # The prediction for sequence X_seq[i] will correspond to the time index of X_inference_scaled[i+L-1]
    
    if len(X_inference_scaled) < sequence_length:
        log.error(f"Inference data length ({len(X_inference_scaled)}) is less than sequence length ({sequence_length}). Cannot make predictions.")
        return None

    X_seq_inference = create_sequences_for_lstm(X_inference_scaled, sequence_length)

    # 3. Load model
    log.info(f"Loading price predictor model from {model_path}...")
    input_dim = X_seq_inference.shape[2]
    predictor = PricePredictor(input_dim=input_dim)
    predictor.load(model_path)

    # 4. Make predictions
    predictions = predictor.predict(X_seq_inference)
    log.info("Price prediction inference finished.")
    return predictions

if __name__ == '__main__':
    # Example usage: Load a small part of the final_dataset for inference
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    processed_data_path = os.path.join(base_dir, 'data', 'aexp', 'final_dataset.csv')
    
    try:
        full_df = pd.read_csv(processed_data_path, index_col=0, parse_dates=True)
        # Use a subset of data for demonstration
        sample_inference_data = full_df.iloc[100:200] # Example: take 100 data points
        predictions = run_price_prediction(sample_inference_data)
        if predictions is not None:
            print(f"Sample Predictions (first 5):\n{predictions[:5]}")
    except FileNotFoundError:
        log.error(f"Could not find {processed_data_path}. Please run data processing first.")
    except Exception as e:
        log.error(f"An error occurred during sample inference: {e}")
