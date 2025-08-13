import os
import pandas as pd
import numpy as np
import joblib
import argparse
from src.model.price_predictor import PricePredictor
from src.utils.custom_logger import log

def run_price_prediction(data_df, model_path, sequence_length=24):
    """
    Runs price prediction on new data.
    Args:
        data_df (pd.DataFrame): The dataframe containing features for inference.
        model_path (str): The full path to the trained model file.
        sequence_length (int): The length of the input sequence for the LSTM model.
    Returns:
        np.ndarray: Predicted prices.
    """
    log.info("Starting price prediction inference...")

    if not os.path.exists(model_path):
        log.error(f"Error: Price predictor model not found at {model_path}. Please train the model first.")
        return None

    # 2. Prepare inference data
    log.info("Preparing inference data...")
    # Drop target columns if they exist in inference data
    X_inference = data_df.drop(columns=['price', 'day_ahead_price', 'issue_time_utc_str'], errors='ignore').values
    # Ensure X_inference has the correct number of features (229) as expected by the model
    if X_inference.shape[1] > 229:
        X_inference = X_inference[:, :229]
    elif X_inference.shape[1] < 229:
        log.error(f"Error: X_inference has {X_inference.shape[1]} features, but the model expects 229. Cannot proceed.")
        return None

    def create_sequences_for_lstm(X_data, seq_length):
        X_seq = []
        for i in range(len(X_data) - seq_length + 1):
            X_seq.append(X_data[i:(i + seq_length)])
        return np.array(X_seq)

    if len(X_inference) < sequence_length:
        log.error(f"Inference data length ({len(X_inference)}) is less than sequence length ({sequence_length}). Cannot make predictions.")
        return None

    X_seq_inference = create_sequences_for_lstm(X_inference, sequence_length)

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
    parser = argparse.ArgumentParser(description="Run price prediction inference.")
    parser.add_argument('--exp_path', type=str, required=True, help='Path to the experiment directory containing the model.')
    args = parser.parse_args()

    # Construct model path from experiment path
    model_path = os.path.join(args.exp_path, 'price_predictor.pth')

    # Example usage: Load a small part of the final_dataset for inference
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    processed_data_path = os.path.join(base_dir, 'data', 'processed', 'aggregated_data.csv')
        
    try:
        full_df = pd.read_csv(processed_data_path, index_col=0, parse_dates=True)
        # Use a subset of data for demonstration
        sample_inference_data = full_df.iloc[100:200] # Example: take 100 data points
        predictions = run_price_prediction(sample_inference_data, model_path)
        
        if predictions is None:
            log.error("Inference resulted in no predictions. Exiting with error.")
            exit(1)

        print(f"Sample Predictions (first 5):\n{predictions[:5]}")

    except FileNotFoundError:
        log.error(f"Could not find {processed_data_path}. Please run data processing first.")
        exit(1)
    except Exception as e:
        log.error(f"An error occurred during sample inference: {e}")
        exit(1)
