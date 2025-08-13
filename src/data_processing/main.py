import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.data_processing.loader import load_price_data, load_all_point_data, align_data
from src.data_processing.preprocessor import generate_day_ahead_price_demo
from src.utils.fix_filename import rename_files_in_directory

def process_all_data():
    """
    Main function to process all data, merge them, and save the final dataset.
    """
    # --- Fix potential garbled filenames ---
    features_base_path = os.path.join(project_root, 'data', 'raw', 'station', 'hres', 'results')
    print(f"Fixing filenames in {features_base_path}...")
    rename_files_in_directory(features_base_path)

    # --- 1. Process Price Data ---
    price_raw_path = os.path.join(project_root, 'data', 'raw', '甘肃统一 结算点价格.xlsx')
    price_processed_path = os.path.join(project_root, 'data', 'processed', 'processed_price.csv')
    os.makedirs(os.path.dirname(price_processed_path), exist_ok=True)

    price_df = load_price_data(price_raw_path)
    price_df.to_csv(price_processed_path, index=False)
    print(f"Processed price data saved to {price_processed_path}")

    # --- 2. Load and Align Feature Data ---
    features_base_path = os.path.join(project_root, 'data', 'raw', 'station', 'hres', 'results')
    
    print(f"\nLoading all point data from {features_base_path}...")
    all_point_data_df = load_all_point_data(features_base_path)
    
    if all_point_data_df.empty:
        print("No point data loaded. Exiting feature processing.")
        return

    print("Aligning weather data with price data...")
    final_aligned_df = align_data(price_df.copy(), all_point_data_df.copy())

    if final_aligned_df.empty:
        print("No data processed after alignment. Exiting.")
        return

    # Ensure 'datetime_bj' is the index for further processing
    final_aligned_df = final_aligned_df.set_index('datetime_bj')

    # --- 3. Generate Day-Ahead Price Demo Data ---
    print("Generating day-ahead price demo data...")
    # Assuming the date range for demo data should cover the aligned data's date range
    start_date = final_aligned_df.index.min().strftime('%Y-%m-%d')
    end_date = final_aligned_df.index.max().strftime('%Y-%m-%d')
    day_ahead_price_df = generate_day_ahead_price_demo(start_date, end_date)

    # Merge day-ahead price data with the aligned data
    final_aligned_df = final_aligned_df.merge(day_ahead_price_df, left_index=True, right_index=True, how='left')
    print("Day-ahead price data merged.")

    final_processed_df = final_aligned_df

    # --- Save Final Processed Data ---
    final_dataset_path = os.path.join(project_root, 'data', 'processed', 'aggregated_data.csv')
    final_processed_df.to_csv(final_dataset_path)

    print(f"\nFinal processed dataset saved to {final_dataset_path}")
    print("-- Final Processed Dataset Head --")
    print(final_processed_df.head())
    print("\n-- Final Processed Dataset Info --")
    final_processed_df.info()


if __name__ == '__main__':
    process_all_data()
