import pandas as pd
from typing import NewType
import os
from glob import glob
from tqdm import tqdm
from ..utils.fix_filename import rename_files_in_directory

DataFrame = NewType('DataFrame', pd.DataFrame)

def load_price_data(file_path: str) -> DataFrame:
    """
    Loads the price data from the excel file.
    """
    df = pd.read_excel(file_path)
    # Clip the '统一日前' column to the range [40, 650]
    df['统一日前'] = df['统一日前'].clip(lower=40, upper=650)
    return df

def load_feature_csv(file_path: str) -> DataFrame:
    """
    Loads a single feature CSV file.
    """
    return pd.read_csv(file_path)

def load_all_point_data(base_path: str) -> DataFrame:
    """
    Loads all point*.csv files from the nested directory structure,
    fixing filenames if necessary, and concatenates them into a single DataFrame.
    Adds 'issue_time_utc_str' column to each DataFrame based on its path.
    """
    # First, fix filenames in the base_path recursively
    rename_files_in_directory(base_path)

    all_files = glob(os.path.join(base_path, '**', '*point*.csv'), recursive=True)
    
    if not all_files:
        print(f"No point*.csv files found in {base_path}")
        return pd.DataFrame()

    df_list = []
    for file_path in tqdm(all_files, desc="Loading point data files"):
        try:
            # Extract issue_time_utc_str from path: e.g., .../20240101/12/point_data.csv -> 20240101/12
            parts = file_path.split(os.sep)
            # Assuming the structure is .../results/YYYYMMDD/HH/point*.csv
            if len(parts) >= 2:
                issue_time_utc_str = f"{parts[-3]}/{parts[-2]}"
            else:
                issue_time_utc_str = "unknown"

            # Extract full_point_name from filename: e.g., 河东_point_0_102.34_33.59_2024010120-2024010514.csv -> 河东_point_0
            file_name = os.path.basename(file_path)
            # Regex to extract "河西_point_数字" or "河东_point_数字"
            import re
            match = re.match(r'(河[东西])_point_(\d+)', file_name)
            if match:
                full_point_name = f"{match.group(1)}_point_{match.group(2)}"
            else:
                full_point_name = file_name.split('.csv')[0] # Fallback

            df = pd.read_csv(file_path)
            df['issue_time_utc_str'] = issue_time_utc_str
            df['full_point_name'] = full_point_name
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    if not df_list:
        return pd.DataFrame()

    # Concatenate all dataframes
    combined_df = pd.concat(df_list, ignore_index=True)

    # Pivot the table to have full_point_name as columns
    # Assuming 'preTime' is the common time column in the CSVs
    # And other columns are features for each point
    # We need to identify which columns are features and which is time
    # Let's assume the first column is 'preTime' and others are features
    # This part might need adjustment based on actual CSV content
    
    # For now, let's just return the combined_df and handle pivoting in preprocessor.py or main.py
    return combined_df

def align_data(price_df: DataFrame, weather_df: DataFrame) -> DataFrame:
    """
    Aligns weather data with price data based on the specified time alignment rules.
    D+1为起报时间+28H ~起报时间+52H
    """
    # Ensure time columns are datetime objects
    # Assuming price_df has a '时间' column (Beijing Time)
    # Assuming weather_df has a 'preTime' column (UTC time, need to convert to Beijing Time)
    
    # Convert price_df '时间' to datetime and set as index
    price_df['时间'] = pd.to_datetime(price_df['时间'])
    price_df = price_df.set_index('时间')
    
    # Convert weather_df 'preTime' to datetime
    # Assuming 'preTime' column in weather_df is in 'YYYY-MM-DD HH:MM:SS' format
    weather_df['preTime'] = pd.to_datetime(weather_df['preTime'])

    aligned_data_list = []

    # Group weather data by issue_time_utc_str (e.g., '20240101/12')
    for issue_time_utc_str, group_df in tqdm(weather_df.groupby('issue_time_utc_str'), desc="Aligning data"):
        try:
            # Extract report date and hour from issue_time_utc_str
            report_date_str, report_hour_str = issue_time_utc_str.split('/')
            report_datetime_utc = pd.to_datetime(report_date_str + report_hour_str, format='%Y%m%d%H')
            
            # Calculate the start and end time for alignment in Beijing Time
            # D+1为起报时间+28H ~起报时间+52H
            # Calculate the target date for alignment (D+2 from report date)
            # If report_datetime_utc is 2024-01-01 12:00:00 UTC, report_date_utc is 2024-01-01
            # target_date_bj will be 2024-01-03
            report_date_utc = report_datetime_utc.date()
            target_date_bj = pd.to_datetime(report_date_utc) + pd.Timedelta(days=2)

            # Set start and end times for alignment in Beijing Time (00:00:00 to 23:45:00 on target_date_bj)
            start_align_time_bj = target_date_bj
            end_align_time_bj = target_date_bj + pd.Timedelta(hours=23, minutes=45)

            # Filter weather data for the alignment window
            # Convert weather_df 'preTime' to Beijing Time for filtering
            group_df['time_bj'] = group_df['preTime'] # preTime is already in Beijing Time
            filtered_weather_df = group_df[(group_df['time_bj'] >= start_align_time_bj) & (group_df['time_bj'] <= end_align_time_bj)]
            
            if filtered_weather_df.empty:
                continue

            daily_price_df = price_df[price_df.index.date == target_date_bj.date()]

            if daily_price_df.empty:
                continue

            # Resample price data to 15-minute granularity if not already
            # Assuming price_df is already 15min granularity, but good to be explicit
            # daily_price_df = daily_price_df.resample('15min').mean() # Or ffill/bfill

            # Merge based on Beijing Time
            # We need to align weather_df['time_bj'] with price_df.index
            # For each unique full_point_name, we need to merge its features
            
            # Create a common time index for merging (15-minute intervals for the target day)
            common_time_index = pd.date_range(start=start_align_time_bj.floor('15min'), 
                                              end=end_align_time_bj.ceil('15min'), 
                                              freq='15min')
            
            # Create a base DataFrame with the common time index
            aligned_df = pd.DataFrame(index=common_time_index)
            
            # Merge price data
            aligned_df = aligned_df.merge(daily_price_df[['统一日前']], left_index=True, right_index=True, how='left')
            aligned_df = aligned_df.rename(columns={'统一日前': 'actual_price'})

            # Merge weather features for each point
            for point_name, point_df in filtered_weather_df.groupby('full_point_name'):
                # Select relevant feature columns (excluding 'preTime', 'issue_time_utc_str', 'full_point_name', 'time_bj')
                feature_cols = [col for col in point_df.columns if col not in ['preTime', 'issue_time_utc_str', 'full_point_name', 'time_bj']]
                
                # Create a temporary DataFrame with time_bj as index and point-prefixed columns
                temp_df = point_df.set_index('time_bj')[feature_cols]
                temp_df = temp_df.add_prefix(f'{point_name}_')
                
                aligned_df = aligned_df.merge(temp_df, left_index=True, right_index=True, how='left')
            
            # Add issue_time_utc_str as a column
            aligned_df['issue_time_utc_str'] = issue_time_utc_str
            aligned_data_list.append(aligned_df)

        except Exception as e:
            print(f"Error aligning data for {issue_time_utc_str}: {e}")
            continue
    
    if not aligned_data_list:
        return pd.DataFrame()

    final_aligned_df = pd.concat(aligned_data_list)
    return final_aligned_df.reset_index().rename(columns={'index': 'datetime_bj'})