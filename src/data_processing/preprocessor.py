import pandas as pd
from typing import NewType
from tqdm import tqdm
import numpy as np

DataFrame = NewType('DataFrame', pd.DataFrame)

def generate_day_ahead_price_demo(start_date: str, end_date: str) -> DataFrame:
    """
    Generates demo day-ahead price data for a given date range.
    The prices are simulated to have some daily and hourly patterns.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    all_data = []

    for single_date in tqdm(dates, desc="Generating day-ahead prices"):
        # Generate 96 data points for 24 hours (15-minute intervals)
        time_index = pd.date_range(start=single_date.strftime('%Y-%m-%d 00:00:00'),
                                   periods=96, freq='15min')
        
        # Simulate a daily pattern (e.g., higher prices during peak hours)
        # Base price around 200-300
        base_price = np.random.uniform(200, 300, 96)
        
        # Add hourly fluctuations
        hour_of_day = time_index.hour + time_index.minute / 60
        # Peak around 10-12 and 18-20
        hourly_pattern = 50 * (np.sin(hour_of_day / 24 * 2 * np.pi - np.pi/2) + 1) # roughly 0 to 100
        
        # Add some random noise
        noise = np.random.normal(0, 10, 96)
        
        day_ahead_price = base_price + hourly_pattern + noise
        
        # Clip prices to be within a reasonable range, e.g., 50 to 500
        day_ahead_price = np.clip(day_ahead_price, 50, 500)

        df_day = pd.DataFrame({'day_ahead_price': day_ahead_price}, index=time_index)
        all_data.append(df_day)
    
    return pd.concat(all_data)
