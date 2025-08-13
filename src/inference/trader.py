import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

from src.agent.trading_agent import DQNAgent
from src.agent.environment import PowerTradingEnv
from src.model.price_predictor import PricePredictor # Import PricePredictor
from src.utils.custom_logger import log
from src.inference.predictor import run_price_prediction # Import the price prediction function

def run_full_trading_simulation(data_path, agent_model_path, price_predictor_model_path):
    """
    Runs a full trading simulation using the trained RL agent and price predictor.

    Args:
        data_path (str): Path to the final_dataset.csv.
        agent_model_path (str): Path to the trained trading agent model (.pth).
        price_predictor_model_path (str): Path to the trained price predictor model (.pth).

    Returns:
        pd.DataFrame: A DataFrame containing the simulation results (actions, rewards, prices, etc.).
    """
    log.info("Starting full trading simulation...")

    # 1. Initialize environment
    try:
        full_df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        env = PowerTradingEnv(data_df=full_df)
    except (FileNotFoundError, ValueError) as e:
        log.error(f"Error initializing environment from {data_path}: {e}")
        return None

    # 2. Initialize agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n # For discrete action space
    agent = DQNAgent(state_dim, action_dim)
    
    if not os.path.exists(agent_model_path):
        log.error(f"Error: Trading agent model not found at {agent_model_path}. Please train the agent first.")
        return None
    agent.load(agent_model_path)
    log.info(f"Successfully loaded trading agent from {agent_model_path}")

    # 3. Run price prediction
    predicted_prices = run_price_prediction(full_df, price_predictor_model_path, sequence_length=24)
    
    if predicted_prices is None:
        log.error("Price prediction failed. Exiting simulation.")
        return None
    log.info(f"Successfully ran price prediction using model from {price_predictor_model_path}")

    # Align predicted prices with the original dataframe's index
    predicted_prices_series = pd.Series(np.nan, index=full_df.index)
    pred_start_index = 24 - 1
    pred_end_index = pred_start_index + len(predicted_prices)
    predicted_prices_series.iloc[pred_start_index:pred_end_index] = predicted_prices.flatten()

    simulation_results = []
    state, _ = env.reset()
    total_reward = 0

    for step in tqdm(range(len(env.data_df)), desc="Running Trading Simulation"):
        action = agent.act(state)
        next_state, reward, terminated, truncated, info = env.step(action) 
        
        current_time_index = env.data_df.index[step]
        result = {
            "time": current_time_index,
            "real_price": info["actual_price"],
            "day_ahead_price": env.data_df.iloc[step]['day_ahead_price'],
            "predicted_price_by_model": predicted_prices_series.loc[current_time_index],
            "bid_price_by_agent": info["bid_price"],
            "reward": reward,
            "cumulative_reward": total_reward + reward,
            "reward_flag": info["reward_flag"]
        }
        simulation_results.append(result)
        
        state = next_state
        total_reward += reward

        if terminated or truncated:
            break

    log.info(f"Full trading simulation finished. Total cumulative reward: {total_reward:.2f}")
    return pd.DataFrame(simulation_results).set_index("time")

def evaluate_results(results_df):
    """
    Evaluates the price prediction and trading strategy based on the simulation results.
    """
    log.info("Starting evaluation of simulation results...")
    # ... (evaluation logic remains the same)
    price_eval_df = results_df.dropna(subset=['predicted_price_by_model']).copy()
    
    if not price_eval_df.empty:
        y_pred_price = price_eval_df['predicted_price_by_model'].values
        y_true_price = price_eval_df['real_price'].values
        numerator = np.abs(y_pred_price - y_true_price)
        denominator = (y_pred_price + y_true_price) / 2
        denominator[denominator == 0] = 1e-6
        mape_clipped = np.clip(numerator / denominator, 0, 1)
        avg_price_prediction_error = np.mean(mape_clipped)
        log.info(f"Average Price Prediction Error (clipped 0-1): {avg_price_prediction_error:.4f}")
    else:
        avg_price_prediction_error = np.nan
        log.warning("No valid price predictions to evaluate.")

    results_df['date'] = results_df.index.date
    daily_rewards = results_df.groupby('date')['reward'].sum()
    total_cumulative_reward = results_df['reward'].sum()
    log.info(f"Total Cumulative Trading Reward: {total_cumulative_reward:.2f}")

    evaluation_metrics = {
        "avg_price_prediction_error": avg_price_prediction_error,
        "total_cumulative_trading_reward": total_cumulative_reward,
        "daily_trading_rewards": daily_rewards.to_dict()
    }
    log.info("Evaluation finished.")
    return evaluation_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run full trading simulation using experiment paths.")
    parser.add_argument('--agent_exp_path', type=str, required=True, help='Path to the trading agent experiment directory.')
    parser.add_argument('--price_exp_path', type=str, required=True, help='Path to the price predictor experiment directory.')
    args = parser.parse_args()

    # Construct full model paths from experiment paths
    agent_model_path = os.path.join(args.agent_exp_path, 'trading_agent.pth')
    price_predictor_model_path = os.path.join(args.price_exp_path, 'price_predictor.pth')

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    processed_data_path = os.path.join(base_dir, 'data', 'processed', 'aggregated_data.csv')
    results_output_path = os.path.join(base_dir, 'logs', 'simulation_results.csv')

    results_df = run_full_trading_simulation(processed_data_path, agent_model_path, price_predictor_model_path)
    
    if results_df is None or results_df.empty:
        log.error("Simulation failed or produced no results. Exiting.")
        exit(1)

    print("\n--- Simulation Results (First 5 Rows) ---")
    print(results_df.head())
    
    results_df.to_csv(results_output_path)
    log.info(f"Full simulation results saved to {results_output_path}")

    evaluation_metrics = evaluate_results(results_df)
    print("\n--- Evaluation Metrics ---")
    for metric, value in evaluation_metrics.items():
        if metric != "daily_trading_rewards":
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: (see log for details)")