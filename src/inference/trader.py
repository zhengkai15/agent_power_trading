import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.agent.trading_agent import TradingAgent
from src.agent.environment import TradingEnv
from src.model.price_predictor import PricePredictor # Import PricePredictor
from src.utils.logging import log
from src.inference.predictor import run_price_prediction # Import the price prediction function

def run_full_trading_simulation(data_path, agent_model_path, price_predictor_model_path, scaler_path):
    """
    Runs a full trading simulation using the trained RL agent and price predictor.

    Args:
        data_path (str): Path to the final_dataset.csv.
        agent_model_path (str): Path to the trained trading agent model (.pth).
        price_predictor_model_path (str): Path to the trained price predictor model (.pth).
        scaler_path (str): Path to the scaler (.pkl).

    Returns:
        pd.DataFrame: A DataFrame containing the simulation results (actions, rewards, prices, etc.).
    """
    log.info("Starting full trading simulation...")

    # 1. Initialize environment
    try:
        env = TradingEnv(data_path=data_path)
    except ValueError as e:
        log.error(f"Error initializing environment: {e}")
        return pd.DataFrame()

    # 2. Initialize agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    agent = TradingAgent(state_dim, action_dim, max_action)
    
    if not os.path.exists(agent_model_path):
        log.error(f"Error: Trading agent model not found at {agent_model_path}. Please train the agent first.")
        return pd.DataFrame()
    agent.load(agent_model_path)

    # 3. Initialize price predictor (optional, for logging/comparison)
    # The agent directly uses environment features, but we can still run the predictor
    # to see its output alongside the agent's decision.
    # For the agent's decision, it uses the raw features from the environment.
    # The price predictor is for evaluating price prediction accuracy.
    
    # We need to pass the full dataframe to run_price_prediction to get predictions for all steps
    full_df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    predicted_prices = run_price_prediction(full_df, sequence_length=24) # Assuming 24 for sequence length
    
    if predicted_prices is None:
        log.error("Price prediction failed. Exiting simulation.")
        return pd.DataFrame()

    # Align predicted prices with the original dataframe's index
    # The predictions are for the *end* of each sequence. 
    # If X_seq_inference[i] predicts for data_df.iloc[i + sequence_length - 1],
    # then predicted_prices[i] corresponds to data_df.index[i + sequence_length - 1]
    
    # Create a Series for predicted prices, aligning with the correct time index
    # The first (sequence_length - 1) predictions are not available, so pad with NaN
    predicted_prices_series = pd.Series(np.nan, index=full_df.index)
    # Fill from the point where predictions start
    predicted_prices_series.iloc[24-1:24-1+len(predicted_prices)] = predicted_prices.flatten() # Assuming sequence_length=24

    simulation_results = []
    state, _ = env.reset()
    total_reward = 0

    # Use tqdm for the simulation loop
    for step in tqdm(range(env.total_steps), desc="Running Trading Simulation"):
        if step < env.observation_space.shape[0]: # Skip initial steps if observation requires a full sequence
            # For the first few steps, the observation might not be a full sequence
            # The agent expects a full state_dim observation.
            # If the environment provides a partial observation, the agent might fail.
            # For now, let's assume env.reset() and env.step() always provide full state_dim.
            pass

        action = agent.select_action(state, add_noise=False) # No noise during inference
        action = np.clip(action, env.action_space.low[0], env.action_space.high[0])

        next_state, reward, terminated, truncated, info = env.step(action)
        
        # Record results for this step
        current_time_index = env.df.index[step]
        result = {
            "time": current_time_index,
            "real_price": info["real_price"],
            "day_ahead_price": info["day_ahead_price"],
            "predicted_price_by_model": predicted_prices_series.loc[current_time_index], # Predicted by price_predictor
            "bid_price_by_agent": info["bid_price"],
            "reward": reward,
            "cumulative_reward": total_reward + reward, # Calculate cumulative reward
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

    Args:
        results_df (pd.DataFrame): DataFrame containing simulation results.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    log.info("Starting evaluation of simulation results...")

    # 1. Price Prediction Evaluation: abs(pred-tgt)/((pred+target)/2) clip to 0~1
    # Filter out NaN values from predicted_price_by_model (due to sequence length)
    price_eval_df = results_df.dropna(subset=['predicted_price_by_model']).copy()
    
    if not price_eval_df.empty:
        y_pred_price = price_eval_df['predicted_price_by_model'].values
        y_true_price = price_eval_df['real_price'].values

        numerator = np.abs(y_pred_price - y_true_price)
        denominator = (y_pred_price + y_true_price) / 2
        # Avoid division by zero
        denominator[denominator == 0] = 1e-6
        
        mape_clipped = numerator / denominator
        mape_clipped = np.clip(mape_clipped, 0, 1)
        avg_price_prediction_error = np.mean(mape_clipped)
        log.info(f"Average Price Prediction Error (clipped 0-1): {avg_price_prediction_error:.4f}")
    else:
        avg_price_prediction_error = np.nan
        log.warning("No valid price predictions to evaluate.")

    # 2. Trading Strategy Evaluation: Daily cumulative reward
    # The 'reward' column already contains the per-step reward.
    # We need to sum rewards per day.
    results_df['date'] = results_df.index.date
    daily_rewards = results_df.groupby('date')['reward'].sum()
    
    total_cumulative_reward = results_df['reward'].sum()
    log.info(f"Total Cumulative Trading Reward: {total_cumulative_reward:.2f}")
    log.info("Daily Trading Rewards (first 5 days):\n" + str(daily_rewards.head()))

    evaluation_metrics = {
        "avg_price_prediction_error": avg_price_prediction_error,
        "total_cumulative_trading_reward": total_cumulative_reward,
        "daily_trading_rewards": daily_rewards.to_dict() # Convert to dict for easier logging/return
    }
    log.info("Evaluation finished.")
    return evaluation_metrics

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    processed_data_path = os.path.join(base_dir, 'data', 'aexp', 'final_dataset.csv')
    agent_model_path = os.path.join(base_dir, 'models', 'trading_agent.pth')
    price_predictor_model_path = os.path.join(base_dir, 'models', 'price_predictor.pth')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
    results_output_path = os.path.join(base_dir, 'data', 'aexp', 'simulation_results.csv')

    # Run simulation
    results_df = run_full_trading_simulation(processed_data_path, agent_model_path, price_predictor_model_path, scaler_path)
    
    if not results_df.empty:
        print("\n--- Simulation Results Head ---")
        print(results_df.head())
        print("\n--- Simulation Results Info ---")
        results_df.info()
        # Save results
        results_df.to_csv(results_output_path)
        log.info(f"Simulation results saved to {results_output_path}")

        # Evaluate results
        evaluation_metrics = evaluate_results(results_df)
        print("\n--- Evaluation Metrics ---")
        for metric, value in evaluation_metrics.items():
            if metric != "daily_trading_rewards": # Print daily rewards separately if needed
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: (see log for details)")
    else:
        log.error("Simulation results are empty. Cannot perform evaluation.")
