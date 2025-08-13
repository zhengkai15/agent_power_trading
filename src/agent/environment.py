import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class PowerTradingEnv(gym.Env):
    """
    A custom Gymnasium environment for simulating electricity power trading.
    The agent's goal is to maximize trading profit based on price predictions.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, data_df: pd.DataFrame, initial_power_units: float = 100.0, render_mode=None):
        super().__init__()
        self.data_df = data_df.copy()
        self.initial_power_units = initial_power_units
        self.current_power_units = initial_power_units
        self.current_step = 0
        self.total_profit = 0.0
        self.current_day = None # Track current day for daily profit calculation
        self.daily_profit = 0.0 # Accumulate profit for the current day

        # Define action and observation space
        # Action: [bid_price_ratio, trade_amount_ratio]
        # bid_price_ratio: A ratio to adjust the predicted price for bidding (e.g., 0.9 to 1.1)
        # trade_amount_ratio: A ratio of current_power_units to trade (e.g., -1 for sell all, 1 for buy all)
        # For simplicity, let's define discrete actions for now:
        # 0: Hold (do nothing)
        # 1: Bid low, sell some (e.g., bid at pred_price * 0.9, sell 10 units)
        # 2: Bid high, buy some (e.g., bid at pred_price * 1.1, buy 10 units)
        # 3: Bid low, sell more (e.g., bid at pred_price * 0.8, sell 20 units)
        # 4: Bid high, buy more (e.g., bid at pred_price * 1.2, buy 20 units)
        self.action_space = spaces.Discrete(5) # 5 discrete actions

        # Observation space: [current_power_units, current_profit, actual_price, day_ahead_price, predicted_price, time_features...]
        # We need to dynamically determine the observation space size based on data_df columns
        # Exclude 'issue_time_utc_str' and 'datetime_bj' (if it's an index)
        state_features = [col for col in data_df.columns if col not in ['issue_time_utc_str', 'actual_price']]
        self.observation_features = ['current_power_units', 'current_profit'] + state_features
        
        # Determine the bounds for observation space
        # For simplicity, let's use a large range for now. In a real scenario, these should be carefully defined.
        low = np.array([-np.inf] * len(self.observation_features), dtype=np.float32)
        high = np.array([np.inf] * len(self.observation_features), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        # Get current row of data
        current_row = self.data_df.iloc[self.current_step]
        
        # Extract relevant features for observation
        # We need to ensure 'day_ahead_price' is present in the data_df
        # And 'predicted_price' will be provided by the predictor during inference/training
        # For now, let's assume 'day_ahead_price' is a feature and 'predicted_price' will be added later.
        
        # Create a dictionary for the current state
        state_dict = {
            'current_power_units': self.current_power_units,
            'current_profit': self.total_profit,
        }
        
        # Add all other relevant features from the current row
        for feature in self.observation_features[2:]:
            state_dict[feature] = current_row[feature]
            
        return np.array(list(state_dict.values()), dtype=np.float32)

    def _get_info(self):
        return {
            "current_step": self.current_step,
            "total_profit": self.total_profit,
            "current_power_units": self.current_power_units,
            "actual_price": self.data_df.iloc[self.current_step]['actual_price']
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_power_units = self.initial_power_units
        self.current_step = 0
        self.total_profit = 0.0
        self.current_day = self.data_df.index[self.current_step].date() # Initialize current day
        self.daily_profit = 0.0 # Reset daily profit

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        done = False # Initialize done to False at the beginning of the step function
        # Check if a new day has started
        current_date = self.data_df.index[self.current_step].date()
        if current_date != self.current_day:
            # If it's a new day, the reward for the previous day is the daily_profit
            reward = self.daily_profit
            self.total_profit += self.daily_profit # Add daily profit to total profit
            self.daily_profit = 0.0 # Reset daily profit for the new day
            self.current_day = current_date # Update current day
        else:
            reward = 0.0 # Default reward for intermediate steps

        current_actual_price = self.data_df.iloc[self.current_step]['actual_price']
        current_day_ahead_price = self.data_df.iloc[self.current_step]['day_ahead_price']

        # For simplicity, let's assume a fixed predicted price for now, 
        # which will be replaced by the actual model prediction during training/inference.
        # Let's use day_ahead_price as a proxy for predicted_price for initial environment setup.
        predicted_price = current_day_ahead_price # Placeholder

        # Define trading parameters based on action
        trade_amount = 0
        bid_price = predicted_price
        transaction_successful = False

        if action == 0: # Hold
            pass
        elif action == 1: # Bid low, sell some
            bid_price = predicted_price * 0.9
            trade_amount = -10 # Sell 10 units
        elif action == 2: # Bid high, buy some
            bid_price = predicted_price * 1.1
            trade_amount = 10 # Buy 10 units
        elif action == 3: # Bid low, sell more
            bid_price = predicted_price * 0.8
            trade_amount = -20 # Sell 20 units
        elif action == 4: # Bid high, buy more
            bid_price = predicted_price * 1.2
            trade_amount = 20 # Buy 20 units

        # Execute trade if trade_amount is not zero
        if trade_amount != 0:
            # Check if trade is successful based on bid_price vs actual_price and day_ahead_price
            # If (bid_price - day_ahead_price) and (actual_price - day_ahead_price) are same sign, trade is successful
            # This is based on the profit calculation rule: (挂单价格-日前价格) 和 （真实价格-日前价格）同号说明预测正确
            
            # Calculate profit if trade is successful
            if np.sign(bid_price - current_day_ahead_price) == np.sign(current_actual_price - current_day_ahead_price):
                transaction_successful = True
                # Profit calculation: 单位挂单电量 * （挂单价格-日前价格）* flag
                # Here, flag is 1 because it's successful
                profit_per_unit = (bid_price - current_actual_price)
                self.daily_profit += profit_per_unit * trade_amount # Accumulate daily profit
                self.current_power_units += trade_amount # Update power units based on trade
            else:
                # If prediction is wrong, profit is 0 for that trade based on the rule
                pass # No profit/loss if prediction is wrong, do not add to daily_profit

        self.current_step += 1

        if self.current_step >= len(self.data_df) -1:
            done = True
            # If the episode ends, add any remaining daily profit to the total reward
            if self.daily_profit != 0.0:
                reward += self.daily_profit # Add remaining daily profit as final reward
                self.total_profit += self.daily_profit # Add to total profit
                self.daily_profit = 0.0 # Reset for next episode

        observation = self._get_obs()
        info = self._get_info()
        info['bid_price'] = bid_price
        info['reward_flag'] = transaction_successful

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, done, False, info

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):
        # Implement rendering logic here if needed for visualization
        # For a CLI agent, this might just be printing current state
        pass

    def close(self):
        pass