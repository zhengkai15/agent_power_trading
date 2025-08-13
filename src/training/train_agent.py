import os
import numpy as np
from tqdm import tqdm
import datetime # 导入 datetime 模块
import pandas as pd # 导入 pandas 用于保存损失/奖励曲线
import matplotlib.pyplot as plt # 导入 matplotlib

from src.agent.trading_agent import DQNAgent
from src.agent.environment import PowerTradingEnv
from src.utils.custom_logger import log

def plot_rewards_curve(rewards_file_path, output_dir):
    """
    绘制RL训练奖励曲线并保存。
    """
    log.info(f"开始绘制RL训练奖励曲线 from {rewards_file_path}...")
    try:
        rewards_df = pd.read_csv(rewards_file_path)
    except FileNotFoundError:
        log.error(f"错误: 未找到奖励文件 at {rewards_file_path}。无法绘制曲线。")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(rewards_df['episode'], rewards_df['reward'], label='Episode Reward')
    plt.title('RL Agent Training Reward Curve')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    
    plot_output_path = os.path.join(output_dir, 'training_reward_curve.png')
    plt.savefig(plot_output_path)
    plt.close() # 关闭图形，释放内存
    log.info(f"RL训练奖励曲线已保存到: {plot_output_path}")

def train_rl_agent():
    """
    Main function to train the RL agent.
    """
    log.info("Starting RL agent training process...")

    # Define paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    processed_data_path = os.path.join(base_dir, 'data', 'processed', 'aggregated_data.csv') # 修正路径，使用 aggregated_data.csv
    
    # 创建实验子文件夹
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, 'aexp', f'rl_exp_{timestamp}')
    os.makedirs(experiment_dir, exist_ok=True)
    log.info(f"为本次RL实验创建了目录: {experiment_dir}")

    agent_model_output_path = os.path.join(experiment_dir, 'trading_agent.pth')

    # 确保模型输出目录存在 (已由 experiment_dir 创建)
    # os.makedirs(os.path.dirname(agent_model_output_path), exist_ok=True)

    # 1. Load processed data
    log.info(f"从 {processed_data_path} 加载处理过的数据...")
    try:
        df = pd.read_csv(processed_data_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        log.error(f"错误: 未找到处理过的数据文件 at {processed_data_path}。请先运行数据处理脚本。")
        return

    if df.empty:
        log.error("数据为空，RL训练中止。")
        return

    # 2. Initialize environment
    try:
        env = PowerTradingEnv(data_df=df)
    except ValueError as e:
        log.error(f"Error initializing environment: {e}")
        return

    # 2. Initialize agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n # For discrete action space
    agent = DQNAgent(state_dim, action_dim)

    # 3. Training loop
    num_episodes = 100 # 增加训练回合数以更好地观察奖励曲线
    batch_size = 64
    episode_rewards = [] # 存储每个 episode 的总奖励

    for episode in tqdm(range(num_episodes), desc="Training RL Agent"):
        state, _ = env.reset() # env.reset() now returns (observation, info)
        episode_reward = 0
        done = False
        truncated = False
        
        while not done and not truncated:
            action = agent.act(state) # Agent selects action
            # Clip action to environment's action space bounds (not needed for discrete actions)

            next_state, reward, done, truncated, _ = env.step(action) # env.step() returns 5 values
            
            # Store experience in replay buffer
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward

            # Train agent if enough samples in replay buffer
            if len(agent.replay_buffer) > batch_size:
                agent.learn()
            
        
        episode_rewards.append(episode_reward) # 记录当前 episode 的总奖励
        log.info(f"Episode {episode + 1}/{num_episodes}, Total Reward: {episode_reward:.2f}")
        # agent.noise.reset() # Reset noise for new episode (not needed for DQN)
        agent.update_target_network() # Update target network after each episode

    # 4. 保存代理模型
    agent.save(agent_model_output_path)
    log.info(f"RL代理模型已保存到: {agent_model_output_path}")

    # 5. 保存奖励曲线
    rewards_df = pd.DataFrame({'episode': range(1, len(episode_rewards) + 1), 'reward': episode_rewards})
    rewards_output_path = os.path.join(experiment_dir, 'episode_rewards.csv')
    rewards_df.to_csv(rewards_output_path, index=False)
    log.info(f"RL代理训练奖励已保存到: {rewards_output_path}")

    # 绘制奖励曲线
    plot_rewards_curve(rewards_output_path, experiment_dir)

    log.info("RL agent training process finished.")

if __name__ == '__main__':
    train_rl_agent()
