import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import joblib
from sklearn.model_selection import train_test_split
import datetime # 导入 datetime 模块
import matplotlib.pyplot as plt # 导入 matplotlib

from src.model.price_predictor import PricePredictor
from src.utils.custom_logger import log

def plot_training_losses(losses_file_path, output_dir):
    """
    绘制训练损失曲线并保存。
    """
    log.info(f"开始绘制训练损失曲线 from {losses_file_path}...")
    try:
        losses_df = pd.read_csv(losses_file_path)
    except FileNotFoundError:
        log.error(f"错误: 未找到损失文件 at {losses_file_path}。无法绘制曲线。")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(losses_df['epoch'], losses_df['loss'], label='Training Loss')
    plt.title('Price Predictor Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plot_output_path = os.path.join(output_dir, 'training_loss_curve.png')
    plt.savefig(plot_output_path)
    plt.close() # 关闭图形，释放内存
    log.info(f"训练损失曲线已保存到: {plot_output_path}")

def evaluate(y_true, y_pred):
    """
    根据定义的指标评估模型。
    指标: abs(pred-tgt)/((pred+target)/2) 裁剪到 0-1
    """
    # 确保 y_true 和 y_pred 是一维的
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    numerator = np.abs(y_pred - y_true)
    denominator = (y_pred + y_true) / 2
    denominator[denominator == 0] = 1e-6
    
    mape = numerator / denominator
    mape = np.clip(mape, 0, 1)
    
    return mape.mean()

def train_price_predictor():
    """
    训练电价预测模型的主函数。
    """
    log.info("开始电价预测器训练流程...")

    # 定义路径
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    processed_data_path = os.path.join(base_dir, 'data', 'processed', 'aggregated_data.csv')
    
    # 创建实验子文件夹
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, 'aexp', f'exp_{timestamp}')
    os.makedirs(experiment_dir, exist_ok=True)
    log.info(f"为本次实验创建了目录: {experiment_dir}")

    model_output_path = os.path.join(experiment_dir, 'price_predictor.pth') # PyTorch 模型扩展名
    # scaler_output_path = os.path.join(experiment_dir, 'scaler.pkl') # 如果需要保存 scaler，也保存到实验目录

    # 确保模型输出目录存在 (已由 experiment_dir 创建)
    # os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    # 1. 加载处理过的数据
    log.info(f"从 {processed_data_path} 加载处理过的数据...")
    try:
        df = pd.read_csv(processed_data_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        log.error(f"错误: 未找到处理过的数据文件 at {processed_data_path}。请先运行数据处理脚本。")
        return

    if df.empty:
        log.error("数据为空，训练中止。")
        return

    # 2. 特征和目标分离
    log.info("分离特征和目标变量...")
    y = df[['actual_price']]
    X = df.drop(columns=['actual_price', 'issue_time_utc_str'])

    # 3. 特征工程 (暂时跳过，根据用户要求)
    # log.info("进行特征工程：时间特征、滑窗特征、Shift特征...")

    # 时间特征
    # X['hour'] = X.index.hour
    # X['minute'] = X.index.minute
    # X['dayofweek'] = X.index.dayofweek
    # X['dayofyear'] = X.index.dayofyear
    # X['month'] = X.index.month
    # X['quarter'] = X.index.quarter
    # X['year'] = X.index.year
    # X['weekofyear'] = X.index.isocalendar().week.astype(int) # 使用 isocalendar().week

    # 滑窗特征 (以 'day_ahead_price' 为例，也可以对其他特征做)
    # 确保 'day_ahead_price' 存在且是数值类型
    # if 'day_ahead_price' in X.columns:
    #     X['day_ahead_price_roll_mean_4'] = X['day_ahead_price'].rolling(window=4).mean()
    #     X['day_ahead_price_roll_std_4'] = X['day_ahead_price'].rolling(window=4).std()
    #     X['day_ahead_price_roll_mean_24'] = X['day_ahead_price'].rolling(window=24).mean()
    #     X['day_ahead_price_roll_std_24'] = X['day_ahead_price'].rolling(window=24).std()
    
    # Shift特征 (以 'day_ahead_price' 为例，也可以对其他特征做)
    # if 'day_ahead_price' in X.columns:
    #     X['day_ahead_price_lag_1'] = X['day_ahead_price'].shift(1)
    #     X['day_ahead_price_lag_24'] = X['day_ahead_price'].shift(24) # 24个15分钟点是6小时前

    # 填充 NaN 值，因为滑窗和shift操作会产生NaN
    X = X.fillna(method='bfill').fillna(method='ffill') # 向后填充，再向前填充

    # 转换为 numpy 数组
    X = X.values
    y = y.values

    # 检查 NaN 或无穷大值
    if np.isnan(X).any() or np.isinf(X).any():
        log.warning("特征数据中存在 NaN 或无穷大值，尝试处理...")
        X = np.nan_to_num(X, nan=0.0, posinf=1e5, neginf=-1e5) # 将 NaN 替换为 0，无穷大替换为大/小值

    if np.isnan(y).any() or np.isinf(y).any():
        log.warning("目标数据中存在 NaN 或无穷大值，尝试处理...")
        y = np.nan_to_num(y, nan=0.0, posinf=1e5, neginf=-1e5)

    # 3. 特征缩放 (已跳过，根据用户要求)

    # 4. 创建序列数据
    log.info("创建序列数据...")
    sequence_length = 24 # 15分钟粒度，24个点是6小时

    def create_sequences(data, seq_length):
        xs = []
        ys = []
        for i in range(len(data) - seq_length):
            x = data[i:(i + seq_length)]
            y = data[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    # 对于预测，我们希望预测下一个时间步的 'price'，所以 y 应该与 X 的最后一个时间步对齐
    # 这里的 y 已经是 price，所以我们需要调整 create_sequences 来处理 X 和 y
    # 重新定义 create_sequences 以便它接受 X 和 y 并返回 X_seq, y_seq
    def create_sequences_for_lstm(X_data, y_data, seq_length):
        X_seq, y_seq = [], []
        for i in range(len(X_data) - seq_length):
            X_seq.append(X_data[i:(i + seq_length)])
            y_seq.append(y_data[i + seq_length]) # 预测序列后的下一个价格
        return np.array(X_seq), np.array(y_seq)

    X_seq, y_seq = create_sequences_for_lstm(X, y, sequence_length)

    # 5. 划分训练集和测试集
    log.info("划分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, shuffle=False) # LSTM数据通常不shuffle
    
    log.info(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

    # 6. 训练模型
    # input_dim 现在是每个时间步的特征数量
    input_dim = X_train.shape[2]
    predictor = PricePredictor(input_dim=input_dim)
    
    # 训练模型并获取损失
    log.info("开始训练电价预测模型...")
    training_losses = predictor._train_loop(X_train, y_train, num_epochs=10)
    log.info("电价预测模型训练完成。")

    # 7. 保存损失曲线
    losses_df = pd.DataFrame({'epoch': range(1, len(training_losses) + 1), 'loss': training_losses})
    losses_output_path = os.path.join(experiment_dir, 'training_losses.csv')
    losses_df.to_csv(losses_output_path, index=False)
    log.info(f"训练损失已保存到: {losses_output_path}")

    # 绘制损失曲线
    plot_training_losses(losses_output_path, experiment_dir)

    # 8. 评估模型
    log.info("在测试集上评估模型...")
    predictions = predictor.predict(X_test)
    score = evaluate(y_test, predictions)
    log.info(f"模型在测试集上的评估分数: {score:.4f}")

    # 9. 保存模型
    predictor.save(model_output_path)

    log.info("电价预测器训练流程完成。")

if __name__ == '__main__':
    train_price_predictor()