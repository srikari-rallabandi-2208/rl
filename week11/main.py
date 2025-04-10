import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse


# Environment for RL
class PricePredictionEnv:
    def __init__(self, df, target_col, N=10, P_min=None, P_max=None):
        self.df = df
        self.target_col = target_col
        self.X = df.drop(columns=[target_col]).values
        self.y = df[target_col].values
        self.N = N  # Number of discrete actions
        self.P_min = P_min if P_min is not None else self.y.min()
        self.P_max = P_max if P_max is not None else self.y.max()
        self.action_space = np.arange(N)
        self.current_step = 0
        self.num_steps = len(df)
        self.done = False

    def reset(self):
        self.current_step = 0
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        return self.X[self.current_step] if self.current_step < self.num_steps else None

    def _action_to_prediction(self, action):
        return self.P_min + (self.P_max - self.P_min) * action / (self.N - 1)

    def step(self, action):
        true_val = self.y[self.current_step]
        predicted_val = self._action_to_prediction(action)
        reward = -(true_val - predicted_val) ** 2
        self.current_step += 1
        self.done = self.current_step >= self.num_steps
        next_obs = self._get_observation()
        return next_obs, reward, self.done, {}


# Policy Gradient Agent
class PolicyGradientAgent:
    def __init__(self, input_dim, action_dim, lr=0.001):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(action_dim, activation='softmax')
        ])
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.action_dim = action_dim

    def choose_action(self, obs, greedy=False):
        obs_tensor = tf.convert_to_tensor(obs, dtype=tf.float32)
        probs = self.model(obs_tensor[None, :]).numpy()[0]
        return np.argmax(probs) if greedy else np.random.choice(self.action_dim, p=probs)

    def train(self, obs, action, reward):
        with tf.GradientTape() as tape:
            probs = self.model(obs[None, :])
            action_one_hot = tf.one_hot([action], self.action_dim)
            log_prob = tf.reduce_sum(tf.math.log(probs + 1e-10) * action_one_hot)
            loss = -log_prob * reward
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


# Q-Learning Agent
class QLearningAgent:
    def __init__(self, input_dim, action_dim, lr=0.001, gamma=0.99):
        self.q_model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(action_dim)
        ])
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.gamma = gamma

    def choose_action(self, obs):
        obs_tensor = tf.convert_to_tensor(obs, dtype=tf.float32)
        q_values = self.q_model(obs_tensor[None, :]).numpy()[0]
        return np.argmax(q_values)

    def train(self, obs, action, reward, next_obs, done):
        obs_tensor = tf.convert_to_tensor(obs, dtype=tf.float32)
        next_obs_tensor = tf.convert_to_tensor(next_obs, dtype=tf.float32) if next_obs is not None else None
        with tf.GradientTape() as tape:
            q_values = self.q_model(obs_tensor[None, :])
            q_next = self.q_model(next_obs_tensor[None, :]) if not done else tf.zeros_like(q_values)
            target = reward + self.gamma * tf.reduce_max(q_next) * (1 - int(done))
            action_one_hot = tf.one_hot([action], q_values.shape[1])
            q_action = tf.reduce_sum(q_values * action_one_hot, axis=1)
            loss = tf.reduce_mean((target - q_action) ** 2)
        grads = tape.gradient(loss, self.q_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_model.trainable_variables))
        return loss


# Utility Functions
def split_dataframe(df, train_ratio=0.8):
    train_size = int(len(df) * train_ratio)
    return df[:train_size], df[train_size:]


def plot_prediction_scatter(y_true, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('True Prices')
    plt.ylabel('Predicted Prices')
    plt.title(title)
    plt.show()


def evaluate_rl_agent(agent, env, is_policy_gradient=True, greedy=True):
    obs = env.reset()
    y_true, y_pred = [], []
    while not env.done:
        if is_policy_gradient:
            action = agent.choose_action(obs, greedy=greedy)
        else:
            action = agent.choose_action(obs)
        predicted_price = env._action_to_prediction(action)
        true_price = env.y[env.current_step]
        y_true.append(true_price)
        y_pred.append(predicted_price)
        obs, _, _, _ = env.step(action)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    print(f"[RL Evaluation] MSE: {mse:.4f}, RMSE: {rmse:.4f}")
    return y_true, y_pred


# Main Functions
def run_supervised(csv_path, save_model_path=None):
    df = pd.read_csv(csv_path)
    target_col = 'Estimated_Price'
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    train_size = int(0.8 * len(df))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

    if save_model_path:
        model.save(save_model_path)

    y_pred = model.predict(X_test).flatten()
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    print(f"[Supervised] Test MSE: {mse:.4f}, RMSE: {rmse:.4f}")
    plot_prediction_scatter(y_test, y_pred, 'Supervised Learning: True vs Predicted Prices')


def run_policy_gradient(csv_path, save_model_path=None, N=10, episodes=50):
    df = pd.read_csv(csv_path)
    target_col = 'Estimated_Price'
    train_df, test_df = split_dataframe(df)
    P_min, P_max = train_df[target_col].min(), train_df[target_col].max()
    train_env = PricePredictionEnv(train_df, target_col, N, P_min, P_max)
    test_env = PricePredictionEnv(test_df, target_col, N, P_min, P_max)
    agent = PolicyGradientAgent(input_dim=train_env.X.shape[1], action_dim=N)

    for ep in range(episodes):
        obs = train_env.reset()
        total_reward = 0
        while not train_env.done:
            action = agent.choose_action(obs)
            next_obs, reward, done, _ = train_env.step(action)
            agent.train(obs, action, reward)
            obs = next_obs
            total_reward += reward
        print(f"[Policy Gradient] Episode {ep + 1}/{episodes}, Total Reward: {total_reward:.4f}")

    if save_model_path:
        agent.model.save(save_model_path)

    y_true, y_pred = evaluate_rl_agent(agent, test_env, is_policy_gradient=True)
    plot_prediction_scatter(y_true, y_pred, 'Policy Gradient: True vs Predicted Prices')


def run_q_learning(csv_path, save_model_path=None, N=10, episodes=50):
    df = pd.read_csv(csv_path)
    target_col = 'Estimated_Price'
    train_df, test_df = split_dataframe(df)
    P_min, P_max = train_df[target_col].min(), train_df[target_col].max()
    train_env = PricePredictionEnv(train_df, target_col, N, P_min, P_max)
    test_env = PricePredictionEnv(test_df, target_col, N, P_min, P_max)
    agent = QLearningAgent(input_dim=train_env.X.shape[1], action_dim=N)

    for ep in range(episodes):
        obs = train_env.reset()
        total_reward = 0
        while not train_env.done:
            action = agent.choose_action(obs)
            next_obs, reward, done, _ = train_env.step(action)
            agent.train(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward
        print(f"[Q-Learning] Episode {ep + 1}/{episodes}, Total Reward: {total_reward:.4f}")

    if save_model_path:
        agent.q_model.save(save_model_path)

    y_true, y_pred = evaluate_rl_agent(agent, test_env, is_policy_gradient=False)
    plot_prediction_scatter(y_true, y_pred, 'Q-Learning: True vs Predicted Prices')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['supervised', 'policy_grad', 'q_learning'], required=True)
    parser.add_argument('--csv_path', type=str, required=True)
    args = parser.parse_args()

    if args.mode == 'supervised':
        run_supervised(args.csv_path, 'supervised_model')
    elif args.mode == 'policy_grad':
        run_policy_gradient(args.csv_path, 'policy_model', N=10, episodes=50)
    elif args.mode == 'q_learning':
        run_q_learning(args.csv_path, 'q_model', N=10, episodes=50)