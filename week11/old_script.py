import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


########################################################################
# 1. Data-Loading and Splitting
########################################################################

def load_data(csv_path):
    """
    Load CSV data from the given path into a Pandas DataFrame.

    :param csv_path: str
        Path to the dataset CSV file.
    :return: pd.DataFrame
        DataFrame containing the CSV data.
    """
    df = pd.read_csv(csv_path)
    return df


def create_tf_datasets(df, target_column="target_price", test_ratio=0.2, batch_size=64):
    """
    Create train/test tf.data.Dataset objects from a Pandas DataFrame.

    The function:
      1) Shuffles the entire DataFrame.
      2) Splits it into train and test sets.
      3) Converts the sets into tf.data.Dataset with (features, labels).

    :param df: pd.DataFrame
        Contains all data (including the target).
    :param target_column: str
        Name of the column to predict.
    :param test_ratio: float
        Fraction of data reserved for the test set.
    :param batch_size: int
        Number of samples per batch for both train and test Datasets.
    :return: (train_ds, test_ds)
        A tuple of tf.data.Dataset objects for training and testing.
    """
    # Shuffle DataFrame rows
    df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Separate features from target
    X = df_shuffled.drop(columns=[target_column]).values
    y = df_shuffled[target_column].values

    # Convert into a single tf.data.Dataset
    full_ds = tf.data.Dataset.from_tensor_slices((X, y))

    # Compute sizes
    total_size = len(df_shuffled)
    test_size = int(total_size * test_ratio)
    train_size = total_size - test_size

    # Split via take() and skip()
    train_ds = full_ds.take(train_size)
    test_ds = full_ds.skip(train_size)

    # Shuffle only training data
    train_ds = train_ds.shuffle(buffer_size=10000).batch(batch_size)
    test_ds = test_ds.batch(batch_size)

    return train_ds, test_ds


########################################################################
# 2. Supervised Learning with TensorFlow
########################################################################

def build_mlp_model(input_dim, hidden_units=[64, 64], learning_rate=0.001):
    """
    Build a simple MLP (Multi-Layer Perceptron) model using Keras.

    :param input_dim: int
        Dimension of input features.
    :param hidden_units: list of int
        Sizes of hidden layers.
    :param learning_rate: float
        Learning rate for the optimizer.
    :return: tf.keras.Model
        Compiled Keras model ready for training.
    """
    model = tf.keras.Sequential()
    # Input layer: shape=(input_dim,)
    model.add(tf.keras.Input(shape=(input_dim,)))

    # Hidden layers
    for units in hidden_units:
        model.add(tf.keras.layers.Dense(units, activation='relu'))

    # Output layer for regression (single value)
    model.add(tf.keras.layers.Dense(1, activation='linear'))

    # Compile with MSE loss (typical for regression)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )

    return model


def train_supervised(train_ds, test_ds, input_dim, epochs=5, hidden_units=[64, 64], lr=0.001):
    """
    Train the MLP model in a supervised manner.

    :param train_ds: tf.data.Dataset
        Dataset for training.
    :param test_ds: tf.data.Dataset
        Dataset for validation.
    :param input_dim: int
        Number of input features.
    :param epochs: int
        Number of training epochs.
    :param hidden_units: list of int
        Hidden layer sizes.
    :param lr: float
        Learning rate for Adam.
    :return: (model, history)
        The trained model and its Keras History object.
    """
    model = build_mlp_model(input_dim, hidden_units, learning_rate=lr)
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
        verbose=1
    )
    return model, history


def evaluate_supervised(model, test_ds):
    """
    Evaluate the trained model on the test set. Prints metrics and returns
    predictions & ground truth for further analysis.

    :param model: tf.keras.Model
        Trained model.
    :param test_ds: tf.data.Dataset
        Test dataset.
    :return: (y_true_all, y_pred_all)
        True labels and predicted values in numpy arrays.
    """
    # Evaluate returns loss and rmse
    loss, rmse = model.evaluate(test_ds, verbose=0)
    print(f"\n[Supervised] Test MSE: {loss:.4f}")
    print(f"[Supervised] Test RMSE: {rmse:.4f}")

    # Collect predictions
    y_pred_all = []
    y_true_all = []
    for X_batch, y_batch in test_ds:
        preds = model.predict(X_batch)
        y_pred_all.extend(preds.flatten().tolist())
        y_true_all.extend(y_batch.numpy().flatten().tolist())

    y_pred_all = np.array(y_pred_all)
    y_true_all = np.array(y_true_all)

    # Mean Percentage Error
    epsilon = 1e-10
    mperc_error = np.mean(np.abs(y_true_all - y_pred_all) / (np.abs(y_true_all) + epsilon)) * 100
    print(f"[Supervised] Mean Percentage Error: {mperc_error:.4f}%")

    # R^2 Score
    ss_res = np.sum((y_true_all - y_pred_all) ** 2)
    ss_tot = np.sum((y_true_all - np.mean(y_true_all)) ** 2) + epsilon
    r2 = 1 - (ss_res / ss_tot)
    print(f"[Supervised] R^2 Score: {r2:.4f}")

    return y_true_all, y_pred_all


def save_supervised_model(model, save_path="trained_model"):
    """
    Save the trained model in TF SavedModel format.

    :param model: tf.keras.Model
        Model to save.
    :param save_path: str
        Directory path to save the model.
    """
    model.save(save_path)
    print(f"[Supervised] Model saved to: {save_path}")


def load_supervised_model(save_path="trained_model"):
    """
    Load a trained model from disk.

    :param save_path: str
        Directory path where the model is saved.
    :return: tf.keras.Model
        Loaded Keras model.
    """
    model = tf.keras.models.load_model(save_path)
    print(f"[Supervised] Model loaded from: {save_path}")
    return model


########################################################################
# 3. Plotting Functions
########################################################################

def plot_training_history(history):
    """
    Plot the training & validation loss (MSE) over epochs.

    :param history: tf.keras.callbacks.History
        Keras History object with training stats.
    """
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training Loss vs. Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()


def plot_prediction_scatter(y_true, y_pred, title='Predicted vs. Actual'):
    """
    Scatter plot of predicted vs. actual values.

    :param y_true: np.array
        Ground truth values.
    :param y_pred: np.array
        Model predictions.
    :param title: str
        Title of the plot.
    """
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(title)

    # Plot y=x line for reference
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'k--')
    plt.show()


def plot_returns(returns, title='Episode Return'):
    """
    Plot the returns (sum of rewards) per episode.

    :param returns: list or np.array
        Episode returns to plot.
    :param title: str
        Title of the plot.
    """
    plt.figure()
    plt.plot(returns, label='Episode Return')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title(title)
    plt.legend()
    plt.show()


########################################################################
# 4. Reinforcement Learning Environment & Agents
########################################################################

class PricePredictionEnv:
    """
    Example RL Environment:
      - Each 'step' moves through the dataset row by row.
      - The 'observation' is the feature row (excluding the target column).
      - The 'action' is discrete for demonstration (e.g., guess -1, 0, or +1).
      - The 'reward' is negative of squared error between the guess and the true target.

    NOTE: In a real bond price prediction scenario, you would likely:
      1) Use a continuous action or a more sophisticated approach.
      2) Possibly incorporate multi-step returns, discount factors, etc.
    """

    def __init__(self, df, target_column='target_price'):
        """
        Initialize environment with a DataFrame.

        :param df: pd.DataFrame
            Data containing features and the target price.
        :param target_column: str
            Column name for the target price.
        """
        self.df = df.reset_index(drop=True)
        self.target_col = target_column

        # Convert DataFrame to numpy for quick indexing
        self.X = self.df.drop(columns=[self.target_col]).values
        self.y = self.df[self.target_col].values

        # Total steps
        self.num_steps = len(self.df)

        # Current state index
        self.current_step = 0
        # Whether the episode is done
        self.done = False

        # Example discrete action space of size 3
        # Could be [-1.0, 0.0, +1.0] as guesses
        self.action_space = 3
        # Observation dimension
        self.observation_dim = self.X.shape[1]

    def reset(self):
        """
        Reset environment to the first step of the dataset.

        :return: np.array
            The first observation (features) in the dataset.
        """
        self.current_step = 0
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        """
        Returns the observation (feature vector) at the current step.
        """
        if self.done:
            return None
        return self.X[self.current_step]

    def step(self, action):
        """
        Execute one time step.

        :param action: int
            Chosen discrete action in [0, 1, 2].
        :return: (obs, reward, done, info)
            obs: next state (or None if done)
            reward: float
            done: bool
            info: dict
        """
        # True target price
        true_val = self.y[self.current_step]

        # Convert discrete action to a predicted value
        predicted_val = self._action_to_prediction(action)

        # Reward: negative MSE
        reward = -(true_val - predicted_val) ** 2

        # Move to the next step
        self.current_step += 1
        if self.current_step >= self.num_steps:
            self.done = True
            obs = None
        else:
            obs = self._get_observation()

        info = {}
        return obs, reward, self.done, info

    def _action_to_prediction(self, action):
        """
        Simple example of how to map an integer action to a float prediction.

        NOTE: Realistic scenarios would have a policy network directly predict
        the bond price or some relevant quantity, rather than this arbitrary mapping.
        """
        mapping = {
            0: -1.0,
            1: 0.0,
            2: 1.0
        }
        return mapping[action]


########################################
# 4a. Policy Gradient Agent (Stub)
########################################

class PolicyGradientAgent:
    """
    Simple policy gradient agent with a small neural network.

    The agent models pi(a|s) as a categorical distribution
    over discrete actions a, given state s.
    """

    def __init__(self, obs_dim, action_space, lr=0.001):
        """
        Initialize a small neural network for the policy.

        :param obs_dim: int
            Dimension of state (observation).
        :param action_space: int
            Number of discrete actions.
        :param lr: float
            Learning rate for the optimizer.
        """
        self.obs_dim = obs_dim
        self.action_space = action_space

        # Define policy network
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(obs_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(action_space, activation='softmax')
        ])

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def choose_action(self, obs):
        """
        Sample an action from the policy distribution given state obs.

        :param obs: np.array of shape (obs_dim,)
        :return: int
            Discrete action in [0, action_space - 1].
        """
        obs = obs.reshape((1, -1))  # Add batch dimension
        probs = self.model(obs, training=False).numpy().flatten()
        action = np.random.choice(self.action_space, p=probs)
        return action

    def update(self, observations, actions, rewards):
        """
        Policy gradient update using REINFORCE-like method.

        :param observations: list[np.array]
            List of states visited in the episode.
        :param actions: list[int]
            Actions taken in the episode.
        :param rewards: list[float]
            Rewards collected in the episode.
        """
        observations = np.array(observations)
        actions = np.array(actions)
        rewards = np.array(rewards)

        # Compute return for each step (assuming no discount or gamma=1 for simplicity)
        # Typically you'd incorporate a discount factor: Gt = sum_{k=0..∞} gamma^k * r_{t+k}
        returns = np.cumsum(rewards[::-1])[::-1]  # naive approach: just accumulate

        with tf.GradientTape() as tape:
            # Forward pass
            logits = self.model(observations, training=True)
            # Create one-hot action masks
            action_masks = tf.one_hot(actions, self.action_space)
            # Log probabilities for chosen actions
            # (logits -> softmax -> distribution, but we can do log-softmax more stably)
            log_probs = tf.reduce_sum(action_masks * tf.math.log(logits + 1e-10), axis=1)

            # Policy gradient loss = - mean( log_prob(a|s) * return )
            # (This is a simplified REINFORCE approach.)
            loss = -tf.reduce_mean(log_probs * returns)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


########################################
# 4b. Q-Learning Agent (Stub)
########################################

class QLearningAgent:
    """
    Simple Q-learning with a neural network approximator for Q(s,a).

    This is a minimal demonstration of a single-step update approach.
    """

    def __init__(self, obs_dim, action_space, lr=0.001, gamma=0.99, epsilon=0.1):
        """
        Initialize the Q-network.

        :param obs_dim: int
            Dimension of observations.
        :param action_space: int
            Number of discrete actions.
        :param lr: float
            Learning rate for the optimizer.
        :param gamma: float
            Discount factor for future rewards.
        :param epsilon: float
            Exploration rate for epsilon-greedy policy.
        """
        self.obs_dim = obs_dim
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon

        # Define Q-network
        self.q_model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(obs_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(action_space, activation='linear')
        ])
        self.optimizer = tf.keras.optimizers.Adam(lr)

    def choose_action(self, obs):
        """
        Epsilon-greedy policy.

        :param obs: np.array of shape (obs_dim,)
        :return: int
            Chosen action.
        """
        if np.random.rand() < self.epsilon:
            # Random action
            return np.random.randint(0, self.action_space)
        else:
            obs = obs.reshape((1, -1))
            q_values = self.q_model(obs, training=False).numpy()[0]
            return np.argmax(q_values)

    def update(self, obs, action, reward, next_obs, done):
        """
        Single-step Q-learning update.

        Q(s,a) = Q(s,a) + alpha [ r + gamma max_{a'} Q(s',a') - Q(s,a) ]

        :param obs: np.array
            Current state.
        :param action: int
            Action taken.
        :param reward: float
            Reward received.
        :param next_obs: np.array or None
            Next state (None if done).
        :param done: bool
            Whether the episode ended.
        """
        obs = obs.reshape((1, -1))
        if next_obs is None:
            next_obs = np.zeros_like(obs)
        else:
            next_obs = next_obs.reshape((1, -1))

        with tf.GradientTape() as tape:
            # Current Q values for the given state (batch_size=1)
            q_vals = self.q_model(obs, training=True)
            q_val = tf.gather(q_vals[0], action)

            # Next Q values for next state
            next_q_vals = self.q_model(next_obs, training=False)
            max_next_q = tf.reduce_max(next_q_vals[0])

            # Target = r + gamma * max_next_q  (0 if done)
            target = reward + (0.0 if done else self.gamma * max_next_q)

            # MSE Loss
            loss = tf.square(target - q_val)

        grads = tape.gradient(loss, self.q_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_model.trainable_variables))


########################################################################
# 5. RL Training Loops (Policy Gradient or Q-learning)
########################################################################

def train_policy_gradient(env, episodes=10, print_every=1, save_model_path=None):
    """
    Basic training loop for policy gradient using REINFORCE approach.

    :param env: PricePredictionEnv
        The environment to train on.
    :param episodes: int
        Number of episodes (full passes through the data).
    :param print_every: int
        How often to print the episode's return.
    :param save_model_path: str or None
        If not None, path to save the policy network after training.
    """
    agent = PolicyGradientAgent(env.observation_dim, env.action_space, lr=0.001)
    all_rewards = []

    for ep in range(episodes):
        obs = env.reset()
        done = False

        ep_rews = []
        obs_buf = []
        act_buf = []

        # Rollout
        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, done, _ = env.step(action)

            # Store transitions
            obs_buf.append(obs)
            act_buf.append(action)
            ep_rews.append(reward)

            obs = next_obs if not done else None

        # Update policy with the entire episode
        agent.update(obs_buf, act_buf, ep_rews)

        # Episode total return
        ep_return = np.sum(ep_rews)
        all_rewards.append(ep_return)

        if (ep + 1) % print_every == 0:
            print(f"[PolicyGradient] Episode {ep + 1}/{episodes} - Return: {ep_return:.4f}")

    # Plot training returns
    plot_returns(all_rewards, title='Policy Gradient Training Return')

    # Optionally save the agent’s model
    if save_model_path is not None:
        agent.model.save(save_model_path)
        print(f"[PolicyGradient] Policy network saved at '{save_model_path}'")


def train_q_learning(env, episodes=10, print_every=1, save_model_path=None):
    """
    Basic Q-learning training loop.

    :param env: PricePredictionEnv
        The environment.
    :param episodes: int
        Number of full passes (episodes).
    :param print_every: int
        Interval for printing return info.
    :param save_model_path: str or None
        If provided, path to save the Q-network after training.
    """
    agent = QLearningAgent(env.observation_dim, env.action_space, lr=0.001)
    all_rewards = []

    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_rews = []

        # Rollout
        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, done, _ = env.step(action)
            agent.update(obs, action, reward, next_obs, done)
            ep_rews.append(reward)
            obs = next_obs if not done else None

        # Sum of rewards this episode
        ep_return = np.sum(ep_rews)
        all_rewards.append(ep_return)

        if (ep + 1) % print_every == 0:
            print(f"[Q-Learning] Episode {ep + 1}/{episodes} - Return: {ep_return:.4f}")

    # Plot training returns
    plot_returns(all_rewards, title='Q-Learning Training Return')

    # Optionally save the Q-network
    if save_model_path is not None:
        agent.q_model.save(save_model_path)
        print(f"[Q-Learning] Q-network saved at '{save_model_path}'")


########################################################################
# 6. MAIN ENTRY POINT: Command-Line Arguments & Pipeline
########################################################################

def run_supervised(csv_path):
    """
    Full supervised learning pipeline:
      - Loads data from CSV.
      - Creates TF train/test Datasets.
      - Trains model.
      - Saves model.
      - Evaluates and plots predictions vs. actual.
    """
    df = load_data(csv_path)

    # The target_column must exist in your CSV, or rename as needed
    target_col = 'target_price'

    # Create train/test datasets
    train_ds, test_ds = create_tf_datasets(
        df,
        target_column=target_col,
        test_ratio=0.2,
        batch_size=64
    )

    # Determine input dimension (number of features)
    input_dim = df.drop(columns=[target_col]).shape[1]

    # Train MLP
    model, history = train_supervised(
        train_ds,
        test_ds,
        input_dim,
        epochs=5,
        hidden_units=[64, 64],
        lr=0.001
    )

    # Save model
    save_supervised_model(model, save_path="trained_supervised_model")

    # Plot training loss curves
    plot_training_history(history)

    # Evaluate + Plot predictions
    y_true, y_pred = evaluate_supervised(model, test_ds)
    plot_prediction_scatter(y_true, y_pred, title='Supervised Model Predictions')


def run_policy_gradient(csv_path):
    """
    Demonstration of training a policy gradient agent on the entire dataset
    as if it were an RL environment. The reward is negative MSE.
    """
    df = load_data(csv_path)

    # The environment will traverse the entire dataset row-by-row
    env = PricePredictionEnv(df, target_column='target_price')

    # Train the policy gradient for a set number of episodes
    train_policy_gradient(env, episodes=10, print_every=1, save_model_path=None)


def run_q_learning(csv_path):
    """
    Demonstration of training a Q-learning agent on the entire dataset
    as if it were an RL environment, with negative MSE as reward.
    """
    df = load_data(csv_path)
    env = PricePredictionEnv(df, target_column='target_price')

    # Train Q-Learning
    train_q_learning(env, episodes=10, print_every=1, save_model_path=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        type=str,
                        choices=["supervised", "policy_grad", "q_learning"],
                        default="supervised",
                        help="Which pipeline to run: 'supervised', 'policy_grad', or 'q_learning'")
    parser.add_argument("--csv_path",
                        type=str,
                        default="data/synthetic_4M_total.csv",
                        help="Path to the CSV data file")

    args = parser.parse_args()

    if args.mode == "supervised":
        print("[Main] Running supervised learning pipeline...")
        run_supervised(args.csv_path)
    elif args.mode == "policy_grad":
        print("[Main] Running Policy Gradient pipeline...")
        run_policy_gradient(args.csv_path)
    elif args.mode == "q_learning":
        print("[Main] Running Q-Learning pipeline...")
        run_q_learning(args.csv_path)


if __name__ == "__main__":
    main()
