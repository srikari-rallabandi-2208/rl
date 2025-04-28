import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from sklearn.preprocessing import StandardScaler

##############################################################################
# 1. Plotting Function
##############################################################################

def plot_returns(returns, filename="returns.png"):
    """
    Plots a list/array of episode returns and saves it as a PNG.
    """
    plt.figure()
    plt.plot(returns, marker='o')
    plt.title("Episode Returns")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[Plot] Saved {filename}")
    plt.show()


##############################################################################
# 2. Data Loading & Normalization
##############################################################################

def load_and_normalize_data(csv_path, target_col,
                            standardize_features=True,
                            standardize_target=True):
    """
    Load CSV into a DataFrame, separate features (X) and target (y).
    Optionally apply StandardScaler to features and/or target.

    :param csv_path: str
        Path to the dataset CSV.
    :param target_col: str
        Column name for the target.
    :param standardize_features: bool
        If True, apply StandardScaler to features.
    :param standardize_target: bool
        If True, apply StandardScaler to target.
    :return: X, y, scaler_X, scaler_y
        X, y are numpy arrays (float32), possibly scaled.
        scaler_X, scaler_y are scalers (or None) in case you want to invert transforms.
    """
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV columns!")

    # Separate out features & target
    X = df.drop(columns=[target_col]).values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    scaler_X = None
    scaler_y = None

    # Standardize X
    if standardize_features:
        scaler_X = StandardScaler()
        X = scaler_X.fit_transform(X)

    # Standardize y
    if standardize_target:
        scaler_y = StandardScaler()
        y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    return X, y, scaler_X, scaler_y


##############################################################################
# 3. Discrete RL Environment (Policy Grad / Q-Learning)
##############################################################################

class DiscretePricePredictionEnv:
    """
    Steps through data row-by-row. The target y is presumably scaled.
    The agent picks an action in {0,1,2}, which we map to [-1,0,+1].
    Reward = - ( (true - guess)^2 ).
    """

    def __init__(self, X, y):
        """
        :param X: shape (N, dim)
        :param y: shape (N,)
        """
        self.X = X
        self.y = y
        self.num_steps = len(X)

        self.current_step = 0
        self.done = False

        self.action_space = 3  # discrete
        self.observation_dim = X.shape[1]

    def reset(self):
        self.current_step = 0
        self.done = False
        return self.X[self.current_step]

    def step(self, action):
        guess = self._map_action_to_guess(action)
        true_val = self.y[self.current_step]

        reward = -(true_val - guess)**2

        self.current_step += 1
        if self.current_step >= self.num_steps:
            self.done = True
            next_obs = None
        else:
            next_obs = self.X[self.current_step]
        return next_obs, reward, self.done, {}

    def _map_action_to_guess(self, action_idx):
        mapping = {0: -1.0, 1: 0.0, 2: 1.0}
        return mapping[action_idx]


##############################################################################
# 4. Discrete RL Agents (Policy Gradient & Q-learning)
##############################################################################

class PolicyGradientAgent:
    """
    Minimal REINFORCE (on-policy) for discrete actions.
    """

    def __init__(self, obs_dim, act_dim, lr=1e-3):
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(obs_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(act_dim, activation='softmax')
        ])
        self.optimizer = tf.keras.optimizers.Adam(lr)

    def choose_action(self, obs):
        obs_tensor = obs.reshape((1, -1))
        probs = self.model(obs_tensor, training=False).numpy()[0]
        action = np.random.choice(self.act_dim, p=probs)
        return action

    def update(self, obs_buf, act_buf, rew_buf):
        """
        After collecting a full episode's (obs, act, rew), do one gradient update.
        """
        obs_buf = np.array(obs_buf, dtype=np.float32)
        act_buf = np.array(act_buf, dtype=np.int32)
        rew_buf = np.array(rew_buf, dtype=np.float32)

        # no discount for simplicity
        returns = np.cumsum(rew_buf[::-1])[::-1]

        with tf.GradientTape() as tape:
            logits = self.model(obs_buf, training=True)  # shape [N, act_dim]
            # one-hot
            oh_act = tf.one_hot(act_buf, depth=self.act_dim)
            # sum log probs for the chosen action
            log_probs = tf.reduce_sum(oh_act * tf.math.log(logits + 1e-10), axis=1)
            # Policy gradient loss = - E[ log_prob(a|s)*return ]
            loss = -tf.reduce_mean(log_probs * returns)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


class QLearningAgent:
    """
    Minimal Q-learning with single-step updates, no replay buffer, discrete actions.
    """

    def __init__(self, obs_dim, act_dim, lr=1e-3, gamma=0.99, eps=0.1):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.eps = eps

        self.q_model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(obs_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(act_dim, activation='linear')
        ])
        self.optimizer = tf.keras.optimizers.Adam(lr)

    def choose_action(self, obs):
        if np.random.rand() < self.eps:
            return np.random.randint(0, self.act_dim)
        else:
            obs_tensor = obs.reshape((1, -1))
            qvals = self.q_model(obs_tensor, training=False).numpy()[0]
            return np.argmax(qvals)

    def update(self, obs, action, reward, next_obs, done):
        """
        Single-step Q-learning update:
        Q(s,a) <- Q(s,a) + alpha [r + gamma * max_a' Q(s',a') - Q(s,a)]
        """
        obs_tensor = obs.reshape((1, -1))
        if next_obs is None:
            next_obs = np.zeros_like(obs_tensor)
        else:
            next_obs = next_obs.reshape((1, -1))

        with tf.GradientTape() as tape:
            qvals = self.q_model(obs_tensor, training=True)
            qval = qvals[0, action]

            next_qvals = self.q_model(next_obs, training=False)[0]
            max_next_q = tf.reduce_max(next_qvals)
            target = reward + (0.0 if done else self.gamma * max_next_q)
            loss = tf.square(target - qval)

        grads = tape.gradient(loss, self.q_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_model.trainable_variables))


##############################################################################
# 5. Continuous RL Environment (DDPG)
##############################################################################

class ContinuousPricePredictionEnv:
    """
    For DDPG-like methods, we define a continuous action space:
    agent outputs a real number guess in [action_low, action_high].
    Reward = -(true - guess)**2
    Steps row-by-row in (X,y).
    """

    def __init__(self, X, y, action_low=-2.0, action_high=2.0):
        self.X = X
        self.y = y
        self.num_steps = len(X)
        self.obs_dim = X.shape[1]

        self.action_low = action_low
        self.action_high = action_high

        self.current_step = 0
        self.done = False

    def reset(self):
        self.current_step = 0
        self.done = False
        return self.X[self.current_step]

    def step(self, action):
        guess = np.clip(action, self.action_low, self.action_high)
        true_val = self.y[self.current_step]

        reward = -(true_val - guess)**2

        self.current_step += 1
        if self.current_step >= self.num_steps:
            self.done = True
            next_obs = None
        else:
            next_obs = self.X[self.current_step]
        return next_obs, reward, self.done, {}


##############################################################################
# 6. Minimal DDPG Implementation
##############################################################################

class ReplayBuffer:
    """
    For storing transitions (s, a, r, s') used by DDPG or TD3.
    """

    def __init__(self, max_size=100000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state_buf = None
        self.next_state_buf = None
        self.actions_buf = None
        self.rewards_buf = None
        self.done_buf = None

    def init_buffers(self, obs_dim, act_dim):
        self.state_buf = np.zeros((self.max_size, obs_dim), dtype=np.float32)
        self.next_state_buf = np.zeros((self.max_size, obs_dim), dtype=np.float32)
        self.actions_buf = np.zeros((self.max_size, act_dim), dtype=np.float32)
        self.rewards_buf = np.zeros((self.max_size,), dtype=np.float32)
        self.done_buf = np.zeros((self.max_size,), dtype=np.float32)

    def store(self, state, action, reward, next_state, done):
        idx = self.ptr
        self.state_buf[idx] = state
        self.actions_buf[idx] = action
        self.rewards_buf[idx] = reward
        self.done_buf[idx] = float(done)
        if next_state is not None:
            self.next_state_buf[idx] = next_state
        else:
            self.next_state_buf[idx] = 0.0

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=64):
        idxes = np.random.randint(0, self.size, size=batch_size)
        return dict(
            s=self.state_buf[idxes],
            a=self.actions_buf[idxes],
            r=self.rewards_buf[idxes],
            s2=self.next_state_buf[idxes],
            d=self.done_buf[idxes],
        )


def build_actor(obs_dim, act_dim=1, act_low=-2.0, act_high=2.0):
    """
    Simple MLP actor for continuous action.
    Tanh output => [-1,+1], then scale to [act_low, act_high].
    """
    inputs = tf.keras.layers.Input(shape=(obs_dim,))
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    raw_out = tf.keras.layers.Dense(act_dim, activation='tanh')(x)
    scale = (act_high - act_low) / 2.0
    mid = (act_high + act_low) / 2.0
    scaled_out = raw_out * scale + mid

    model = tf.keras.Model(inputs=inputs, outputs=scaled_out)
    return model


def build_critic(obs_dim, act_dim=1):
    """
    Q(s,a) approximator. We'll just concat s,a.
    """
    s_in = tf.keras.layers.Input(shape=(obs_dim,))
    a_in = tf.keras.layers.Input(shape=(act_dim,))
    concat = tf.keras.layers.Concatenate()([s_in, a_in])
    x = tf.keras.layers.Dense(64, activation='relu')(concat)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    out = tf.keras.layers.Dense(1, activation='linear')(x)
    model = tf.keras.Model(inputs=[s_in, a_in], outputs=out)
    return model


class DDPGAgent:
    """
    Minimal DDPG approach for 1-D continuous action.
    - We maintain an actor and a critic.
    - A target actor/critic for stable training.
    - A replay buffer.
    - Noise for exploration.
    """

    def __init__(self, obs_dim, act_dim=1, act_low=-2.0, act_high=2.0,
                 gamma=0.99, tau=0.005, lr=1e-3, buffer_size=100000):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.act_low = act_low
        self.act_high = act_high

        # Actor & Critic
        self.actor = build_actor(obs_dim, act_dim, act_low, act_high)
        self.critic = build_critic(obs_dim, act_dim)
        self.actor_optimizer = tf.keras.optimizers.Adam(lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(lr)

        # Target networks
        self.actor_target = build_actor(obs_dim, act_dim, act_low, act_high)
        self.critic_target = build_critic(obs_dim, act_dim)

        # Copy weights initially
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())

        # Replay buffer
        self.replay = ReplayBuffer(max_size=buffer_size)
        self.replay.init_buffers(obs_dim, act_dim)

        # For exploration noise
        self.noise_std = 0.1  # you can reduce this over time

    def choose_action(self, state):
        """
        Predict action from actor, add some Gaussian noise for exploration.
        """
        state_tensor = state.reshape(1, -1).astype(np.float32)
        a = self.actor(state_tensor, training=False).numpy()[0]
        # exploration noise
        a += np.random.normal(0, self.noise_std, size=a.shape)
        # clamp
        a = np.clip(a, self.act_low, self.act_high)
        return a

    def store_transition(self, s, a, r, s2, done):
        self.replay.store(s, a, r, s2, done)

    def update(self, batch_size=64):
        """
        Sample from replay, update actor & critic.
        Typically call this multiple times per step in environment (DDPG approach).
        """
        if self.replay.size < batch_size:
            return  # not enough data

        batch = self.replay.sample_batch(batch_size)
        # Convert everything to tf tensors
        s = tf.convert_to_tensor(batch['s'], dtype=tf.float32)
        a = tf.convert_to_tensor(batch['a'], dtype=tf.float32)
        r = tf.convert_to_tensor(batch['r'], dtype=tf.float32)
        s2 = tf.convert_to_tensor(batch['s2'], dtype=tf.float32)
        d = tf.convert_to_tensor(batch['d'], dtype=tf.float32)

        # 1) Critic update
        with tf.GradientTape() as tape:
            a2 = self.actor_target(s2, training=False)
            q_next = self.critic_target([s2, a2], training=False)
            target_q = r[:, None] + self.gamma * (1 - d[:, None]) * q_next
            q_vals = self.critic([s, a], training=True)
            critic_loss = tf.reduce_mean((q_vals - target_q)**2)
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # 2) Actor update
        with tf.GradientTape() as tape:
            actions = self.actor(s, training=True)
            q = self.critic([s, actions], training=False)
            # maximize q => minimize -q
            actor_loss = -tf.reduce_mean(q)
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # 3) Soft update targets
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

    def _soft_update(self, net, net_targ):
        w = np.array(net.get_weights(), dtype=object)
        w_t = np.array(net_targ.get_weights(), dtype=object)
        net_targ.set_weights(self.tau*w + (1 - self.tau)*w_t)


##############################################################################
# 7. Training Loops
##############################################################################

def map_action_to_guess(action_idx):
    """
    For discrete environment MSE logging outside the env.
    0->-1, 1->0, 2->+1
    """
    mapping = {0: -1.0, 1: 0.0, 2: 1.0}
    return mapping[action_idx]


def train_q_learning(env, episodes=10):
    """
    Simple Q-learning loop. Logs returns & average MSE to qlearning_log.csv.
    """
    agent = QLearningAgent(env.observation_dim, env.action_space)
    log_csv = "qlearning_log.csv"

    with open(log_csv, "w") as f:
        f.write("episode,return,avg_mse\n")

    all_returns = []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_rews = []
        sq_errs = []

        while not done:
            act = agent.choose_action(obs)
            guess = map_action_to_guess(act)
            true_val = env.y[env.current_step]

            next_obs, reward, done, _ = env.step(act)

            # Single-step Q update
            agent.update(obs, act, reward, next_obs, done)

            sq_errs.append((true_val - guess)**2)
            ep_rews.append(reward)
            obs = next_obs if not done else None

        ep_return = np.sum(ep_rews)
        avg_mse = np.mean(sq_errs)
        all_returns.append(ep_return)

        with open(log_csv, "a") as f:
            f.write(f"{ep+1},{ep_return},{avg_mse}\n")

        print(f"[Q-Learning] Ep {ep+1}/{episodes} | Return={ep_return:.3f}, AvgMSE={avg_mse:.6f}")

    plot_returns(all_returns, filename="qlearning_returns.png")


def train_policy_gradient(env, episodes=10):
    """
    Single-episode REINFORCE approach:
    - gather entire (obs, act, rew) trajectory
    - update policy once per episode
    """
    agent = PolicyGradientAgent(env.observation_dim, env.action_space)
    log_csv = "policy_grad_log.csv"

    with open(log_csv, "w") as f:
        f.write("episode,return,avg_mse\n")

    all_returns = []
    for ep in range(episodes):
        obs_buf = []
        act_buf = []
        rew_buf = []
        sq_errs = []

        obs = env.reset()
        done = False
        while not done:
            act = agent.choose_action(obs)
            guess = map_action_to_guess(act)
            true_val = env.y[env.current_step]

            next_obs, reward, done, _ = env.step(act)

            obs_buf.append(obs)
            act_buf.append(act)
            rew_buf.append(reward)
            sq_errs.append((true_val - guess)**2)

            obs = next_obs if not done else None

        # single update at the end
        agent.update(obs_buf, act_buf, rew_buf)
        ep_return = np.sum(rew_buf)
        avg_mse = np.mean(sq_errs)
        all_returns.append(ep_return)

        with open(log_csv, "a") as f:
            f.write(f"{ep+1},{ep_return},{avg_mse}\n")

        print(f"[PolicyGrad] Ep {ep+1}/{episodes} | Return={ep_return:.3f}, AvgMSE={avg_mse:.6f}")

    plot_returns(all_returns, filename="policygrad_returns.png")


def train_ddpg(env, episodes=10, batch_size=64, updates_per_step=1):
    """
    We'll collect transitions in a replay buffer, do DDPG updates after each step.
    Log returns + average MSE to ddpg_log.csv
    """
    agent = DDPGAgent(obs_dim=env.obs_dim, act_dim=1,
                      act_low=env.action_low, act_high=env.action_high,
                      lr=1e-3, buffer_size=50000)
    log_csv = "ddpg_log.csv"

    with open(log_csv, "w") as f:
        f.write("episode,return,avg_mse\n")

    all_returns = []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_rews = []
        sq_errs = []

        while not done:
            action = agent.choose_action(obs)  # shape (1,)
            next_obs, reward, done, _ = env.step(action[0])
            agent.store_transition(obs, action, reward, next_obs, done)

            true_val = env.y[env.current_step - 1]  # we advanced after step
            sq_errs.append((true_val - action[0])**2)

            obs = next_obs
            ep_rews.append(reward)

            # do multiple updates
            for _ in range(updates_per_step):
                agent.update(batch_size=batch_size)

        ep_return = np.sum(ep_rews)
        avg_mse = np.mean(sq_errs)
        all_returns.append(ep_return)

        with open(log_csv, "a") as f:
            f.write(f"{ep+1},{ep_return},{avg_mse}\n")

        print(f"[DDPG] Episode {ep+1}/{episodes} | Return={ep_return:.3f}, AvgMSE={avg_mse:.6f}")

    plot_returns(all_returns, filename="ddpg_returns.png")


##############################################################################
# 8. Main Script
##############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str,
                        choices=["q_learning", "policy_grad", "ddpg"],
                        default="q_learning",
                        help="Which RL approach to run.")
    parser.add_argument("--csv_path", type=str,
                        default="data/synthetic_4M_total.csv",
                        help="Path to your CSV dataset.")
    parser.add_argument("--target_col", type=str,
                        default="Estimated_Price",
                        help="Column name for the target price.")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to run for RL.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Minibatch size for DDPG updates.")
    parser.add_argument("--normalize_features", action="store_true",
                        help="If set, standardize feature columns.")
    parser.add_argument("--normalize_target", action="store_true",
                        help="If set, standardize target column.")
    args = parser.parse_args()

    # 1) Load & normalize data
    X, y, scaler_X, scaler_y = load_and_normalize_data(
        csv_path=args.csv_path,
        target_col=args.target_col,
        standardize_features=args.normalize_features,
        standardize_target=args.normalize_target
    )

    # 2) Build environment and run RL training
    if args.mode == "q_learning":
        env = DiscretePricePredictionEnv(X, y)
        train_q_learning(env, episodes=args.episodes)

    elif args.mode == "policy_grad":
        env = DiscretePricePredictionEnv(X, y)
        train_policy_gradient(env, episodes=args.episodes)

    elif args.mode == "ddpg":
        env = ContinuousPricePredictionEnv(X, y, action_low=-2.0, action_high=2.0)
        train_ddpg(env, episodes=args.episodes, batch_size=args.batch_size)

    else:
        print("Unknown mode. Must be one of [q_learning, policy_grad, ddpg].")


if __name__ == "__main__":
    main()
