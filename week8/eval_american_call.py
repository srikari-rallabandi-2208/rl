"""
Evaluation script:
1) Loads the static network (price regression).
2) Loads the RL policy (exercise strategy).
3) Compares both to the ground truth from QuantLib data (AmericanPrice).
4) Generates example plots, prints RMSE, etc.
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from main_static_american_call import StaticPriceNet
from rl_agent import PolicyGradientAgent
from american_call_env import AmericanCallEnv


def evaluate_static_model(df, model_path="data/static_model.pth"):
    """
    Evaluate the static regressor on the entire dataset: predict price and compute RMSE.
    """
    model = StaticPriceNet(input_dim=6, hidden_dim=64)
    if not os.path.exists(model_path):
        print("No static model found, skipping.")
        return

    model.load_state_dict(torch.load(model_path))
    model.eval()

    X = df[["S", "K", "r", "q", "sigma", "T"]].values.astype(np.float32)
    y_true = df["AmericanPrice"].values

    with torch.no_grad():
        preds = model(torch.from_numpy(X)).numpy().flatten()

    mse = np.mean((preds - y_true) ** 2)
    rmse = np.sqrt(mse)
    print(f"[Static Model] RMSE vs. AmericanPrice: {rmse:.4f}")

    # Optionally make a quick scatter plot
    plt.figure()
    plt.scatter(y_true[:500], preds[:500], marker='o', alpha=0.5)
    plt.plot([0, max(y_true[:500])], [0, max(y_true[:500])], '--')
    plt.xlabel("True AmericanPrice")
    plt.ylabel("Predicted Price")
    plt.title("Static Model Predictions vs. True (sample of 500 pts)")
    plt.savefig("data/static_model_scatter.png", dpi=150)
    plt.close()


def evaluate_rl_agent(df, policy_path="data/rl_policy_model.pth", n_eval=300):
    """
    Evaluate the RL agent by simulating a few random parameter sets, calculating the average payoff
    across multiple paths, then compare that average payoff (discounted) to the known AmericanPrice.

    Because an American call's fair value is also the "expected discounted payoff" under the
    optimal exercise strategy, we approximate the RL's "policy-based price" by Monte Carlo.
    """
    agent = PolicyGradientAgent()
    if not os.path.exists(policy_path):
        print("No RL policy model found, skipping.")
        return

    agent.load(policy_path)

    # We'll pick n_eval random samples from df
    df_sample = df.sample(n=n_eval).reset_index(drop=True)

    results = []
    n_paths = 200  # For each param set, run 200 paths
    gamma = 1.0  # We'll do no discounting in the environment, or we can do exp(-r*dt)...

    for idx, row in df_sample.iterrows():
        S0, K, r, q, sigma, T, true_price = row["S"], row["K"], row["r"], row["q"], row["sigma"], row["T"], row[
            "AmericanPrice"]
        # Evaluate RL by simulating multiple stock paths, having the agent choose exercise times
        payoffs = []
        for _ in range(n_paths):
            env = AmericanCallEnv(S0=S0, K=K, r=r, q=q, sigma=sigma, T=T, num_steps=50)
            state = env.reset()
            done = False
            while not done:
                action = agent.select_action(state)  # RL policy
                next_state, reward, done, _ = env.step(action)
                if not done:
                    state = next_state
            payoffs.append(reward)  # The environment returns final payoff

        mean_payoff = np.mean(payoffs)  # rough Monte Carlo estimate of RL policy's exercise value
        # If you want to discount each step properly, you could store the path's discounted payoff.

        results.append([S0, K, r, q, sigma, T, true_price, mean_payoff])

    eval_df = pd.DataFrame(results, columns=["S", "K", "r", "q", "sigma", "T", "AmericanPrice", "RLMeanPayoff"])

    # RMSE vs. the "AmericanPrice" from QuantLib
    # The difference is that true_price is theoretically the "optimal" exercise value in risk-neutral land.
    # Our RLMeanPayoff is from a random test. If the agent is near-optimal, these should be close.
    diffs = eval_df["RLMeanPayoff"] - eval_df["AmericanPrice"]
    rmse = np.sqrt(np.mean(diffs ** 2))
    print(f"[RL Agent] RMSE vs. AmericanPrice on {n_eval} random samples: {rmse:.4f}")

    # Quick scatter
    plt.figure()
    plt.scatter(eval_df["AmericanPrice"], eval_df["RLMeanPayoff"], marker='x')
    plt.plot([0, eval_df["AmericanPrice"].max()], [0, eval_df["AmericanPrice"].max()], '--')
    plt.xlabel("True AmericanPrice")
    plt.ylabel("RL Estimated (Mean Payoff)")
    plt.title("RL Agent vs. True AmericanPrice (Sample)")
    plt.savefig("data/rl_agent_scatter.png", dpi=150)
    plt.close()


def main():
    # Load entire dataset
    df = pd.read_csv("data/american_call_data.csv")
    df = df[df["AmericanPrice"] >= 0].reset_index(drop=True)

    evaluate_static_model(df, model_path="data/static_model.pth")
    evaluate_rl_agent(df, policy_path="data/rl_policy_model.pth", n_eval=300)

    print("Evaluation complete. Plots saved in 'data/' folder.")


if __name__ == "__main__":
    main()
