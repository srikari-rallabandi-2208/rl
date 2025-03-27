"""
main_dynamic_rl_3m.py

An example of using row-by-row environment for 3M data.
WARNING: This might be extremely slow if PDE solver is called 3M times.

Usage:
  python main_dynamic_rl_3m.py \
      --csv data/prepared_3m/train.csv \
      --episodes 1
"""

import argparse
from tf_engine_converged import TsiveriotisFernandesEngineConverged
from rl_agent_converged import PolicyGradientAgentConverged
from environment_3m_data import DataEnv3M  # The environment we just wrote


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to the 3M CSV (train or test).")
    parser.add_argument("--episodes", type=int, default=1, help="Number of times to iterate the entire dataset.")
    args = parser.parse_args()

    # PDE engine
    engine = TsiveriotisFernandesEngineConverged(
        S0=100, K=100, r=0.05, sigma=0.20, T=1.0, q=0.0, spread=0.01,
        M=100, N=100, par=100.0
    )
    engine.solve_pde()

    # RL agent
    agent = PolicyGradientAgentConverged(input_dim=3, hidden_dim=64, output_dim=2,
                                         lr=1e-3, gamma=0.99)

    # Our new data environment
    env = DataEnv3M(args.csv, engine)

    for ep in range(args.episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_reward(reward)
            state = next_state
        agent.update_policy()
        print(f"[Episode {ep + 1}/{args.episodes}] done.")

    print("Finished. You can save the agent or evaluate similarly.")


if __name__ == "__main__":
    main()
