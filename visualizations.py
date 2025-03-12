"""
visualizations.py

This module provides functions to generate and save comparative visualizations.
It compares:
  - The PDE-based convertible bond price (TF model),
  - The online RL agent's conversion decisions,
  - And the static (offline) RL learner's predictions.
The generated plots are saved as PNG files in the data folder.
"""

import matplotlib.pyplot as plt


def plot_comparison_online_rl(times, stock_prices, pde_prices, rl_decisions, filename="data/online_rl_comparison.png"):
    """
    Plots and saves a comparison for the online RL agent (interacting with the PDE engine).
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(times, stock_prices, label="Stock Price", color="blue")
    # Mark conversion actions for online RL.
    rl_convert = (rl_decisions == 1)
    plt.scatter(times[rl_convert], stock_prices[rl_convert], color="red", marker="x", s=50, label="RL Convert")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.title("Online RL: Stock Price and Conversion Decisions")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(times, pde_prices, label="PDE Price", color="green")
    plt.xlabel("Time")
    plt.ylabel("PDE Price")
    plt.title("PDE-based Convertible Bond Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Online RL comparison plot saved as {filename}")


def plot_comparison_static(times, stock_prices, pde_prices, static_preds, filename="data/static_rl_comparison.png"):
    """
    Plots and saves a comparison for the static (offline) learner's decisions compared with the PDE price.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(times, stock_prices, label="Stock Price", color="blue")
    # Mark predicted conversion points (static learner).
    static_convert = (static_preds == 1)
    plt.scatter(times[static_convert], stock_prices[static_convert], color="magenta", marker="x", s=50,
                label="Static Convert")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.title("Static Data Learner: Stock Price and Conversion Predictions")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(times, pde_prices, label="PDE Price", color="green")
    plt.xlabel("Time")
    plt.ylabel("PDE Price")
    plt.title("PDE-based Convertible Bond Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Static RL comparison plot saved as {filename}")
