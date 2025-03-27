# Convertible Bond PDE & RL Demo

This repository contains a minimal, **self-contained** example that:
1. Prepares data (80:20 split).
2. Trains an **offline static** regressor to predict PDE-based convertible bond prices.
3. Demonstrates an **online RL** environment for a "hold vs. convert" scenario using a Tsiveriotis-Fernandes PDE engine.

> **Note**: The PDE solver code in `tf_engine_converged.py` is a stub generating random grids. Replace it with your real PDE solver for correct results.

## Files

- **data_preparation.py**  
  Reads your CSV, renames columns, splits 80:20 into `train.csv` and `test.csv`.

- **tf_engine_converged.py**  
  Minimal PDE engine wrapper; uses a fake `explicit_FD` function that returns random arrays.

- **environment_converged.py**  
  RL environment that steps through a synthetic stock path (GBM) or real path.  
  State = [S, t, PDE_price], Action = hold/convert.

- **rl_agent_converged.py**  
  A simple REINFORCE agent for discrete actions. PolicyNetwork has input_dim=3, output_dim=2.

- **main_static_rl_converged.py**  
  A script that trains a feed-forward regressor to match the PDE’s `tf_model_price`.  
  - Reads train/test CSV from the data prep step.  
  - Prints RMSE, R^2, and saves a scatter plot.

- **main_dynamic_rl_converged.py**  
  Generates a random GBM path each episode and trains the RL agent to maximize payoff.  
  - PDE engine is used to get the bond price at each step.  
  - The agent decides hold or convert.  

## Usage

### 1. Prepare your data

If you have a CSV, e.g. `synthetic_data_40k_with_model_price.csv`, do:
```bash
python data_preparation.py \
  --input data/synthetic_data_40k_with_model_price.csv \
  --output-dir data/prepared_40k
```
This yields:
```bash
data/prepared_40k/train.csv
data/prepared_40k/test.csv
```
They contain columns, including tf_model_price.
### 2. Offline Static Regression
Train a feed-forward net to replicate PDE prices:
```python
python main_static_rl_converged.py \
  --train data/prepared_40k/train.csv \
  --test data/prepared_40k/test.csv \
  --output data/prepared_40k
```
Output:
```bash
RMSE, R^2
data/prepared_40k/static_pred_vs_actual.png
data/prepared_40k/static_regressor.pth
```
### 3. Online Dynamic RL
We run a PDE-based environment that generates a synthetic GBM path. Then we train the RL agent:
```python
python main_dynamic_rl_converged.py
```
Output:
```bash
Trains for 50 episodes by default.
Prints final test-run reward.
Saves dynamic_rl_agent.pth.
```