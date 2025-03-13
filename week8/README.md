# American Call Pricing with RL and Static Network

This project demonstrates how to:
1. Generate American-call-option prices (ground truth) via QuantLib.
2. Train two models:
   - **Static** network (offline) that regresses the American option price from parameters.
   - **Dynamic** RL (online) that learns an exercise policy by simulating stock price paths.
3. Evaluate and visualize comparisons (RMSE, plots, etc.) with the QuantLib "true" prices.

## Prerequisites

- Python 3.8+ recommended
- `pip install -r requirements.txt`
- Ensure you can import `QuantLib` (the Python bindings).

## Files Overview

1. **quantlib_data_generation.py**  
   Generates data for American call options using QuantLib. Creates a CSV with columns `[S,K,r,q,sigma,T,AmericanPrice]`.

2. **main_static_american_call.py**  
   Loads the CSV and trains a feed-forward neural network to *regress* the American option price. Saves `static_model.pth`.

3. **american_call_env.py**  
   Defines a gym-like environment for an American call option: the agent chooses to hold or exercise at each discrete step.

4. **rl_agent.py**  
   REINFORCE-like policy gradient agent used for the dynamic RL approach.

5. **main_american_call_rl.py**  
   Trains the RL agent: simulates many stock paths, agent learns an exercise policy. Saves `rl_policy_model.pth`.

6. **eval_american_call.py**  
   - Loads both the static model and the RL agent.
   - Compares their estimated prices to the true AmericanPrice from QuantLib.
   - Plots the stock path plus exercise decisions (RL) vs. a “reference exercise boundary,” 
     and also calculates & prints RMSEs.

## How to Run

1. **Generate Data**  
   ```bash
   python quantlib_data_generation.py
    ```
   This creates data/american_call_data.csv.
2. **Train Static Model**
    ```python
    python main_static_american_call.py
    ```
   This trains a simple feed-forward net to predict the American call price from [S,K,r,q,sigma,T].

    Outputs static_model.pth.

3. **Train Dynamic RL**
    ```python
    python main_american_call_rl.py
    ```
   This runs a REINFORCE policy gradient. It can take a while depending on the number of episodes.

    Outputs rl_policy_model.pth.
4. **Evaluate & Visualize**
   ```python
    python eval_american_call.py
    ```
   This compares both models (static & RL) with the ground truth from QuantLib.

    You’ll see prints of RMSE, etc., and two plots:
   
    Pathwise RL exercise decisions vs. a reference boundary Scatter/line chart comparing static predictions to actual

    
After these steps, check out the generated images & console outputs for your analysis.
