# Convertible Bond Pricing: TF Model vs. RL Approaches

This project compares two methods for pricing convertible bonds:
- **TF Model (Tsiveriotis-Fernandes PDE Solver):** A model-based approach using a Crank-Nicolson finite-difference scheme.
- **Reinforcement Learning (RL) Approaches:**
  - **Online RL:** A policy-gradient agent that interacts with an environment based on the PDE pricing engine.
  - **Static RL (Supervised Learning):** A classifier trained on pre-generated data to imitate an optimal conversion decision.

The goal is to analyze how well RL agents can replicate or approximate the TF model’s decisions, and to explore ways to bridge any gaps.

---

## Project Overview

### Data Pipeline
1. **Data Generation:**  
   The script `src/data_generation.py` creates synthetic convertible bond data (e.g., bond price, IVOL, CDS, stock price, ttm_days, etc.) and saves it as `data/synthetic_data.xlsx`.

2. **Data Preparation:**  
   `src/data_preparation.py` reads the synthetic data, renames columns as required by the TF model, and preserves key columns (including `ttm_days`). The output is saved as `data/prepared_data.xlsx`.

3. **Pricing Estimation (TF Model):**  
   The script `src/pricing_estimation.py` uses the TF PDE solver (via `explicit_FD` in `src/pricing_model.py`) to compute an `Estimated_Price` for each row from `prepared_data.xlsx`. The final file `data/model_price_all.xlsx` now includes both `ttm_days` and `Estimated_Price`.

---

### RL Approaches

#### 1. Online RL
- **Description:**  
  The online RL agent, defined in `src/main_rl.py`, uses a REINFORCE-style policy-gradient algorithm to interact with an environment (`ConvertibleBondEnvTF` in `src/environment.py`) that computes the bond price using the TF engine.
- **Model Storage:**  
  The trained online RL model is saved as `data/policy_model.pth`.
- **Visualization:**  
  The evaluation script (e.g., `src/eval_rl.py`) generates the online RL comparison plot and saves it as `images/online_rl_comparison.png`.

#### 2. Static RL (Supervised Learning)
- **Description:**  
  The static RL (or supervised learning) approach, defined in `src/main_static_rl.py`, loads data from `model_price_all.xlsx`, computes a binary label for each record (using the rule: if conversion value \((S/K) \times \text{par}\) is greater than `Estimated_Price`, label as 1 for convert; otherwise 0), and trains a classifier.
- **Model Storage:**  
  The trained static model is saved as `data/static_policy_model.pth`.
- **Visualization:**  
  An evaluation script generates the static RL comparison plot and saves it as `images/static_rl_comparison.png`.

---

## Key Observations

### Online RL Plot
- **Stock Price Trajectory:**  
  The stock price evolves from about 100 to 110. The agent’s conversion actions (red X’s) are sparse, indicating that the agent rarely finds conversion optimal.
- **PDE Price Behavior:**  
  The PDE-based convertible bond price increases steeply near maturity (reaching values in the tens of thousands). This suggests that either the PDE boundary conditions, coupon logic, or other parameters may be causing inflated prices.
- **Inference:**  
  With such high PDE prices, the immediate conversion value is relatively low; thus, the online RL agent prefers to hold, leading to few conversion decisions.

### Static RL Plot
- **Time Range:**  
  The static plot primarily covers data near maturity (e.g., \( t \approx 1.0 \) to \( 1.025 \)). This indicates that the dataset used for static training has limited time variation.
- **Stock Price & PDE Price:**  
  The PDE price remains very high, while the static learner appears to predict conversion actions frequently (pink X’s). This discrepancy suggests that either the labeling logic (conversion value vs. PDE price) or the dataset characteristics are misaligned.
- **Inference:**  
  If most data points are near maturity with extremely high PDE prices, the labeling rule should ideally indicate “hold.” Frequent convert predictions may mean the static learner is overfitting to a narrow data range or that the data does not capture earlier time dynamics.

### Bridging the Gap Between RL and TF
- **Data Diversity:**  
  - **Time Variation:** More data spanning the entire bond lifecycle (from issuance to maturity) would enable both RL approaches to learn more granular behavior.
  - **Market Regimes:** Incorporating data that reflects different historical cycles and volatilities can improve robustness.
- **Parameter Tuning:**  
  Ensure that the TF PDE parameters (interest rate, spread, coupon, \(S_{\max}\), etc.) are calibrated to yield realistic bond prices (e.g., not orders of magnitude higher than par).
- **Hybrid Methods:**  
  Reward shaping or teacher-student frameworks that integrate PDE outputs into the RL reward signal can help align RL policies with the TF model.
- **Model Architecture:**  
  Pure RL approaches (such as policy-gradient models like GPRO) may work, but their performance will depend on careful environment design and parameter tuning.

---

## Usage Instructions

### 1. Installing Dependencies
Create a virtual environment (if desired) and install required packages:
```bash
pip install -r requirements.txt
```

### 2. Data Generation and Preparation
run 

```python
python src/data_generation.py
python src/data_preparation.py
```

This produces data/synthetic_data.xlsx and then data/prepared_data.xlsx.

### 3. TF Model Pricing

run
```python
python src/pricing_estimation.py
```

This applies the TF PDE solver and saves data/model_price_all.xlsx (with both ttm_days and Estimated_Price).

### 4. Online RL Training

run

```python
python src/main_rl.py
```

This trains the online RL agent and saves data/policy_model.pth.

### 5. Static RL Training

run 
```python
python src/main_static_rl.py
```

This trains the static classifier and saves data/static_policy_model.pth.

### 6. Evaluation and Visualization

run
```python
python src/eval_comparisions.py
```

This generates and saves comparison plots as:

	•	images/online_rl_comparison.png
	•	images/static_rl_comparison.png

This project provides a flexible framework for comparing a TF-based PDE pricing model with both online and offline RL approaches. Key challenges include ensuring realistic PDE outputs, obtaining a diverse dataset covering the full bond lifecycle, and potentially integrating hybrid methods to guide RL training. Future extensions could include multi-factor models, jump-diffusion processes, or hybrid PDE-RL frameworks to further align model-free learning with traditional pricing theory.