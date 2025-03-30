# Converged Tsiveriotis--Fernandes Model and Surrogate

This repository contains a modular implementation of the Tsiveriotis--Fernandes (TF) PDE model using a Crank–Nicolson solver, along with code for surrogate models to replicate PDE outputs at scale.

## Files

1. **pde_solver.py**  
   - `TsiveriotisFernandesSolver`: A class that performs Crank–Nicolson to solve the TF PDE for convertible bonds.

2. **main.py**  
   - Main script to:
     - Load and preprocess bond data.
     - Cluster rows by time to maturity (TTM).
     - Choose `(dx, dt)` for each cluster.
     - Solve the PDE for each row.
     - Save the results (including `Estimated_Price`) to an Excel file.

3. **surrogate_models.py**  
   - `SurrogateModelTrainer`: trains three different models—Random Forest, Gradient Boosting, and a simple feed-forward Neural Network (MLP)—on PDE-labeled data.

4. **README.md**  
   - This documentation.

## Requirements

- Python 3.8+ recommended
- `numpy`
- `pandas`
- `scikit-learn`
- `openpyxl` (to save results in Excel)
- `matplotlib` (optional, if you want to visualize)
- (Optional) `torch` or `tensorflow` if you plan more advanced neural networks

Install dependencies:

```bash
pip install numpy pandas scikit-learn openpyxl
```
# Usage
## PDE Pricing with main.py
1. Edit main.py:
Set the path to your CSV file for input_csv (in the __main__ block).

Adjust the nrows to test on fewer rows if desired.

Update the output Excel file path.

2. Run:
```python
python main.py
```

3. After completion, an Excel file (e.g., model_price_CN_dxdt.xlsx) will have a new Estimated_Price column representing the TF PDE solution.

## Surrogate on PDE Outputs
1. Generate PDE-labeled data by sampling if the dataset is large, e.g.:
```python
df_sample = df.sample(10000, random_state=42)
df_sample = estimate_prices(df_sample)
# Now df_sample has PDE-labeled 'Estimated_Price'
```
2. Train surrogate models:
```python
from surrogate_models import SurrogateModelTrainer

trainer = SurrogateModelTrainer()
feature_cols = ["IVOL","CDS","S","cp","cfq","cv","d","r","T"]
target_col   = "Estimated_Price"

trainer.train_all_surrogates(df_sample, feature_cols, target_col)
# Prints validation MSE, R^2 for RandomForest, GradientBoost, NeuralNet

```
3. Predict on a new dataset using the chosen model:
```python
df_new["SurrogatePrice"] = trainer.predict('rf', df_new)

```
## Customization
1. Adjust PDE grid:

    In main.py or in the PDE solver initialization, update S_max, dx, and dt as needed.

2. Refine Clustering:

    In cluster_data, you could cluster also by volatility or moneyness if you like.

3. Hyperparameters:

    surrogate_models.py provides default parameters for each model. Tweak as necessary to handle complexities of your dataset.