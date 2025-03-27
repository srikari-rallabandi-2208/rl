# Convertible Bond PDE + RL Codebase

This project demonstrates how to:
1. **Generate synthetic convertible-bond data** with PDE-based "Model_Price" using a Tsiveriotis-Fernandes approach (somewhat simplified).
2. **Train a static regressor** to replicate PDE results (offline approach).
3. **Train a dynamic RL agent** to make hold/convert decisions (online approach), generating comparison plots.

## Files Overview

1. **pde_engine.py**  
   - Contains the PDEEngine class and a `explicit_FD` function.  
   - PDEEngine can compute a PDE-based price for each row of data.  

2. **data_generator.py**  
   - Reads a large CSV (like `synthetic_data_with_correlation.csv`).
   - Shuffles the data, runs PDE row by row up to a max of 40K, 100K, 300K, etc., and saves a final CSV with `Model_Price`.

3. **main_static_regression.py**  
   - Reads a CSV containing `Model_Price`.
   - Splits 80:20, trains a feed-forward net to replicate PDE, and saves a scatter plot of predicted vs. actual PDE.

4. **main_dynamic_rl.py**  
   - A demonstration of a dynamic environment that calls PDE on each time step of a simulated GBM path.  
   - Trains a policy gradient agent for “hold vs convert,” then plots the final path with PDE price vs. stock price.

## Steps to Run

1. **Install Dependencies**

   ```bash
   pip install numpy pandas numba torch scikit-learn matplotlib
   ```
2. **Generate Datasets (40K, 100K, 300K)**

We assume you have synthetic_data_with_correlation.csv with columns like [bond_price, BestIVOL, CDS, S, coupon_rate, coupon_frequency, conversion_price, dividend, interest_rate, ttm_days, issuance_date, first_coupon_date, ...].

    a. 40K:
```python
python data_generator.py \
  --input data/synthetic_data_with_correlation.csv \
  --output data/final_40k.csv \
  --num-rows 40000
```
    b. 100K:

```python
python data_generator.py \
  --input data/synthetic_data_with_correlation.csv \
  --output data/final_40k.csv \
  --num-rows 100000
```
    
    c. 300K:
    
```python
python data_generator.py \
  --input data/synthetic_data_with_correlation.csv \
  --output data/final_40k.csv \
  --num-rows 300000
```

This will produce a CSV that contains Model_Price for each valid row.

3. **Run Static Regression for each dataset**

```python
python main_static_regression.py \
  --input data/final_40k.csv \
  --output data/static_40k
```

4. **Run Dynamic RL (Demo)***

```python
python main_dynamic_rl.py \
  --output data/dyn_40k \
  --episodes 30
```

That trains a simple RL agent on randomly generated GBM paths, each calling PDE on the fly.
It saves data/dyn_40k/dynamic_rl_plot.png.

This is not directly using the 40K CSV row-by-row; it’s a demonstration of PDE-based dynamic RL with a synthetic path. If you want to do row-by-row from your CSV in RL, you’d need a custom environment that iterates each row as a time step.

5. **Compare**

For the static approach, you have an R² vs. PDE.

For the dynamic approach, you have a final payoff or a final plot.

You can replicate the dynamic approach for 40K, 100K, 300K by just re-using the PDE logic with different parameters or environment designs.

1. Performance: PDE with M=700, N=15000 is quite large. For 100K or 300K, it can take a long time. Consider parallelization or adjusting M/N.

2. Plot: The dynamic script produces “stock vs PDE vs conversion actions.” You can compare across different data sizes or parameter settings.

3. Row-by-Row RL: If you specifically want to treat each row of your 40K/100K dataset as a time step, create an environment that iterates over those rows instead of generating a GBM path.