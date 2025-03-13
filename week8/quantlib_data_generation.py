"""
---

# File: `quantlib_data_generation.py`

"""
import QuantLib as ql
import numpy as np
import pandas as pd
import os


def generate_american_call_data(N=100_000, output_file="data/american_call_data.csv", seed=42):
    """
    Generate random American call option parameters and compute their prices via QuantLib.
    :param N: number of samples
    :param output_file: CSV path to save the data
    :param seed: random seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    records = []
    today = ql.Date(1, 1, 2025)
    calendar = ql.NullCalendar()  # for simple date adjustments

    for _ in range(N):
        S = rng.uniform(10, 200)  # Underlying spot
        K = rng.uniform(10, 200)  # Strike
        r = rng.uniform(0.00, 0.10)  # Risk-free rate
        q = rng.uniform(0.00, 0.05)  # Dividend yield
        sigma = rng.uniform(0.10, 0.50)  # Volatility
        T = rng.uniform(0.5, 2.0)  # Time to maturity in years

        # Convert T in years to a QuantLib date
        maturity_date = today + int(365 * T)

        # Set up the option
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
        exercise = ql.AmericanExercise(today, maturity_date)
        option = ql.VanillaOption(payoff, exercise)

        # Construct the Black-Scholes-Merton process
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
        r_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, ql.Actual365Fixed()))
        q_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, q, ql.Actual365Fixed()))
        vol_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(today, calendar, sigma, ql.Actual365Fixed())
        )
        bsm_process = ql.BlackScholesMertonProcess(spot_handle, q_ts, r_ts, vol_ts)

        # Pricing engine for American call
        engine = ql.BinomialVanillaEngine(bsm_process, "crr", 200)
        option.setPricingEngine(engine)

        # Compute the price
        price = option.NPV()

        # Store
        records.append([S, K, r, q, sigma, T, price])

    # Convert to DataFrame
    df = pd.DataFrame(records, columns=["S", "K", "r", "q", "sigma", "T", "AmericanPrice"])

    # Ensure data folder exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved {N} American call samples to {output_file}")


if __name__ == "__main__":
    generate_american_call_data()
