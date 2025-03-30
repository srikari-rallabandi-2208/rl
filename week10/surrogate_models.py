"""
surrogate_models.py

Defines a SurrogateModelTrainer class that trains and evaluates three models:
1) Random Forest
2) Gradient Boosting
3) MLP Neural Network

Use this to replicate PDE-labeled data for large-scale predictions.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score


class SurrogateModelTrainer:
    """
    A trainer class for multiple surrogate models: RandomForest, GradientBoost, and MLP (NN).

    Example Usage:
    -------------
    trainer = SurrogateModelTrainer()
    trainer.train_all_surrogates(
        df_pde,
        feature_cols=["IVOL","CDS","S","cp","cfq","cv","d","r","T"],
        target_col="Estimated_Price"
    )
    # Then pick which model to use for predictions:
    preds = trainer.predict('rf', new_data_df)
    """

    def __init__(self):
        # Model placeholders
        self.rf_model  = None
        self.gb_model  = None
        self.nn_model  = None

        # Data scaling
        self.scaler = None
        self.feature_cols = []
        self.target_col   = None

    def prepare_data(self, df, feature_cols, target_col,
                     test_size=0.2, random_state=42):
        """
        Splits df into train/validation sets, scales features, and saves internal references.

        Returns: (X_train_scaled, X_val_scaled, y_train, y_val)
        """
        self.feature_cols = feature_cols
        self.target_col   = target_col

        X = df[feature_cols].values
        y = df[target_col].values

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Feature scaling
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled   = self.scaler.transform(X_val)

        return X_train_scaled, X_val_scaled, y_train, y_val

    def train_random_forest(self, X_train, y_train,
                            n_estimators=100, random_state=42):
        """
        Train a RandomForestRegressor on (X_train, y_train).
        """
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        self.rf_model = rf

    def train_gradient_boosting(self, X_train, y_train,
                                n_estimators=100, learning_rate=0.1,
                                max_depth=3, random_state=42):
        """
        Train a GradientBoostingRegressor on (X_train, y_train).
        """
        gb = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )
        gb.fit(X_train, y_train)
        self.gb_model = gb

    def train_neural_network(self, X_train, y_train,
                             hidden_layer_sizes=(64,64),
                             max_iter=500, random_state=42):
        """
        Train an MLPRegressor as a simple feed-forward neural network.
        """
        nn = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            max_iter=max_iter,
            random_state=random_state
        )
        nn.fit(X_train, y_train)
        self.nn_model = nn

    def evaluate_model(self, model, X_val, y_val, model_name="Model"):
        """
        Evaluate a trained model: R^2 and MSE on validation set.
        """
        preds = model.predict(X_val)
        mse_val = mean_squared_error(y_val, preds)
        r2_val  = r2_score(y_val, preds)
        print(f"{model_name} -> MSE: {mse_val:.4f}, R^2: {r2_val:.4f}")

    def train_all_surrogates(self, df, feature_cols, target_col):
        """
        1) Prepare data
        2) Train Random Forest, Gradient Boost, MLP
        3) Print validation performance
        """
        X_train_scaled, X_val_scaled, y_train, y_val = self.prepare_data(
            df, feature_cols, target_col
        )

        # 1) Random Forest
        self.train_random_forest(X_train_scaled, y_train)
        self.evaluate_model(self.rf_model, X_val_scaled, y_val, "RandomForest")

        # 2) Gradient Boost
        self.train_gradient_boosting(X_train_scaled, y_train)
        self.evaluate_model(self.gb_model, X_val_scaled, y_val, "GradientBoost")

        # 3) Neural Network
        self.train_neural_network(X_train_scaled, y_train)
        self.evaluate_model(self.nn_model, X_val_scaled, y_val, "NeuralNet")

    def predict(self, model_type, df):
        """
        Predict using one of the trained surrogate models ('rf', 'gb', 'nn').

        Parameters:
        -----------
        model_type : str
            One of 'rf', 'gb', or 'nn'.
        df : pd.DataFrame
            New data with at least the same feature columns used in training.

        Returns:
        --------
        np.ndarray of predictions
        """
        if self.scaler is None:
            raise ValueError("No scaler found. You must call train_all_surrogates or prepare_data first.")

        X_new = df[self.feature_cols].values
        X_new_scaled = self.scaler.transform(X_new)

        if model_type == 'rf' and self.rf_model is not None:
            return self.rf_model.predict(X_new_scaled)
        elif model_type == 'gb' and self.gb_model is not None:
            return self.gb_model.predict(X_new_scaled)
        elif model_type == 'nn' and self.nn_model is not None:
            return self.nn_model.predict(X_new_scaled)
        else:
            raise ValueError(f"Model type '{model_type}' not recognized or not trained.")
