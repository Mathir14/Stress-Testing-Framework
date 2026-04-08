"""
Stress Testing Module
Functions for applying various stress tests to evaluate model robustness
"""

import numpy as np
import pandas as pd


class StressTester:
    """
    Class for applying stress tests to datasets and evaluating model robustness
    """

    def __init__(self):
        self.original_data = None
        self.stressed_data = {}
        self.stress_results = {}

    def add_gaussian_noise(self, X, noise_level=0.1):
        """
        Add Gaussian noise to features

        Args:
            X: Input features (DataFrame or array)
            noise_level: Standard deviation of noise relative to feature std

        Returns:
            Noisy features
        """
        if isinstance(X, pd.DataFrame):
            # Create a copy and work with its underlying values
            result = X.copy()
            values = result.values

            for i in range(values.shape[1]):
                std = values[:, i].std()
                noise = np.random.normal(0, std * noise_level, size=len(values))
                values[:, i] = values[:, i] + noise

            # Values were modified in place, result already has the changes
            return result
        else:
            X_noisy = X.copy()
            for i in range(X.shape[1]):
                std = X[:, i].std()
                noise = np.random.normal(0, std * noise_level, size=X.shape[0])
                X_noisy[:, i] = X[:, i] + noise
            return X_noisy

    def add_uniform_noise(self, X, noise_range=0.1):
        """
        Add uniform noise to features

        Args:
            X: Input features
            noise_range: Range of uniform noise as fraction of feature range

        Returns:
            Noisy features
        """
        if isinstance(X, pd.DataFrame):
            # Create a copy and work with its underlying values
            result = X.copy()
            values = result.values

            for i in range(values.shape[1]):
                data_range = values[:, i].max() - values[:, i].min()
                noise = np.random.uniform(
                    -data_range * noise_range,
                    data_range * noise_range,
                    size=len(values),
                )
                values[:, i] = values[:, i] + noise

            # Values were modified in place, result already has the changes
            return result
        else:
            X_noisy = X.copy()
            for i in range(X.shape[1]):
                data_range = X[:, i].max() - X[:, i].min()
                noise = np.random.uniform(
                    -data_range * noise_range, data_range * noise_range, size=X.shape[0]
                )
                X_noisy[:, i] = X[:, i] + noise
            return X_noisy

    def feature_dropout(self, X, dropout_rate=0.2):
        """
        Randomly set features to zero (dropout)

        Args:
            X: Input features
            dropout_rate: Fraction of features to drop per sample

        Returns:
            Features with dropout
        """
        if isinstance(X, pd.DataFrame):
            # Create a copy and work with its underlying values
            result = X.copy()
            values = result.values
            mask = np.random.random(values.shape) > dropout_rate
            # Modify values in place
            result.iloc[:, :] = values * mask
            return result
        else:
            X_dropout = X.copy()
            mask = np.random.random(X.shape) > dropout_rate
            X_dropout = X_dropout * mask
            return X_dropout

    def feature_corruption(self, X, corruption_rate=0.1, corruption_type="zero"):
        """
        Corrupt random features

        Args:
            X: Input features
            corruption_rate: Fraction of values to corrupt
            corruption_type: 'zero', 'mean', 'random', or 'extreme'

        Returns:
            Corrupted features
        """
        if isinstance(X, pd.DataFrame):
            # Create a copy and work with its underlying values
            result = X.copy()
            values = result.values

            for i in range(values.shape[1]):
                n_corrupt = int(len(values) * corruption_rate)
                corrupt_idx = np.random.choice(len(values), n_corrupt, replace=False)

                if corruption_type == "zero":
                    values[corrupt_idx, i] = 0
                elif corruption_type == "mean":
                    values[corrupt_idx, i] = values[:, i].mean()
                elif corruption_type == "random":
                    values[corrupt_idx, i] = np.random.uniform(
                        values[:, i].min(), values[:, i].max(), size=n_corrupt
                    )
                elif corruption_type == "extreme":
                    # Randomly assign min or max values
                    extremes = np.random.choice(
                        [values[:, i].min(), values[:, i].max()], n_corrupt
                    )
                    values[corrupt_idx, i] = extremes

            # Values were modified in place, result already has the changes
            return result
        else:
            X_corrupt = X.copy()
            for i in range(X.shape[1]):
                n_corrupt = int(X.shape[0] * corruption_rate)
                corrupt_idx = np.random.choice(X.shape[0], n_corrupt, replace=False)

                if corruption_type == "zero":
                    X_corrupt[corrupt_idx, i] = 0
                elif corruption_type == "mean":
                    X_corrupt[corrupt_idx, i] = X[:, i].mean()
                elif corruption_type == "random":
                    X_corrupt[corrupt_idx, i] = np.random.uniform(
                        X[:, i].min(), X[:, i].max(), size=n_corrupt
                    )
                elif corruption_type == "extreme":
                    extremes = np.random.choice(
                        [X[:, i].min(), X[:, i].max()], n_corrupt
                    )
                    X_corrupt[corrupt_idx, i] = extremes
            return X_corrupt

    def scale_perturbation(self, X, scale_factor=1.5):
        """
        Perturb feature scales

        Args:
            X: Input features
            scale_factor: Scaling factor for perturbation

        Returns:
            Scaled features
        """
        if isinstance(X, pd.DataFrame):
            # Create a copy and work with its underlying values
            result = X.copy()
            values = result.values

            for i in range(values.shape[1]):
                # Randomly scale or inverse scale
                factor = scale_factor if np.random.random() > 0.5 else 1 / scale_factor
                values[:, i] = values[:, i] * factor

            # Values were modified in place, result already has the changes
            return result
        else:
            X_scaled = X.copy()
            for i in range(X.shape[1]):
                factor = scale_factor if np.random.random() > 0.5 else 1 / scale_factor
                X_scaled[:, i] = X[:, i] * factor
            return X_scaled

    def distribution_shift(self, X, shift_type="mean", shift_amount=0.5):
        """
        Shift feature distributions

        Args:
            X: Input features
            shift_type: 'mean' or 'variance'
            shift_amount: Amount of shift (for mean) or scaling factor (for variance)

        Returns:
            Shifted features
        """
        if isinstance(X, pd.DataFrame):
            # Create a copy and work with its underlying values
            result = X.copy()
            values = result.values

            for i in range(values.shape[1]):
                if shift_type == "mean":
                    # Shift mean by shift_amount * std
                    values[:, i] = values[:, i] + (values[:, i].std() * shift_amount)
                elif shift_type == "variance":
                    # Scale variance
                    mean = values[:, i].mean()
                    values[:, i] = mean + (values[:, i] - mean) * shift_amount

            # Values were modified in place, result already has the changes
            return result
        else:
            X_shifted = X.copy()
            for i in range(X.shape[1]):
                if shift_type == "mean":
                    X_shifted[:, i] = X[:, i] + (X[:, i].std() * shift_amount)
                elif shift_type == "variance":
                    mean = X[:, i].mean()
                    X_shifted[:, i] = mean + (X[:, i] - mean) * shift_amount
            return X_shifted

    def evaluate_stress_test(self, model, X_original, X_stressed, y_true):
        """
        Evaluate model performance on stressed data

        Args:
            model: Trained model
            X_original: Original features
            X_stressed: Stressed features
            y_true: True labels

        Returns:
            Dictionary with performance metrics
        """
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        # Convert y_true to numpy array if it's a pandas Series (to avoid index issues)
        if isinstance(y_true, pd.Series):
            y_true_array = y_true.values
        else:
            y_true_array = y_true

        # Predictions on original data
        y_pred_original = model.predict(X_original)
        acc_original = accuracy_score(y_true_array, y_pred_original)

        # Predictions on stressed data
        y_pred_stressed = model.predict(X_stressed)
        acc_stressed = accuracy_score(y_true_array, y_pred_stressed)

        # Performance drop
        performance_drop = acc_original - acc_stressed

        # Precision, Recall, F1 for stressed data
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_array, y_pred_stressed, average="weighted", zero_division=0
        )

        # Agreement between original and stressed predictions
        prediction_agreement = (y_pred_original == y_pred_stressed).mean()

        return {
            "accuracy_original": acc_original,
            "accuracy_stressed": acc_stressed,
            "performance_drop": performance_drop,
            "performance_drop_pct": (
                (performance_drop / acc_original * 100) if acc_original > 0 else 0
            ),
            "precision_stressed": precision,
            "recall_stressed": recall,
            "f1_stressed": f1,
            "prediction_agreement": prediction_agreement,
            "y_pred_original": y_pred_original,
            "y_pred_stressed": y_pred_stressed,
        }

    def batch_stress_test(self, model, X, y, stress_configs):
        """
        Run multiple stress tests

        Args:
            model: Trained model
            X: Features
            y: Labels
            stress_configs: List of stress test configurations

        Returns:
            Dictionary of results for each stress test
        """
        results = {}

        for config in stress_configs:
            stress_type = config["type"]
            params = config.get("params", {})
            name = config.get("name", stress_type)

            # Apply stress test
            if stress_type == "gaussian_noise":
                X_stressed = self.add_gaussian_noise(X, **params)
            elif stress_type == "uniform_noise":
                X_stressed = self.add_uniform_noise(X, **params)
            elif stress_type == "feature_dropout":
                X_stressed = self.feature_dropout(X, **params)
            elif stress_type == "feature_corruption":
                X_stressed = self.feature_corruption(X, **params)
            elif stress_type == "scale_perturbation":
                X_stressed = self.scale_perturbation(X, **params)
            elif stress_type == "distribution_shift":
                X_stressed = self.distribution_shift(X, **params)
            else:
                continue

            # Evaluate
            result = self.evaluate_stress_test(model, X, X_stressed, y)
            result["config"] = config
            results[name] = result

        return results
