"""
Data Management Module
Purpose: Prepare dataset for modeling
Features: Upload, Validation, Cleaning, Encoding, Scaling, Train/Val/Test Split
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from core.errors import DatasetError, DatasetLoadError, ValidationError
from core.reporters import NullReporter, Reporter

logger = logging.getLogger(__name__)

#: Missing-value strategy names accepted by :meth:`DataManager.handle_missing_values`.
FILL_STRATEGIES: Tuple[str, ...] = (
    "mean",
    "median",
    "mode",
    "drop",
    "forward_fill",
    "backward_fill",
)


class DataManager:
    """Manages all data operations including loading, validation, preprocessing, and splitting"""

    def __init__(self, reporter: Reporter | None = None):
        self.raw_data = None
        self.processed_data = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scaler = None
        self.label_encoders = {}
        self.target_column = None
        self.feature_columns = None
        self.reporter: Reporter = reporter if reporter is not None else NullReporter()

    def load_dataset(self, uploaded_file: Any) -> pd.DataFrame:
        """
        Load CSV dataset from uploaded file

        Args:
            uploaded_file: File-like object accepted by ``pandas.read_csv``
                (Streamlit's ``UploadedFile`` satisfies this)

        Returns:
            pd.DataFrame: Loaded dataset

        Raises:
            DatasetLoadError: If the file cannot be read or parsed.  The
                pre-remediation implementation called ``st.error`` and returned
                ``None``, which conflated "no data" with "failed" (ADR-004,
                architecture.md §9 item 1).
        """
        try:
            self.raw_data = pd.read_csv(uploaded_file)
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            logger.exception("Failed to load dataset")
            raise DatasetLoadError(
                f"Could not read the uploaded dataset: {exc}. Confirm the file "
                "is a valid, comma-separated CSV with a header row.",
                context={"error_type": type(exc).__name__},
            ) from exc
        except Exception as exc:  # noqa: BLE001 - re-typed at the boundary
            logger.exception("Unexpected failure loading dataset")
            raise DatasetLoadError(
                f"Could not read the uploaded dataset: {exc}.",
                context={"error_type": type(exc).__name__},
            ) from exc

        self.reporter.success(
            f"✅ Dataset loaded successfully! Shape: {self.raw_data.shape}"
        )
        return self.raw_data

    def validate_dataset(self, df: pd.DataFrame) -> Dict:
        """
        Validate dataset and return summary statistics

        Args:
            df: Input DataFrame

        Returns:
            Dict: Validation results and statistics
        """
        if not isinstance(df, pd.DataFrame):
            raise ValidationError(
                f"validate_dataset expects a pandas DataFrame, got "
                f"{type(df).__name__}.",
                context={"received_type": type(df).__name__},
            )
        if df.shape[0] == 0:
            raise DatasetError(
                "The dataset has no rows; nothing to validate.",
                context={"shape": list(df.shape)},
            )

        validation_report = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            "duplicates": df.duplicated().sum(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(
                include=["object"]
            ).columns.tolist(),
            "memory_usage": df.memory_usage(deep=True).sum() / 1024**2,  # MB
        }

        return validation_report

    def get_summary_frames(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Build the dataset-summary tables without rendering them.

        Split out of :meth:`display_summary` so that the presentation concern
        lives in ``views/module_01_data.py`` while the layout-independent
        computation stays here and remains testable (architecture.md §3).

        Args:
            df: Loaded dataset.

        Returns:
            Dict[str, pd.DataFrame]: ``preview``, ``columns`` and ``statistics``.

        Raises:
            DatasetError: If ``df`` has no rows.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValidationError(
                f"get_summary_frames expects a pandas DataFrame, got "
                f"{type(df).__name__}.",
                context={"received_type": type(df).__name__},
            )
        if df.shape[0] == 0:
            raise DatasetError(
                "Cannot build a dataset summary for a frame with no rows.",
                context={"shape": list(df.shape)},
            )

        column_info = pd.DataFrame(
            {
                "Data Type": df.dtypes,
                "Non-Null Count": df.count(),
                "Null Count": df.isnull().sum(),
                "Null %": (df.isnull().sum() / len(df) * 100).round(2),
                "Unique Values": df.nunique(),
            }
        )
        return {
            "preview": df.head(10),
            "columns": column_info,
            "statistics": df.describe(),
        }

    def display_summary(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Delegating wrapper kept for API compatibility

        The rendering moved to ``views/module_01_data.py`` in Wave 4; this method
        now only computes the summary tables and returns them, so the frozen
        public contract in architecture.md §9 survives the Streamlit
        decoupling.

        Args:
            df: Input DataFrame

        Returns:
            Dict[str, pd.DataFrame]: The summary tables a view should render.
        """
        return self.get_summary_frames(df)

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: Dict[str, Any],
        custom_fill_value: float = 0.0,
    ) -> pd.DataFrame:
        """
        Handle missing values based on specified strategy

        Chained assignment with ``inplace=True`` is banned by conventions §5.
        The pre-remediation form (``df[col].fillna(v, inplace=True)``) emitted a
        ``FutureWarning`` on the ``forward_fill`` / ``backward_fill`` branches
        and mutates a throwaway ``Series``, which breaks the moment pandas
        Copy-on-Write is enabled or pandas 3.0 lands.  Every branch now assigns
        the returned frame/Series explicitly.

        Args:
            df: Input DataFrame
            strategy: Mapping of column name to one of :data:`FILL_STRATEGIES`,
                or — for the custom branch — the literal string
                ``"custom"``
            custom_fill_value: Value used when a column's strategy is
                ``"custom"``.  This parameter replaces the previous overload
                where the loop variable doubled as both the strategy name and
                the fill value, which made the accepted vocabulary
                un-introspectable.

        Returns:
            pd.DataFrame: A new DataFrame with missing values handled

        Raises:
            ValidationError: If ``strategy`` is not a mapping, or a column maps
                to a value that is neither a known strategy nor ``"custom"``.
        """
        if not isinstance(strategy, dict):
            raise ValidationError(
                f"strategy must be a mapping of column -> strategy, got "
                f"{type(strategy).__name__}.",
                context={"received_type": type(strategy).__name__},
            )

        df_copy = df.copy(deep=True)
        applied: Dict[str, str] = {}
        skipped: List[str] = []

        for column, method in strategy.items():
            if column not in df_copy.columns:
                skipped.append(str(column))
                logger.warning(
                    "handle_missing_values: column %r is not in the frame; "
                    "strategy skipped.",
                    column,
                )
                continue

            series = df_copy[column]

            if method == "mean":
                fill = series.mean()
            elif method == "median":
                fill = series.median()
            elif method == "mode":
                modes = series.mode()
                if modes.empty:
                    logger.warning(
                        "mode strategy skipped for column %r: every value is "
                        "missing.",
                        column,
                    )
                    skipped.append(str(column))
                    continue
                fill = modes.iloc[0]
            elif method == "drop":
                df_copy = df_copy.dropna(subset=[column])
                applied[str(column)] = "drop"
                continue
            elif method == "forward_fill":
                df_copy[column] = series.ffill()
                applied[str(column)] = "forward_fill"
                continue
            elif method == "backward_fill":
                df_copy[column] = series.bfill()
                applied[str(column)] = "backward_fill"
                continue
            elif method == "custom":
                fill = custom_fill_value
            else:
                raise ValidationError(
                    f"Unknown missing-value strategy {method!r} for column "
                    f"{column!r}. Accepted strategies are: "
                    f"{', '.join((*FILL_STRATEGIES, 'custom'))}, or pass a "
                    "numeric value together with strategy='custom'.",
                    field=str(column),
                    value=method,
                    context={
                        "column": str(column),
                        "accepted": [*FILL_STRATEGIES, "custom"],
                    },
                )

            if pd.isna(fill):
                logger.warning(
                    "Statistic strategy %r skipped for column %r: the statistic "
                    "is NaN because every value is missing.",
                    method,
                    column,
                )
                skipped.append(str(column))
                continue

            df_copy[column] = series.fillna(fill)
            applied[str(column)] = str(method)

        logger.info(
            "handle_missing_values applied %d strategy/-ies, skipped %d",
            len(applied),
            len(skipped),
        )
        return df_copy

    def encode_categorical(
        self, df: pd.DataFrame, columns: List[str], method: str = "label"
    ) -> pd.DataFrame:
        """
        Encode categorical variables

        Args:
            df: Input DataFrame
            columns: List of columns to encode
            method: 'label' for Label Encoding or 'onehot' for One-Hot Encoding

        Returns:
            pd.DataFrame: DataFrame with encoded columns
        """
        df_copy = df.copy(deep=True)

        if method == "label":
            for col in columns:
                if col in df_copy.columns and df_copy[col].dtype == "object":
                    le = LabelEncoder()
                    df_copy[col] = le.fit_transform(df_copy[col].astype(str))
                    self.label_encoders[col] = le

        elif method == "onehot":
            df_copy = pd.get_dummies(df_copy, columns=columns, drop_first=True)

        return df_copy

    def scale_features(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame = None,
        X_test: pd.DataFrame = None,
    ) -> Tuple:
        """
        Scale features using StandardScaler

        Args:
            X_train: Training features
            X_val: Validation features (optional)
            X_test: Test features (optional)

        Returns:
            Tuple: Scaled datasets
        """
        self.scaler = StandardScaler()

        # Fit on training data only
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )

        # Transform validation and test sets
        X_val_scaled = None
        X_test_scaled = None

        if X_val is not None:
            X_val_scaled = pd.DataFrame(
                self.scaler.transform(X_val), columns=X_val.columns, index=X_val.index
            )

        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index,
            )

        return X_train_scaled, X_val_scaled, X_test_scaled

    def split_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> Tuple:
        """
        Split data into train, validation, and test sets

        Args:
            df: Input DataFrame
            target_column: Name of target column
            test_size: Proportion of test set (0-1)
            val_size: Proportion of validation set from remaining data (0-1)
            random_state: Random seed for reproducibility

        Returns:
            Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        self.target_column = target_column

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        self.feature_columns = X.columns.tolist()

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if y.nunique() < 10 else None,
        )

        # Second split: separate validation set from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp if y_temp.nunique() < 10 else None,
        )

        # Store splits
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_split_summary(self) -> Dict:
        """
        Get summary of data splits

        Returns:
            Dict: Summary statistics of splits

        Raises:
            DatasetError: If :meth:`split_data` has not been called, if any
                split is missing, or if the splits are empty.  The
                pre-remediation implementation returned ``None`` before
                splitting — forbidden by conventions §3 as a failure signal —
                and divided by the total split length, which raises
                ``ZeroDivisionError`` when every split is empty.  ``X_val`` was
                additionally unguarded, so ``len(None)`` raised ``TypeError``.
        """
        splits = {
            "train": (self.X_train, self.y_train),
            "validation": (self.X_val, self.y_val),
            "test": (self.X_test, self.y_test),
        }

        missing = [
            name for name, (features, _labels) in splits.items() if features is None
        ]
        if missing:
            raise DatasetError(
                f"Cannot summarise splits: {', '.join(missing)} "
                f"{'is' if len(missing) == 1 else 'are'} not available. Call "
                "split_data() first.",
                context={"missing_splits": missing},
            )

        total = sum(len(features) for features, _labels in splits.values())
        if total == 0:
            raise DatasetError(
                "Cannot summarise splits: every split is empty.",
                context={"total_samples": 0},
            )

        summary = {}
        for name, (features, labels) in splits.items():
            if features is None:  # pragma: no cover - guarded above
                continue
            summary[name] = {
                "samples": len(features),
                "percentage": len(features) / total * 100,
                "class_distribution": labels.value_counts().to_dict(),
            }

        return summary

    def save_processed_data(self, df: pd.DataFrame) -> None:
        """Save processed data"""
        self.processed_data = df

    def get_data(self) -> Dict:
        """
        Get all data splits and related objects

        Returns:
            Dict: Dictionary containing all data and objects
        """
        return {
            "raw_data": self.raw_data,
            "processed_data": self.processed_data,
            "X_train": self.X_train,
            "X_val": self.X_val,
            "X_test": self.X_test,
            "y_train": self.y_train,
            "y_val": self.y_val,
            "y_test": self.y_test,
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "target_column": self.target_column,
            "feature_columns": self.feature_columns,
        }
