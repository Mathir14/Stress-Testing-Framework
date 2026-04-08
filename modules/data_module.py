"""
Data Management Module
Purpose: Prepare dataset for modeling
Features: Upload, Validation, Cleaning, Encoding, Scaling, Train/Val/Test Split
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataManager:
    """Manages all data operations including loading, validation, preprocessing, and splitting"""

    def __init__(self):
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

    def load_dataset(self, uploaded_file) -> pd.DataFrame:
        """
        Load CSV dataset from uploaded file

        Args:
            uploaded_file: Streamlit UploadedFile object

        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            self.raw_data = pd.read_csv(uploaded_file)
            st.success(f"✅ Dataset loaded successfully! Shape: {self.raw_data.shape}")
            return self.raw_data
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")
            return None

    def validate_dataset(self, df: pd.DataFrame) -> Dict:
        """
        Validate dataset and return summary statistics

        Args:
            df: Input DataFrame

        Returns:
            Dict: Validation results and statistics
        """
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

    def display_summary(self, df: pd.DataFrame):
        """
        Display comprehensive dataset summary

        Args:
            df: Input DataFrame
        """
        st.subheader("📊 Dataset Summary")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Duplicates", df.duplicated().sum())

        # Display first few rows
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        # Column information
        st.subheader("📋 Column Information")
        col_info = pd.DataFrame(
            {
                "Data Type": df.dtypes,
                "Non-Null Count": df.count(),
                "Null Count": df.isnull().sum(),
                "Null %": (df.isnull().sum() / len(df) * 100).round(2),
                "Unique Values": df.nunique(),
            }
        )
        st.dataframe(col_info, use_container_width=True)

        # Statistical summary
        st.subheader("📈 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

    def handle_missing_values(self, df: pd.DataFrame, strategy: Dict) -> pd.DataFrame:
        """
        Handle missing values based on specified strategy

        Args:
            df: Input DataFrame
            strategy: Dictionary mapping column names to strategies
                     ('mean', 'median', 'mode', 'drop', 'forward_fill', 'backward_fill', or custom value)

        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        df_copy = df.copy()

        for column, method in strategy.items():
            if column not in df_copy.columns:
                continue

            if method == "mean":
                df_copy[column].fillna(df_copy[column].mean(), inplace=True)
            elif method == "median":
                df_copy[column].fillna(df_copy[column].median(), inplace=True)
            elif method == "mode":
                df_copy[column].fillna(df_copy[column].mode()[0], inplace=True)
            elif method == "drop":
                df_copy.dropna(subset=[column], inplace=True)
            elif method == "forward_fill":
                df_copy[column].fillna(method="ffill", inplace=True)
            elif method == "backward_fill":
                df_copy[column].fillna(method="bfill", inplace=True)
            else:
                # Custom value
                df_copy[column].fillna(method, inplace=True)

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
        df_copy = df.copy()

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
        """
        if self.X_train is None:
            return None

        summary = {
            "train": {
                "samples": len(self.X_train),
                "percentage": len(self.X_train)
                / (len(self.X_train) + len(self.X_val) + len(self.X_test))
                * 100,
                "class_distribution": self.y_train.value_counts().to_dict(),
            },
            "validation": {
                "samples": len(self.X_val),
                "percentage": len(self.X_val)
                / (len(self.X_train) + len(self.X_val) + len(self.X_test))
                * 100,
                "class_distribution": self.y_val.value_counts().to_dict(),
            },
            "test": {
                "samples": len(self.X_test),
                "percentage": len(self.X_test)
                / (len(self.X_train) + len(self.X_val) + len(self.X_test))
                * 100,
                "class_distribution": self.y_test.value_counts().to_dict(),
            },
        }

        return summary

    def save_processed_data(self, df: pd.DataFrame):
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
