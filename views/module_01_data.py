from __future__ import annotations

import pandas as pd
import streamlit as st

from core.errors import DatasetLoadError
from core.state import AppContext, StateKeys
from modules.data_module import DataManager
from views.reporters import render_error, render_errors


def render(ctx: AppContext) -> None:
    """Render the Data Management page.

    Args:
        ctx: The per-script-run application context.

    Returns:
        None
    """

    data_manager = ctx.ensure_service(StateKeys.DATA_MANAGER, DataManager)

    st.header("1️⃣ Data Management Module")
    st.markdown("**Purpose:** Prepare dataset for modeling")

    # Create tabs for different data operations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📁 Upload Data",
            "🔍 Validation",
            "🧹 Cleaning",
            "🔄 Encoding & Scaling",
            "✂️ Train/Val/Test Split",
        ]
    )

    # TAB 1: Upload Data
    with tab1:
        st.subheader("📁 Dataset Upload")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file is not None:
            if st.button("Load Dataset"):
                # ADR-011.8: ``load_dataset`` raises ``DatasetLoadError`` instead
                # of returning ``None`` after printing an error, so the old
                # ``if data is not None:`` guard becomes this ``except`` arm.
                try:
                    data = data_manager.load_dataset(uploaded_file)
                except DatasetLoadError as exc:
                    render_error(exc)
                else:
                    ctx.store[StateKeys.DATA_LOADED] = True
                    data_manager.display_summary(data)

        if ctx.store[StateKeys.DATA_LOADED]:
            if data_manager.raw_data is not None:
                st.success("✅ Dataset is loaded and ready!")

    # TAB 2: Validation
    with tab2:
        st.subheader("🔍 Data Validation")

        if data_manager.raw_data is not None:
            df = data_manager.raw_data

            if st.button("Run Validation"):
                with render_errors():
                    # conventions §7: every mutating action reports success **or** a rendered
                    # ``FrameworkError``.  ``render_errors`` wraps the whole handler rather than
                    # each domain call, because a rejected call must not fall through into the
                    # display code that reads the variable it was going to bind.
                    validation_report = data_manager.validate_dataset(df)

                    # Display validation results
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Total Rows", validation_report["shape"][0])
                        st.metric("Total Columns", validation_report["shape"][1])

                    with col2:
                        st.metric("Numeric Columns", len(validation_report["numeric_columns"]))
                        st.metric(
                            "Categorical Columns",
                            len(validation_report["categorical_columns"]),
                        )

                    with col3:
                        st.metric("Duplicate Rows", validation_report["duplicates"])
                        st.metric("Memory Usage (MB)", f"{validation_report['memory_usage']:.2f}")

                    # Missing values details
                    st.subheader("Missing Values Analysis")
                    missing_df = pd.DataFrame(
                        {
                            "Column": validation_report["missing_values"].keys(),
                            "Missing Count": validation_report["missing_values"].values(),
                            "Missing %": [
                                f"{v:.2f}%" for v in validation_report["missing_percentage"].values()
                            ],
                        }
                    )
                    missing_df = missing_df[missing_df["Missing Count"] > 0]

                    if len(missing_df) > 0:
                        st.warning("⚠️ Columns with missing values:")
                        st.dataframe(missing_df, width="stretch")
                    else:
                        st.success("✅ No missing values detected!")

                    # Column types
                    st.subheader("Column Types")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Numeric Columns:**")
                        st.write(validation_report["numeric_columns"])
                    with col2:
                        st.write("**Categorical Columns:**")
                        st.write(validation_report["categorical_columns"])
        else:
            st.warning("⚠️ Please upload a dataset first!")

    # TAB 3: Cleaning
    with tab3:
        st.subheader("🧹 Data Cleaning")

        if data_manager.raw_data is not None:
            df = data_manager.raw_data

            # Handle duplicates
            st.write("**Remove Duplicates**")
            if df.duplicated().sum() > 0:
                if st.button("Remove Duplicate Rows"):
                    with render_errors():
                        # conventions §7: every mutating action reports success **or** a rendered
                        # ``FrameworkError``.  ``render_errors`` wraps the whole handler rather than
                        # each domain call, because a rejected call must not fall through into the
                        # display code that reads the variable it was going to bind.
                        df = df.drop_duplicates()
                        data_manager.raw_data = df
                        st.success(f"✅ Removed duplicates! New shape: {df.shape}")
            else:
                st.info("No duplicates found")

            st.markdown("---")

            # Handle missing values
            st.write("**Handle Missing Values**")
            missing_cols = df.columns[df.isnull().any()].tolist()

            if missing_cols:
                st.write(f"Columns with missing values: {len(missing_cols)}")

                strategy = {}
                for col in missing_cols:
                    st.write(f"**{col}** ({df[col].dtype})")
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        if df[col].dtype in ["int64", "float64"]:
                            method = st.selectbox(
                                f"Strategy for {col}",
                                [
                                    "mean",
                                    "median",
                                    "drop",
                                    "forward_fill",
                                    "backward_fill",
                                ],
                                key=f"missing_{col}",
                            )
                        else:
                            method = st.selectbox(
                                f"Strategy for {col}",
                                ["mode", "drop", "forward_fill", "backward_fill"],
                                key=f"missing_{col}",
                            )

                    with col2:
                        st.metric("Missing", df[col].isnull().sum())

                    strategy[col] = method

                if st.button("Apply Missing Value Handling"):
                    with render_errors():
                        # conventions §7: every mutating action reports success **or** a rendered
                        # ``FrameworkError``.  ``render_errors`` wraps the whole handler rather than
                        # each domain call, because a rejected call must not fall through into the
                        # display code that reads the variable it was going to bind.
                        df_cleaned = data_manager.handle_missing_values(df, strategy)
                        data_manager.raw_data = df_cleaned
                        st.success("✅ Missing values handled successfully!")
                        st.rerun()
            else:
                st.success("✅ No missing values to handle!")
        else:
            st.warning("⚠️ Please upload a dataset first!")

    # TAB 4: Encoding & Scaling
    with tab4:
        st.subheader("🔄 Encoding & Scaling")

        if data_manager.raw_data is not None:
            df = data_manager.raw_data

            # Encoding section
            st.write("**Categorical Encoding**")
            categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

            if categorical_cols:
                st.write(f"Categorical columns found: {categorical_cols}")

                cols_to_encode = st.multiselect("Select columns to encode:", categorical_cols)

                if cols_to_encode:
                    encoding_method = st.radio(
                        "Encoding method:",
                        ["label", "onehot"],
                        help="Label Encoding: Convert to numbers. One-Hot: Create binary columns.",
                    )

                    if st.button("Apply Encoding"):
                        with render_errors():
                            # conventions §7: every mutating action reports success **or** a rendered
                            # ``FrameworkError``.  ``render_errors`` wraps the whole handler rather than
                            # each domain call, because a rejected call must not fall through into the
                            # display code that reads the variable it was going to bind.
                            df_encoded = data_manager.encode_categorical(
                                df, cols_to_encode, encoding_method
                            )
                            data_manager.raw_data = df_encoded
                            st.success("✅ Encoding applied successfully!")
                            st.write("New shape:", df_encoded.shape)
                            st.rerun()
            else:
                st.info("No categorical columns found")

            st.markdown("---")
            st.info(
                "💡 **Note:** Feature scaling will be applied automatically during train/val/test split"
            )
        else:
            st.warning("⚠️ Please upload a dataset first!")

    # TAB 5: Train/Val/Test Split
    with tab5:
        st.subheader("✂️ Train/Validation/Test Split")

        if data_manager.raw_data is not None:
            df = data_manager.raw_data

            # Select target column
            target_col = st.selectbox("Select Target Column:", df.columns.tolist())

            col1, col2, col3 = st.columns(3)

            with col1:
                test_size = st.slider("Test Set Size", 0.1, 0.3, 0.2, 0.05)
            with col2:
                val_size = st.slider("Validation Set Size", 0.05, 0.2, 0.1, 0.05)
            with col3:
                random_state = st.number_input("Random State", 1, 999, 42)

            train_size = 1 - test_size - val_size

            # Display split proportions
            st.write("**Split Proportions:**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Train", f"{train_size * 100:.1f}%")
            col2.metric("Validation", f"{val_size * 100:.1f}%")
            col3.metric("Test", f"{test_size * 100:.1f}%")

            # Scale features option
            apply_scaling = st.checkbox("Apply Feature Scaling (StandardScaler)", value=True)

            if st.button("Split Dataset", type="primary"):
                # Perform split
                with render_errors():
                    # conventions §7: every mutating action reports success **or** a rendered
                    # ``FrameworkError``.  ``render_errors`` wraps the whole handler rather than
                    # each domain call, because a rejected call must not fall through into the
                    # display code that reads the variable it was going to bind.
                    X_train, X_val, X_test, y_train, y_val, y_test = data_manager.split_data(
                        df, target_col, test_size, val_size, random_state
                    )

                    # Apply scaling if requested
                    if apply_scaling:
                        X_train, X_val, X_test = data_manager.scale_features(X_train, X_val, X_test)
                        data_manager.X_train = X_train
                        data_manager.X_val = X_val
                        data_manager.X_test = X_test

                    st.success("✅ Dataset split successfully!")

                    # Display split summary
                    summary = data_manager.get_split_summary()

                    st.subheader("📊 Split Summary")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.write("**Training Set**")
                        st.metric("Samples", summary["train"]["samples"])
                        st.write("Class Distribution:", summary["train"]["class_distribution"])

                    with col2:
                        st.write("**Validation Set**")
                        st.metric("Samples", summary["validation"]["samples"])
                        st.write(
                            "Class Distribution:",
                            summary["validation"]["class_distribution"],
                        )

                    with col3:
                        st.write("**Test Set**")
                        st.metric("Samples", summary["test"]["samples"])
                        st.write("Class Distribution:", summary["test"]["class_distribution"])

                    if apply_scaling:
                        st.info("✅ Feature scaling applied using StandardScaler")

                    ctx.store[StateKeys.DATA_PREPARED] = True
        else:
            st.warning("⚠️ Please upload a dataset first!")

    # ==================== OTHER MODULES (Placeholders) ====================


__all__ = ["render"]
