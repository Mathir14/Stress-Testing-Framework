from __future__ import annotations

import pandas as pd
import streamlit as st

from core.errors import FrameworkError
from core.perturbations import apply_perturbation, operation_key
from core.state import AppContext, StateKeys
from modules.data_module import DataManager
from modules.model_module import ModelTrainer
from modules.stress_module import StressTester
from views.reporters import render_error


def render(ctx: AppContext) -> None:
    """Render the Stress Testing page.

    Args:
        ctx: The per-script-run application context.

    Returns:
        None
    """

    data_manager = ctx.ensure_service(StateKeys.DATA_MANAGER, DataManager)
    model_trainer = ctx.ensure_service(StateKeys.MODEL_TRAINER, ModelTrainer)
    stress_tester = ctx.ensure_service(StateKeys.STRESS_TESTER, StressTester)

    st.header("4️⃣ Stress Testing Module")
    st.markdown("Test model robustness under various data perturbations and stress conditions.")

    # Check prerequisites
    #
    # The pre-extraction guards read ``st.session_state.data_manager is None``
    # and ``st.session_state.model_trainer is None``.  Both were unreachable:
    # app.py's bootstrap block created the singletons before any page ran.
    # ``ctx.has_data()`` / ``ctx.has_models()`` are the §4.7 replacements and
    # are equivalent in every reachable state — a model can only be trained
    # after data is loaded — while being strictly more informative about which
    # prerequisite is missing.
    if not ctx.has_data():
        st.warning("⚠️ Please load and prepare data in Module 1 first.")
    elif not ctx.has_models():
        st.warning("⚠️ Please train at least one model in Module 2 first.")
    else:
        # ADR-011.4: rebind the singleton to *this* script run's generator.
        # ``batch_stress_test`` is a frozen signature that expands
        # ``self.<op>(X, **params)``, so it has no place to take an explicit
        # ``rng``; assigning the per-run generator is what makes the batch path
        # honour ``ctx.rng`` without changing the frozen API.
        stress_tester.rng = ctx.rng

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "🎯 Single Stress Test",
                "📊 Batch Stress Tests",
                "📈 Results Analysis",
                "💾 Export Results",
            ]
        )

        with tab1:
            st.subheader("Single Stress Test")
            st.markdown("Apply a single stress test and immediately view results.")

            col1, col2 = st.columns(2)

            with col1:
                # Model selection
                model_names = list(model_trainer.trained_models.keys())
                selected_model = st.selectbox(
                    "Select Model", model_names, key="stress_model_select"
                )

                # Dataset selection
                dataset_choice = st.radio(
                    "Select Dataset",
                    ["Validation Set", "Test Set"],
                    horizontal=True,
                    key="stress_dataset_select",
                )

            with col2:
                # Stress test type
                stress_type = st.selectbox(
                    "Stress Test Type",
                    [
                        "Gaussian Noise",
                        "Uniform Noise",
                        "Feature Dropout",
                        "Feature Corruption",
                        "Scale Perturbation",
                        "Distribution Shift",
                    ],
                    key="single_stress_type",
                )

            # Parameters based on stress type
            #
            # ADR-011.2: this is the first of the two UI label dispatch chains.
            # The label is translated to its registry key *once*, and both this
            # chain and the apply chain below branch on that key — so a label
            # can never be added to the selectbox without a matching operator,
            # and the ``if/elif`` ladders can no longer fall through silently
            # (which is what left ``params`` undefined before).
            stress_op = operation_key(stress_type)
            st.markdown("#### Configuration")

            if stress_op == "gaussian_noise":
                noise_level = st.slider(
                    "Noise Level (std multiplier)",
                    0.0,
                    1.0,
                    0.1,
                    0.05,
                    help="Noise = Normal(0, feature_std * noise_level)",
                )
                params = {"noise_level": noise_level}

            elif stress_op == "uniform_noise":
                noise_range = st.slider(
                    "Noise Range (fraction of feature range)",
                    0.0,
                    0.5,
                    0.1,
                    0.05,
                    help="Noise uniformly distributed in [-range*feature_range, +range*feature_range]",
                )
                params = {"noise_range": noise_range}

            elif stress_op == "feature_dropout":
                dropout_rate = st.slider(
                    "Dropout Rate",
                    0.0,
                    0.8,
                    0.2,
                    0.05,
                    help="Fraction of features to randomly set to zero",
                )
                params = {"dropout_rate": dropout_rate}

            elif stress_op == "feature_corruption":
                corruption_rate = st.slider(
                    "Corruption Rate",
                    0.0,
                    0.5,
                    0.1,
                    0.05,
                    help="Fraction of values to corrupt",
                )
                corruption_type_choice = st.selectbox(
                    "Corruption Type",
                    ["zero", "mean", "random", "extreme"],
                    help="How to corrupt values",
                )
                params = {
                    "corruption_rate": corruption_rate,
                    "corruption_type": corruption_type_choice,
                }

            elif stress_op == "scale_perturbation":
                scale_factor = st.slider(
                    "Scale Factor",
                    # ADR-015: the floor is 1.1, not 1.0.  1.0 *is*
                    # ``StressBounds.scale_factor``'s exclusive lower bound, so a
                    # slider starting there published a value the kernel must
                    # reject — conventions §7 then made that rejection a
                    # rendered error, which is correct but still a dead control
                    # position.  1.1 is the smallest value this slider's own 0.1
                    # step can express above the bound, so no granularity
                    # changes and the 1.5 default is untouched.  The bound stays
                    # exclusive: ``scale_perturbation`` perturbs by ``f`` or
                    # ``1/f``, so ``f == 1.0`` is the identity (a no-op that
                    # reports a passing stress test) and ``f == 0.0`` is the
                    # division-by-zero path.  Both ends are genuinely degenerate,
                    # so the kernel bound is right and the widget was the side
                    # that needed correcting.
                    1.1,
                    3.0,
                    1.5,
                    0.1,
                    help="Features randomly scaled by factor or 1/factor",
                )
                params = {"scale_factor": scale_factor}

            elif stress_op == "distribution_shift":
                shift_type_choice = st.radio("Shift Type", ["mean", "variance"], horizontal=True)
                shift_amount = st.slider(
                    "Shift Amount",
                    0.0,
                    2.0,
                    0.5,
                    0.1,
                    help="For mean: shift by amount*std. For variance: scale by amount",
                )
                params = {"shift_type": shift_type_choice, "shift_amount": shift_amount}

            if st.button("🧪 Run Stress Test", key="run_single_stress"):
                with st.spinner("Running stress test..."):
                    model = model_trainer.trained_models[selected_model]

                    # Select dataset
                    if dataset_choice == "Validation Set":
                        X = data_manager.X_val
                        y = data_manager.y_val
                    else:
                        X = data_manager.X_test
                        y = data_manager.y_test

                    # Apply stress test, then evaluate
                    #
                    # ADR-011.2/ADR-011.4: the second UI label dispatch chain
                    # collapses into one call.  ``operation_key`` already
                    # produced ``stress_op`` above, so the label→operator
                    # mapping exists in exactly one place, and ``ctx.rng`` is
                    # passed explicitly instead of being drawn from a module
                    # level global.  ``apply_perturbation`` is the same kernel
                    # the six thin ``StressTester`` delegations call, so this is
                    # not a second implementation.
                    #
                    # conventions §7: every mutating action reports success **or**
                    # a rendered ``FrameworkError``.  The domain calls are
                    # therefore inside the ``try``, and the store write is in the
                    # ``else`` arm, so a rejected parameter can never leave a
                    # half-written result behind.
                    #
                    # This guard is defence in depth, not the fix for the defect
                    # Reviewer MAJOR-1 found.  That defect — the Scale Factor
                    # slider starting at the *exclusive* lower bound 1.0, so a
                    # published control position was deterministically rejected —
                    # is fixed at the source in ADR-015 (the slider floor is now
                    # 1.1), and ``tests/test_validation.py`` now sweeps every
                    # position of all six sliders against the kernel rather than
                    # carving out an exception for this one.  The guard stays
                    # because conventions §7 binds *every* mutating action, and
                    # because a widget, a config default or a future batch
                    # parameter can drift from the bounds again without anyone
                    # noticing until a user hits it.  Nothing in the shipped UI
                    # can reach it today, which is exactly why it needs a test
                    # that injects the rejection at the kernel boundary rather
                    # than one that hopes a slider misconfiguration survives.
                    try:
                        X_stressed = apply_perturbation(X, stress_op, params, rng=ctx.rng)
                        result = stress_tester.evaluate_stress_test(model, X, X_stressed, y)
                    except FrameworkError as exc:
                        render_error(exc)
                    else:
                        # Store result.  Only reached on success, so a rejected
                        # parameter can never leave a half-written result behind.
                        ctx.store[StateKeys.SINGLE_STRESS_RESULT] = {
                            "model": selected_model,
                            "dataset": dataset_choice,
                            "stress_type": stress_type,
                            "params": params,
                            "result": result,
                            "X_stressed": X_stressed,
                        }

                        st.success("✅ Stress test completed!")

            # Display results
            if ctx.store[StateKeys.SINGLE_STRESS_RESULT]:
                st.markdown("---")
                st.markdown("### Results")

                result_data = ctx.store[StateKeys.SINGLE_STRESS_RESULT]
                result = result_data["result"]

                st.markdown(f"**Model:** {result_data['model']}")
                st.markdown(f"**Dataset:** {result_data['dataset']}")
                st.markdown(f"**Stress Test:** {result_data['stress_type']}")

                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Original Accuracy", f"{result['accuracy_original']:.2%}")
                with col2:
                    st.metric(
                        "Stressed Accuracy",
                        f"{result['accuracy_stressed']:.2%}",
                        f"{-result['performance_drop']:.2%}",
                    )
                with col3:
                    st.metric("Performance Drop", f"{result['performance_drop_pct']:.1f}%")
                with col4:
                    st.metric("Prediction Agreement", f"{result['prediction_agreement']:.2%}")

                # Additional metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Precision (Stressed)", f"{result['precision_stressed']:.2%}")
                with col2:
                    st.metric("Recall (Stressed)", f"{result['recall_stressed']:.2%}")
                with col3:
                    st.metric("F1 Score (Stressed)", f"{result['f1_stressed']:.2%}")

        with tab2:
            st.subheader("Batch Stress Tests")
            st.markdown(
                "Run multiple stress tests simultaneously to comprehensively evaluate model robustness."
            )

            col1, col2 = st.columns(2)

            with col1:
                batch_model = st.selectbox(
                    "Select Model",
                    list(model_trainer.trained_models.keys()),
                    key="batch_stress_model",
                )

            with col2:
                batch_dataset = st.radio(
                    "Select Dataset",
                    ["Validation Set", "Test Set"],
                    horizontal=True,
                    key="batch_stress_dataset",
                )

            st.markdown("#### Select Stress Tests")

            col1, col2 = st.columns(2)

            with col1:
                test_gaussian = st.checkbox(
                    "Gaussian Noise (Low: 0.05, Medium: 0.15, High: 0.3)", value=True
                )
                test_uniform = st.checkbox(
                    "Uniform Noise (Low: 0.05, Medium: 0.15, High: 0.3)", value=True
                )
                test_dropout = st.checkbox("Feature Dropout (10%, 20%, 30%)", value=True)

            with col2:
                test_corruption = st.checkbox("Feature Corruption (5%, 15%, 25%)", value=True)
                test_scale = st.checkbox("Scale Perturbation (1.5x, 2.0x, 2.5x)", value=True)
                test_shift = st.checkbox(
                    "Distribution Shift (Mean: 0.5, 1.0 / Var: 0.5, 1.5)", value=True
                )

            if st.button("🚀 Run Batch Stress Tests", key="run_batch_stress"):
                with st.spinner("Running batch stress tests..."):
                    model = model_trainer.trained_models[batch_model]

                    # Select dataset
                    if batch_dataset == "Validation Set":
                        X = data_manager.X_val
                        y = data_manager.y_val
                    else:
                        X = data_manager.X_test
                        y = data_manager.y_test

                    # Build stress test configurations
                    stress_configs = []

                    if test_gaussian:
                        stress_configs.extend(
                            [
                                {
                                    "type": "gaussian_noise",
                                    "params": {"noise_level": 0.05},
                                    "name": "Gaussian Noise (Low)",
                                },
                                {
                                    "type": "gaussian_noise",
                                    "params": {"noise_level": 0.15},
                                    "name": "Gaussian Noise (Medium)",
                                },
                                {
                                    "type": "gaussian_noise",
                                    "params": {"noise_level": 0.3},
                                    "name": "Gaussian Noise (High)",
                                },
                            ]
                        )

                    if test_uniform:
                        stress_configs.extend(
                            [
                                {
                                    "type": "uniform_noise",
                                    "params": {"noise_range": 0.05},
                                    "name": "Uniform Noise (Low)",
                                },
                                {
                                    "type": "uniform_noise",
                                    "params": {"noise_range": 0.15},
                                    "name": "Uniform Noise (Medium)",
                                },
                                {
                                    "type": "uniform_noise",
                                    "params": {"noise_range": 0.3},
                                    "name": "Uniform Noise (High)",
                                },
                            ]
                        )

                    if test_dropout:
                        stress_configs.extend(
                            [
                                {
                                    "type": "feature_dropout",
                                    "params": {"dropout_rate": 0.1},
                                    "name": "Feature Dropout (10%)",
                                },
                                {
                                    "type": "feature_dropout",
                                    "params": {"dropout_rate": 0.2},
                                    "name": "Feature Dropout (20%)",
                                },
                                {
                                    "type": "feature_dropout",
                                    "params": {"dropout_rate": 0.3},
                                    "name": "Feature Dropout (30%)",
                                },
                            ]
                        )

                    if test_corruption:
                        stress_configs.extend(
                            [
                                {
                                    "type": "feature_corruption",
                                    "params": {
                                        "corruption_rate": 0.05,
                                        "corruption_type": "zero",
                                    },
                                    "name": "Corruption (5%, zero)",
                                },
                                {
                                    "type": "feature_corruption",
                                    "params": {
                                        "corruption_rate": 0.15,
                                        "corruption_type": "random",
                                    },
                                    "name": "Corruption (15%, random)",
                                },
                                {
                                    "type": "feature_corruption",
                                    "params": {
                                        "corruption_rate": 0.25,
                                        "corruption_type": "extreme",
                                    },
                                    "name": "Corruption (25%, extreme)",
                                },
                            ]
                        )

                    if test_scale:
                        stress_configs.extend(
                            [
                                {
                                    "type": "scale_perturbation",
                                    "params": {"scale_factor": 1.5},
                                    "name": "Scale Perturbation (1.5x)",
                                },
                                {
                                    "type": "scale_perturbation",
                                    "params": {"scale_factor": 2.0},
                                    "name": "Scale Perturbation (2.0x)",
                                },
                                {
                                    "type": "scale_perturbation",
                                    "params": {"scale_factor": 2.5},
                                    "name": "Scale Perturbation (2.5x)",
                                },
                            ]
                        )

                    if test_shift:
                        stress_configs.extend(
                            [
                                {
                                    "type": "distribution_shift",
                                    "params": {
                                        "shift_type": "mean",
                                        "shift_amount": 0.5,
                                    },
                                    "name": "Mean Shift (0.5)",
                                },
                                {
                                    "type": "distribution_shift",
                                    "params": {
                                        "shift_type": "mean",
                                        "shift_amount": 1.0,
                                    },
                                    "name": "Mean Shift (1.0)",
                                },
                                {
                                    "type": "distribution_shift",
                                    "params": {
                                        "shift_type": "variance",
                                        "shift_amount": 0.5,
                                    },
                                    "name": "Variance Shift (0.5)",
                                },
                                {
                                    "type": "distribution_shift",
                                    "params": {
                                        "shift_type": "variance",
                                        "shift_amount": 1.5,
                                    },
                                    "name": "Variance Shift (1.5)",
                                },
                            ]
                        )

                    # Run batch tests
                    #
                    # Same conventions §7 contract as the single-test path: the
                    # 19 hard-coded configurations are all inside their bounds
                    # (pinned by ``test_every_batch_config_value_is_accepted``),
                    # so the reachable domain failure here is the *frame* — a
                    # repeated column label or a non-numeric column rejected by
                    # ``coerce_numeric_frame``.  That is rendered, not raised.
                    try:
                        results = stress_tester.batch_stress_test(model, X, y, stress_configs)
                    except FrameworkError as exc:
                        render_error(exc)
                    else:
                        # Store results per model
                        if ctx.store[StateKeys.BATCH_STRESS_RESULTS_BY_MODEL] is None:
                            ctx.store[StateKeys.BATCH_STRESS_RESULTS_BY_MODEL] = {}
                        ctx.store[StateKeys.BATCH_STRESS_RESULTS_BY_MODEL][batch_model] = results

                        # Also keep last run for backward compat (Results
                        # Analysis tab)
                        ctx.store[StateKeys.BATCH_STRESS_RESULTS] = {
                            "model": batch_model,
                            "dataset": batch_dataset,
                            "results": results,
                        }

                        st.success(f"✅ Completed {len(results)} stress tests!")

            # Display batch results summary
            if ctx.store[StateKeys.BATCH_STRESS_RESULTS]:
                st.markdown("---")
                st.markdown("### Batch Results Summary")

                batch_data = ctx.store[StateKeys.BATCH_STRESS_RESULTS]
                results = batch_data["results"]

                # Create summary DataFrame
                summary_data = []
                for name, result in results.items():
                    summary_data.append(
                        {
                            "Stress Test": name,
                            "Original Acc": f"{result['accuracy_original']:.2%}",
                            "Stressed Acc": f"{result['accuracy_stressed']:.2%}",
                            "Drop %": f"{result['performance_drop_pct']:.1f}%",
                            "Agreement": f"{result['prediction_agreement']:.2%}",
                            "F1": f"{result['f1_stressed']:.2%}",
                        }
                    )

                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, width="stretch")

        with tab3:
            st.subheader("Results Analysis")

            if not ctx.store[StateKeys.BATCH_STRESS_RESULTS]:
                st.info("👈 Run batch stress tests in the 'Batch Stress Tests' tab first.")
            else:
                batch_data = ctx.store[StateKeys.BATCH_STRESS_RESULTS]
                results = batch_data["results"]

                st.markdown(f"**Model:** {batch_data['model']}")
                st.markdown(f"**Dataset:** {batch_data['dataset']}")

                # Performance drop chart
                st.markdown("#### Performance Drop by Stress Test")

                chart_data = pd.DataFrame(
                    {
                        "Stress Test": list(results.keys()),
                        "Performance Drop (%)": [
                            r["performance_drop_pct"] for r in results.values()
                        ],
                        "Accuracy (Stressed)": [
                            r["accuracy_stressed"] * 100 for r in results.values()
                        ],
                    }
                )

                st.bar_chart(chart_data.set_index("Stress Test")["Performance Drop (%)"])

                # Accuracy comparison
                st.markdown("#### Accuracy: Original vs Stressed")

                acc_data = pd.DataFrame(
                    {
                        "Stress Test": list(results.keys()),
                        "Original": [r["accuracy_original"] * 100 for r in results.values()],
                        "Stressed": [r["accuracy_stressed"] * 100 for r in results.values()],
                    }
                )

                st.line_chart(acc_data.set_index("Stress Test"))

                # Prediction agreement
                st.markdown("#### Prediction Agreement")
                st.markdown("Percentage of predictions that remained the same under stress")

                agreement_data = pd.DataFrame(
                    {
                        "Stress Test": list(results.keys()),
                        "Agreement (%)": [
                            r["prediction_agreement"] * 100 for r in results.values()
                        ],
                    }
                )

                st.bar_chart(agreement_data.set_index("Stress Test")["Agreement (%)"])

                # Worst performing stress tests
                st.markdown("#### Most Challenging Stress Tests")
                st.markdown("Stress tests that caused the largest performance drops")

                worst_tests = sorted(
                    results.items(),
                    key=lambda x: x[1]["performance_drop_pct"],
                    reverse=True,
                )[:5]

                worst_df = pd.DataFrame(
                    [
                        {
                            "Stress Test": name,
                            "Performance Drop": f"{result['performance_drop_pct']:.1f}%",
                            "Stressed Accuracy": f"{result['accuracy_stressed']:.2%}",
                        }
                        for name, result in worst_tests
                    ]
                )

                st.dataframe(worst_df, width="stretch")

        with tab4:
            st.subheader("Export Results")

            if not ctx.store[StateKeys.BATCH_STRESS_RESULTS]:
                st.info("👈 Run batch stress tests first to export results.")
            else:
                batch_data = ctx.store[StateKeys.BATCH_STRESS_RESULTS]
                results = batch_data["results"]

                st.markdown("#### Download Stress Test Results")

                # Create comprehensive export DataFrame
                export_data = []
                for name, result in results.items():
                    export_data.append(
                        {
                            "Model": batch_data["model"],
                            "Dataset": batch_data["dataset"],
                            "Stress_Test": name,
                            "Accuracy_Original": result["accuracy_original"],
                            "Accuracy_Stressed": result["accuracy_stressed"],
                            "Performance_Drop": result["performance_drop"],
                            "Performance_Drop_Pct": result["performance_drop_pct"],
                            "Precision_Stressed": result["precision_stressed"],
                            "Recall_Stressed": result["recall_stressed"],
                            "F1_Stressed": result["f1_stressed"],
                            "Prediction_Agreement": result["prediction_agreement"],
                        }
                    )

                export_df = pd.DataFrame(export_data)

                csv = export_df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Stress Test Results CSV",
                    csv,
                    f"stress_test_results_{batch_data['model']}.csv",
                    "text/csv",
                    key="download_stress_results",
                )

                st.markdown("#### Preview")
                st.dataframe(export_df, width="stretch")


__all__ = ["render"]
