# ML Model Reliability & Stress Testing Framework

A comprehensive framework for testing ML model robustness and reliability under various stress conditions.

## 🚀 Quick Start

### Run the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📋 Module 1: Data Management ✅

### Features Implemented:

- ✅ **Dataset Upload** - Upload CSV files
- ✅ **Data Validation** - Comprehensive validation and statistics
- ✅ **Missing Value Handling** - Multiple strategies (mean, median, mode, drop, fill)
- ✅ **Encoding** - Label encoding and One-Hot encoding for categorical variables
- ✅ **Feature Scaling** - StandardScaler normalization
- ✅ **Train/Val/Test Split** - Configurable split ratios with stratification
- ✅ **Dataset Summary** - Interactive data exploration and statistics

### How to Use:

1. Navigate to "1️⃣ Data Management" in the sidebar
2. **Upload Data** tab: Upload your CSV file
3. **Validation** tab: Review data quality and statistics
4. **Cleaning** tab: Handle duplicates and missing values
5. **Encoding & Scaling** tab: Encode categorical variables
6. **Train/Val/Test Split** tab: Split data and apply scaling

## � Module 2: Baseline Modeling ✅

### Features Implemented:

- ✅ **Model Training** - Logistic Regression, Random Forest, XGBoost
- ✅ **Hyperparameter Configuration** - Customizable model parameters
- ✅ **Model Evaluation** - Accuracy, Precision, Recall, F1-score
- ✅ **Confusion Matrix** - Interactive confusion matrix visualization
- ✅ **Model Comparison** - Compare performance across models
- ✅ **Feature Importance** - Visualize feature importance (RF, XGBoost)
- ✅ **Probability Extraction** - Get prediction probabilities
- ✅ **Model Persistence** - Save trained models

### How to Use:

1. Navigate to "2️⃣ Baseline Modeling" in the sidebar (after preparing data in Module 1)
2. **Model Selection & Training** tab:
   - Select model (Logistic Regression, Random Forest, or XGBoost)
   - Configure hyperparameters
   - Train model and view quick results
3. **Model Evaluation** tab:
   - Select trained model and dataset (Train/Val/Test)
   - View detailed metrics, confusion matrix, and classification report
   - Compare all trained models
4. **Feature Importance** tab:
   - View feature importance for tree-based models
   - Identify most influential features
5. **Save/Load Models** tab:
   - Save trained models to disk
   - View best performing model by selected metric

## 🎯 Module 3: Prediction & Confidence ✅

### Features Implemented:

- ✅ **Predictions Overview** - Generate predictions on validation/test sets
- ✅ **Confidence Scores** - Extract max probability as confidence for each prediction
- ✅ **Confidence Analysis** - Visual analysis of confidence distributions
- ✅ **High-Confidence Errors** - Identify critical misclassifications
- ✅ **Calibration Curves** - Assess whether confidence matches accuracy
- ✅ **Brier Score** - Quantitative calibration metric
- ✅ **Entropy Analysis** - Measure prediction uncertainty
- ✅ **Class-wise Confidence** - Compare confidence across different classes

### How to Use:

1. Navigate to "3️⃣ Prediction & Confidence" in the sidebar (after training models in Module 2)
2. **Predictions Overview** tab:
   - Select a trained model and dataset (Validation/Test)
   - Generate predictions with confidence scores
   - View overall statistics and detailed predictions
   - Download predictions CSV
3. **Confidence Analysis** tab:
   - Adjust confidence threshold slider
   - View confidence distribution (correct vs incorrect predictions)
   - Analyze class-wise confidence patterns
   - Compare high vs low confidence prediction accuracy
4. **High-Confidence Errors** tab:
   - Set minimum confidence threshold for error detection
   - Identify predictions where model was very confident but **wrong**
   - View error distribution by true class
   - Download high-confidence errors for investigation
5. **Calibration & Uncertainty** tab:
   - Examine calibration curve (confidence vs accuracy alignment)
   - View Brier score for overall calibration quality
   - Analyze prediction entropy (uncertainty metric)
   - Review most uncertain predictions

## 🧪 Module 4: Stress Testing ✅

### Features Implemented:

- ✅ **Gaussian Noise Injection** - Add noise based on feature standard deviations
- ✅ **Uniform Noise** - Add uniformly distributed noise
- ✅ **Feature Dropout** - Randomly set features to zero
- ✅ **Feature Corruption** - Corrupt values with zero/mean/random/extreme values
- ✅ **Scale Perturbation** - Randomly scale features
- ✅ **Distribution Shift** - Shift feature means or variances
- ✅ **Single Stress Test** - Run individual stress tests with custom parameters
- ✅ **Batch Stress Testing** - Run multiple stress tests simultaneously
- ✅ **Results Analysis** - Visualize performance degradation
- ✅ **Performance Metrics** - Track accuracy drop, prediction agreement, F1 scores

### How to Use:

1. Navigate to "4️⃣ Stress Testing" in the sidebar (after training models in Module 2)
2. **Single Stress Test** tab:
   - Select a model and dataset (Validation/Test)
   - Choose stress test type (Gaussian Noise, Dropout, Corruption, etc.)
   - Configure parameters with sliders
   - Run test and view immediate results
   - See original vs stressed accuracy, performance drop, and prediction agreement
3. **Batch Stress Tests** tab:
   - Select model and dataset
   - Choose which stress test categories to run
   - Each category includes low/medium/high intensity variants
   - Run all selected tests simultaneously
   - View summary table with all results
4. **Results Analysis** tab:
   - Performance drop bar chart - identify most damaging stress tests
   - Accuracy comparison line chart - original vs stressed
   - Prediction agreement analysis - how many predictions changed
   - Most challenging stress tests ranked by performance drop
5. **Export Results** tab:
   - Download comprehensive CSV with all stress test metrics
   - Preview full results table before downloading

## 📊 Module 5: Post-Stress Evaluation ✅

### Features Implemented:

- ✅ **Robustness Score** - Comprehensive 0-100 score quantifying model robustness
- ✅ **Multi-dimensional Analysis** - Accuracy retention, prediction stability, performance consistency
- ✅ **Vulnerability Analysis** - Identify which stress tests cause the most damage
- ✅ **Stress Type Summary** - Group results by stress category (Noise, Dropout, Corruption, etc.)
- ✅ **Model Comparison** - Compare robustness across multiple trained models
- ✅ **Automated Recommendations** - AI-generated actionable insights for model improvement
- ✅ **Interactive Visualizations** - Radar charts, heatmaps, and bar charts
- ✅ **Severity Classification** - Automatic classification (Low/Medium/High/Critical)

### How to Use:

1. Navigate to "5️⃣ Post-Stress Evaluation" in the sidebar (after running stress tests in Module 4)
2. **Robustness Score** tab:
   - Select a model to analyze
   - Click "Calculate Robustness Score" to get comprehensive metrics
   - View overall score (0-100) and dimensional breakdown:
     - **Accuracy Retention**: How much accuracy is preserved under stress
     - **Prediction Stability**: How consistent predictions are despite perturbations
     - **Performance Consistency**: How uniformly the model performs across tests
   - Examine radar chart showing robustness across all dimensions
   - Get score interpretation (Excellent/Good/Moderate/Low)
3. **Vulnerability Analysis** tab:
   - View which stress tests caused the most performance degradation
   - See vulnerability heatmap highlighting weak points
   - Review detailed table with severity classification
   - Download vulnerability analysis CSV for further investigation
4. **Stress Type Summary** tab:
   - See aggregated results by stress category (Noise, Dropout, Corruption, Scaling, Distribution)
   - Compare average performance drops across categories
   - Analyze prediction agreement patterns
   - Identify which types of perturbations are most damaging
5. **Model Comparison** tab:
   - Select 2+ models to compare their robustness
   - View side-by-side comparison of robustness dimensions
   - See which model is most robust overall
   - Use insights to choose the best model for production
6. **Recommendations** tab:
   - Click "Generate Recommendations" for AI-generated insights
   - Get specific action items based on vulnerability patterns:
     - Noise augmentation for noise-sensitive models
     - Feature engineering for dropout-vulnerable models
     - Outlier handling for corruption-sensitive models
     - Scaling improvements for scale-dependent models
   - Review general best practices for model robustness
   - Download recommendations report

## �📦 Installed Packages

- streamlit
- scikit-learn
- xgboost
- pandas
- numpy
- matplotlib
- plotly
- seaborn
- shap
- reportlab

## 🏗️ Project Structure

```
Mini Project/
│
├── app.py                          # Main Streamlit application ✅
│
├── modules/
│   ├── data_module.py             # Data Management Module ✅
│   ├── model_module.py            # Baseline Modeling Module ✅
│   ├── stress_module.py           # Stress Testing Module ✅
│   ├── post_stress_module.py      # Post-Stress Evaluation Module ✅
│   ├── calibration_module.py      # Calibration Analysis (Coming soon)
│   ├── reliability_module.py      # Reliability Scoring (Coming soon)
│   └── reporting_module.py        # Reporting (Coming soon)
│
├── utils/
│   ├── metrics.py                 # Confidence & uncertainty metrics ✅
│   └── plotting.py                # Confidence & calibration plots ✅
│
└── saved_models/                  # Directory for saved models
```

## 🎯 Upcoming Modules

- 6️⃣ Calibration Analysis
- 7️⃣ Model Comparison
- 8️⃣ Reliability Scoring
- 9️⃣ Visualization & Reports

## 💡 Tips

- Complete modules in order for best results
- Ensure your dataset has a clear target column for classification
- Use validation and test sets to properly evaluate model performance
- Save your trained models before closing the application
- High-confidence errors indicate systematic model failures worth investigating
- Well-calibrated models have confidence scores that match their actual accuracy
