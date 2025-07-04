'''# Credit Default Prediction System

A comprehensive machine learning system for predicting credit default risk using advanced algorithms and providing business intelligence insights.

## 🌟 Features

### Core Functionality
- **Individual Predictions**: Real-time credit risk assessment for single customers
- **Batch Processing**: Efficient processing of multiple credit applications
- **Multiple ML Models**: XGBoost, LightGBM, CatBoost, Random Forest, Logistic Regression
- **Ensemble Predictions**: Combined model predictions for improved accuracy
- **Risk Categorization**: Automatic classification into Low, Medium, and High risk categories

### Advanced Analytics
- **Model Explainability**: SHAP and LIME explanations for predictions
- **Feature Importance**: Understanding key factors in credit decisions
- **Business Intelligence**: Comprehensive dashboards and KPIs
- **Customer Segmentation**: Advanced clustering analysis
- **Compliance Reporting**: Regulatory compliance and audit trails

### User Interface
- **Streamlit Web App**: Modern, responsive web interface
- **Interactive Dashboards**: Real-time data visualization
- **Multi-page Navigation**: Organized feature access
- **Export Capabilities**: Download reports and predictions

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd credit_default_prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Access the application**
Open your browser and navigate to `http://localhost:8501`

## 📊 System Architecture

```
credit_default_prediction/
├── app.py                          # Main Streamlit application
├── config.yaml                     # Configuration settings
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── src/                           # Source code
│   ├── data/                      # Data handling modules
│   │   ├── data_loader.py         # UCI dataset loading
│   │   ├── data_preprocessor.py   # Data cleaning and preprocessing
│   │   └── feature_engineering.py # Feature creation and selection
│   ├── models/                    # ML models and training
│   │   ├── model_trainer.py       # Model training and tuning
│   │   ├── model_evaluator.py     # Model evaluation and metrics
│   │   └── predictor.py           # Prediction engine
│   ├── visualization/             # Plotting and dashboards
│   │   ├── plots.py               # Plotting utilities
│   │   ├── dashboard.py           # Dashboard components
│   │   └── explainability.py     # SHAP/LIME integration
│   └── utils/                     # Utility functions
│       ├── logger.py              # Logging utilities
│       └── helpers.py             # Helper functions
├── pages/                         # Streamlit pages
│   ├── 1_Individual_Prediction.py # Single prediction interface
│   ├── 2_Batch_Prediction.py     # Batch processing
│   ├── 3_Model_Performance.py    # Model metrics dashboard
│   ├── 4_Data_Exploration.py     # Data analysis interface
│   ├── 5_Business_Intelligence.py # BI dashboards
│   ├── 6_Customer_Segmentation.py # Customer analytics
│   └── 7_Compliance_Report.py    # Regulatory compliance
├── components/                    # UI components
│   ├── sidebar.py                 # Sidebar components
│   ├── metrics.py                 # Metrics display
│   ├── charts.py                  # Chart components
│   └── forms.py                   # Form components
├── data/                          # Data storage
│   ├── raw/                       # Raw datasets
│   ├── processed/                 # Processed data
│   └── models/                    # Trained models
├── logs/                          # Application logs
├── tests/                         # Unit tests
└── notebooks/                     # Jupyter notebooks
```

## 🤖 Machine Learning Models

### Supported Algorithms
1. **XGBoost**: Gradient boosting with advanced regularization
2. **LightGBM**: Fast gradient boosting with leaf-wise tree growth
3. **CatBoost**: Gradient boosting optimized for categorical features
4. **Random Forest**: Ensemble of decision trees
5. **Logistic Regression**: Linear model for binary classification

### Model Features
- **Hyperparameter Optimization**: Automated tuning with Optuna
- **Cross-validation**: Robust model evaluation
- **Feature Selection**: Automated feature importance analysis
- **Model Persistence**: Save and load trained models
- **Ensemble Methods**: Combine multiple models for better performance

## 📈 Performance Metrics

The system tracks comprehensive metrics including:

### Classification Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, Precision-Recall AUC
- Matthews Correlation Coefficient
- Cohen's Kappa Score

### Business Metrics
- Default Rate by Risk Category
- Credit Exposure Analysis
- Expected vs Actual Loss
- Approval Rate Analysis
- Portfolio Risk Distribution

## 🔧 Configuration

### config.yaml Structure
```yaml
# Application Settings
app:
  name: "Credit Default Prediction System"
  version: "1.0.0"
  debug: false

# Data Settings
data:
  target_column: "default.payment.next.month"
  test_size: 0.2
  random_state: 42

# Model Settings
models:
  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1

# Business Rules
business_rules:
  high_risk_threshold: 0.7
  medium_risk_threshold: 0.3
  max_credit_limit: 1000000
```

## 📊 Data Requirements

### Input Features
The system expects the following features based on the UCI Credit Default dataset:

**Customer Demographics:**
- `LIMIT_BAL`: Credit limit amount
- `SEX`: Gender (1=male, 2=female)
- `EDUCATION`: Education level (1=graduate, 2=university, 3=high school, 4=others)
- `MARRIAGE`: Marital status (1=married, 2=single, 3=others)
- `AGE`: Age in years

**Payment History (6 months):**
- `PAY_0`, `PAY_2`, `PAY_3`, `PAY_4`, `PAY_5`, `PAY_6`: Payment status

**Bill Amounts (6 months):**
- `BILL_AMT1` through `BILL_AMT6`: Bill statement amounts

**Payment Amounts (6 months):**
- `PAY_AMT1` through `PAY_AMT6`: Payment amounts

## 🎯 Usage Examples

### Individual Prediction
```python
from src.models.predictor import CreditDefaultPredictor

# Initialize predictor
predictor = CreditDefaultPredictor()
predictor.load_models("data/models")

# Make prediction
input_data = {
    'LIMIT_BAL': 50000,
    'SEX': 1,
    'EDUCATION': 2,
    'MARRIAGE': 1,
    'AGE': 35,
    # ... other features
}

result = predictor.predict_single(input_data)
print(f"Risk Category: {result['risk_category']}")
print(f"Default Probability: {result['probability']:.2%}")
```

### Batch Prediction
```python
import pandas as pd

# Load batch data
batch_data = pd.read_csv("customer_applications.csv")

# Make batch predictions
results = predictor.predict_batch(batch_data)
print(f"Processed {results['summary']['total_predictions']} applications")
```

## 📱 Web Interface Pages

### 1. Individual Prediction
- Single customer risk assessment
- Interactive input forms
- Real-time prediction results
- Risk explanation and recommendations

### 2. Batch Prediction
- CSV file upload
- Bulk processing capabilities
- Results download
- Summary statistics

### 3. Model Performance
- Model comparison metrics
- ROC and PR curves
- Confusion matrices
- Feature importance plots

### 4. Data Exploration
- Dataset statistics and distributions
- Correlation analysis
- Missing value analysis
- Outlier detection

### 5. Business Intelligence
- Risk portfolio analysis
- KPI dashboards
- Trend analysis
- Performance monitoring

### 6. Customer Segmentation
- Clustering analysis
- Segment characteristics
- Risk profiling by segment
- Marketing insights

### 7. Compliance Report
- Regulatory compliance metrics
- Audit trail documentation
- Model governance reports
- Risk management summaries

## 🔍 Model Explainability

### SHAP Integration
- Global feature importance
- Local prediction explanations
- Waterfall plots
- Summary plots

### LIME Support
- Local interpretable explanations
- Feature contribution analysis
- Decision boundary visualization

## 🛡️ Security and Compliance

### Data Security
- Input validation and sanitization
- Secure model storage
- Audit logging
- Error handling

### Regulatory Compliance
- Model documentation
- Performance monitoring
- Bias detection and mitigation
- Explainable AI requirements

## 🧪 Testing

### Run Tests
```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_models.py
pytest tests/test_data.py

# Run with coverage
pytest --cov=src tests/
```

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Production Deployment
```bash
# Using Docker
docker build -t credit-default-prediction .
docker run -p 8501:8501 credit-default-prediction
```

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Credit Default dataset
- Streamlit team for the amazing web framework
- scikit-learn, XGBoost, LightGBM, and CatBoost communities
- SHAP and LIME libraries for model explainability

---

**Built with ❤️ for better credit risk management**
'''



## 📋 Project Overview
This is a complete, production-ready credit default prediction system built with modern ML techniques and a user-friendly web interface.

## 🎯 Key Achievements
✅ Complete project structure with 50+ files
✅ 5 ML models: XGBoost, LightGBM, CatBoost, Random Forest, Logistic Regression
✅ Comprehensive data preprocessing and feature engineering
✅ Streamlit web application with 7 pages
✅ Model explainability with SHAP integration
✅ Business intelligence dashboards
✅ Customer segmentation capabilities
✅ Compliance reporting features
✅ Production-ready code with error handling
✅ Comprehensive documentation

## 🏗️ Architecture Components

### Core ML Pipeline
- **Data Loader**: UCI dataset handling with validation
- **Preprocessor**: Data cleaning, scaling, balancing
- **Feature Engineer**: Domain-specific feature creation
- **Model Trainer**: Multi-model training with hyperparameter tuning
- **Model Evaluator**: Comprehensive metrics and visualization
- **Predictor**: Individual and batch prediction engine

### Web Application
- **Main App**: Streamlit application with navigation
- **7 Pages**: Individual prediction, batch processing, model performance, data exploration, BI, segmentation, compliance
- **UI Components**: Reusable sidebar, metrics, charts, forms
- **Visualization**: Interactive plots and dashboards

### Utilities
- **Logger**: Centralized logging with rotation
- **Helpers**: Data validation, file management, configuration
- **Configuration**: YAML-based settings management

## 🚀 Getting Started

1. **Extract the project**
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Run the app**: `streamlit run app.py`
4. **Access**: Open http://localhost:8501

## 📊 Features Implemented

### Machine Learning
- Multi-model ensemble predictions
- Hyperparameter optimization with Optuna
- Cross-validation and model selection
- Feature importance analysis
- Model persistence and loading

### Data Processing
- UCI Credit Default dataset integration
- Comprehensive data validation
- Missing value handling
- Outlier detection and treatment
- Feature engineering and selection
- Data balancing techniques

### Web Interface
- Modern Streamlit application
- Responsive design with custom CSS
- Interactive forms and visualizations
- Real-time predictions
- Batch processing capabilities
- Export functionality

### Business Intelligence
- Risk categorization (Low/Medium/High)
- Portfolio analysis
- KPI dashboards
- Customer segmentation
- Compliance reporting

### Model Explainability
- SHAP integration for explanations
- Feature importance visualization
- Prediction reasoning
- Business rule integration

## 🔧 Technical Stack
- **ML**: scikit-learn, XGBoost, LightGBM, CatBoost, Optuna
- **Web**: Streamlit, Plotly, Matplotlib, Seaborn
- **Data**: Pandas, NumPy, SciPy
- **Utils**: PyYAML, Requests, Joblib

## 📈 Production Readiness
- Comprehensive error handling
- Logging and monitoring
- Configuration management
- Input validation
- Model versioning
- Documentation
- Testing framework

## 🎉 Success Metrics
- **Code Quality**: Clean, documented, modular code
- **Functionality**: All required features implemented
- **Usability**: Intuitive web interface
- **Performance**: Efficient ML pipeline
- **Maintainability**: Well-structured architecture
- **Documentation**: Comprehensive guides and examples

This project represents a complete, enterprise-grade credit default prediction system ready for deployment and use in production environments.
'''