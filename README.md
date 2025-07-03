# Credit Default Prediction System 💳

A comprehensive machine learning application for credit risk assessment and default prediction, built with Streamlit and advanced ML algorithms.

## 🎯 Overview

This application provides a complete solution for credit default prediction, featuring:
- **Individual & Batch Predictions**: Real-time risk assessment for single customers or bulk processing
- **Advanced ML Models**: XGBoost, LightGBM, CatBoost, Random Forest, and Logistic Regression
- **Model Explainability**: SHAP and LIME integration for transparent decision-making
- **Business Intelligence**: Comprehensive dashboards and KPI tracking
- **Customer Segmentation**: Advanced analytics for customer profiling
- **Compliance Features**: Regulatory reporting and fairness metrics

## 🏗️ Architecture

```
credit_default_prediction/
├── app.py                          # Main Streamlit application
├── config.yaml                     # Configuration settings
├── requirements.txt                 # Python dependencies
├── README.md                       # This file
│
├── data/                           # Data storage
│   ├── raw/                        # Raw datasets
│   ├── processed/                  # Processed datasets
│   └── models/                     # Trained models
│
├── src/                            # Core source code
│   ├── data/                       # Data processing modules
│   │   ├── data_loader.py          # Data loading utilities
│   │   ├── data_preprocessor.py    # Data preprocessing
│   │   └── feature_engineering.py  # Feature engineering
│   │
│   ├── models/                     # ML model modules
│   │   ├── model_trainer.py        # Model training
│   │   ├── model_evaluator.py      # Model evaluation
│   │   └── predictor.py            # Prediction engine
│   │
│   ├── utils/                      # Utility modules
│   │   ├── config.py               # Configuration management
│   │   ├── logger.py               # Logging utilities
│   │   └── helpers.py              # Helper functions
│   │
│   └── visualization/              # Visualization modules
│       ├── plots.py                # Plotting utilities
│       ├── dashboard.py            # Dashboard components
│       └── explainability.py       # Model explainability
│
├── pages/                          # Streamlit pages
│   ├── 1_Individual_Prediction.py  # Single prediction interface
│   ├── 2_Batch_Prediction.py       # Batch processing
│   ├── 3_Model_Performance.py      # Model metrics and evaluation
│   ├── 4_Data_Exploration.py       # Data analysis and visualization
│   ├── 5_Business_Intelligence.py  # BI dashboards
│   ├── 6_Customer_Segmentation.py  # Customer analytics
│   └── 7_Compliance_Report.py      # Regulatory compliance
│
├── components/                     # Reusable UI components
│   ├── sidebar.py                  # Sidebar components
│   ├── metrics.py                  # Metrics display
│   ├── charts.py                   # Chart components
│   └── forms.py                    # Form components
│
├── tests/                          # Unit tests
├── notebooks/                      # Jupyter notebooks
└── assets/                         # Static assets
    ├── images/                     # Images and logos
    └── css/                        # Custom styling
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

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

## 📊 Dataset

The application uses the **UCI Credit Default Dataset** which contains:
- **30,000 credit card clients** from Taiwan
- **24 features** including demographic and financial information
- **Binary target variable** indicating default in the next month

### Features Include:
- **Demographic**: Gender, Education, Marriage, Age
- **Financial**: Credit limit, Payment history, Bill amounts, Payment amounts
- **Target**: Default payment next month (0: No, 1: Yes)

## 🤖 Machine Learning Models

### Available Models:
1. **XGBoost** (Default) - Gradient boosting with excellent performance
2. **LightGBM** - Fast gradient boosting with lower memory usage
3. **CatBoost** - Handles categorical features automatically
4. **Random Forest** - Ensemble method with good interpretability
5. **Logistic Regression** - Linear model for baseline comparison

### Model Features:
- **Automated hyperparameter tuning**
- **Cross-validation** for robust evaluation
- **Feature importance analysis**
- **Threshold optimization** for business metrics
- **Model comparison** and selection

## 📈 Key Features

### 1. Individual Prediction
- Real-time credit risk assessment
- Interactive input forms
- Instant risk scoring
- SHAP explanations for decisions

### 2. Batch Prediction
- Upload CSV files for bulk processing
- Parallel processing for large datasets
- Downloadable results with risk scores
- Summary statistics and visualizations

### 3. Model Performance
- Comprehensive evaluation metrics
- ROC and Precision-Recall curves
- Confusion matrix analysis
- Feature importance rankings
- Model comparison dashboard

### 4. Data Exploration
- Interactive data visualization
- Statistical summaries
- Correlation analysis
- Distribution plots
- Missing value analysis

### 5. Business Intelligence
- KPI dashboards
- Risk distribution analysis
- Portfolio performance metrics
- Trend analysis
- Profitability calculations

### 6. Customer Segmentation
- Advanced clustering algorithms
- Customer profiling
- Segment characteristics
- Risk-based segmentation
- Actionable insights

### 7. Compliance Reporting
- Regulatory compliance metrics
- Fairness and bias analysis
- Model documentation
- Audit trail
- Risk governance reports

## 🔧 Configuration

The application is highly configurable through `config.yaml`:

### Key Configuration Sections:
- **App Settings**: Basic application configuration
- **Data Configuration**: Dataset and preprocessing settings
- **Model Parameters**: ML model hyperparameters
- **Visualization**: Chart and dashboard settings
- **Business Rules**: Risk thresholds and business logic
- **Compliance**: Regulatory and fairness settings

### Example Configuration:
```yaml
models:
  default_model: "XGBoost"
  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1

business_intelligence:
  risk_categories:
    low: [0, 0.3]
    medium: [0.3, 0.7]
    high: [0.7, 1.0]
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v --cov=src
```

### Test Coverage:
- Data processing functions
- Model training and evaluation
- Prediction accuracy
- Utility functions
- Configuration management

## 📝 Usage Examples

### Individual Prediction
```python
from src.models.predictor import CreditPredictor

predictor = CreditPredictor()
risk_score = predictor.predict_single({
    'LIMIT_BAL': 20000,
    'SEX': 2,
    'EDUCATION': 2,
    'MARRIAGE': 1,
    'AGE': 24,
    # ... other features
})
```

### Batch Processing
```python
import pandas as pd
from src.models.predictor import CreditPredictor

predictor = CreditPredictor()
df = pd.read_csv('customer_data.csv')
predictions = predictor.predict_batch(df)
```

## 🔍 Model Explainability

### SHAP Integration
- **Global explanations**: Feature importance across all predictions
- **Local explanations**: Individual prediction breakdowns
- **Waterfall plots**: Step-by-step decision process
- **Summary plots**: Feature impact visualization

### LIME Integration
- **Local interpretable explanations**
- **Feature contribution analysis**
- **Alternative explanation method**
- **Complementary to SHAP analysis**

## 📊 Business Metrics

### Key Performance Indicators:
- **Approval Rate**: Percentage of approved applications
- **Default Rate**: Percentage of defaults in approved applications
- **Profit Margin**: Expected profit from approved applications
- **Risk-Adjusted Return**: Return adjusted for risk level

### Risk Categories:
- **Low Risk**: 0-30% default probability
- **Medium Risk**: 30-70% default probability
- **High Risk**: 70-100% default probability

## 🛡️ Compliance & Fairness

### Fairness Metrics:
- **Demographic Parity**: Equal approval rates across groups
- **Equalized Odds**: Equal true positive rates across groups
- **Calibration**: Consistent probability estimates across groups

### Regulatory Features:
- **Model documentation** and versioning
- **Audit trail** for all predictions
- **Bias detection** and mitigation
- **Explainability** for regulatory compliance

## 🚀 Deployment

### Local Deployment
```bash
streamlit run app.py --server.port 8501
```

### Docker Deployment
```bash
docker build -t credit-default-app .
docker run -p 8501:8501 credit-default-app
```

### Cloud Deployment
- **Streamlit Cloud**: Direct deployment from GitHub
- **Heroku**: Web application hosting
- **AWS/GCP/Azure**: Cloud platform deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines:
- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation
- Ensure all tests pass

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for the credit default dataset
- **Streamlit** for the amazing web framework
- **Scikit-learn** and **XGBoost** for machine learning capabilities
- **SHAP** and **LIME** for model explainability

## 📞 Support

For support and questions:
- Create an issue in the GitHub repository
- Contact the development team
- Check the documentation and FAQ

## 🔄 Version History

- **v1.0.0** - Initial release with core functionality
- **v1.1.0** - Added batch processing and explainability
- **v1.2.0** - Enhanced business intelligence features
- **v1.3.0** - Added compliance and fairness metrics

---

**Built with ❤️ for better credit risk management**
