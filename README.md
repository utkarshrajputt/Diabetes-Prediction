# Diabetes Prediction Machine Learning Project

## 🎯 Project Overview

This comprehensive machine learning project predicts diabetes risk using the Pima Indians Diabetes Dataset. The project demonstrates a complete ML workflow from data exploration to model deployment in an interactive web application.

## 📊 Dataset Information

- **Source**: Pima Indians Diabetes Dataset
- **Samples**: 768 instances
- **Features**: 8 medical predictor variables
- **Target**: Binary classification (diabetes/no diabetes)
- **Class Distribution**: ~65% no diabetes, ~35% diabetes

### Features:
1. **Pregnancies** - Number of times pregnant
2. **Glucose** - Plasma glucose concentration (2 hours in oral glucose tolerance test)
3. **BloodPressure** - Diastolic blood pressure (mm Hg)
4. **SkinThickness** - Triceps skin fold thickness (mm)
5. **Insulin** - 2-Hour serum insulin (mu U/ml)
6. **BMI** - Body mass index (weight in kg/(height in m)^2)
7. **DiabetesPedigreeFunction** - Diabetes pedigree function (genetic predisposition)
8. **Age** - Age in years

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **Clone or download the project files**
2. **Create and activate virtual environment:**
   ```bash
   python -m venv diabetes_ml_env
   # Windows
   .\diabetes_ml_env\Scripts\Activate.ps1
   # macOS/Linux
   source diabetes_ml_env/bin/activate
   ```

3. **Install required packages:**
   ```bash
   pip install pandas scikit-learn matplotlib seaborn streamlit joblib jupyter numpy plotly
   ```

### Quick Test

Run the test script to verify everything works:
```bash
python test_workflow.py
```

## 📁 Project Structure

```
diabetes-prediction/
│
├── diabetes.csv                    # Dataset
├── diabetes_ml_analysis.ipynb      # Main Jupyter notebook
├── app.py                          # Streamlit web application
├── test_workflow.py               # Test script
├── README.md                      # This file
│
├── Generated Files (after running notebook/test):
├── best_diabetes_model.pkl        # Trained model (joblib)
├── feature_scaler.pkl             # Feature scaler (joblib)
├── feature_names.pkl              # Feature names list
├── best_diabetes_model_pickle.pkl # Trained model (pickle backup)
└── feature_scaler_pickle.pkl      # Feature scaler (pickle backup)
```

## 🧠 Machine Learning Workflow

### 1. Data Analysis & Preprocessing
- **Exploratory Data Analysis (EDA)**: Comprehensive data exploration with visualizations
- **Data Cleaning**: Handle missing values (zeros) by replacing with median values
- **Feature Engineering**: Create additional categorical features
- **Data Scaling**: StandardScaler for algorithm optimization

### 2. Model Implementation
Implementation of 6 different algorithms:
- **Logistic Regression** (with scaling)
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Support Vector Machine** (with scaling)
- **Naive Bayes** (Gaussian)
- **K-Nearest Neighbors** (with scaling)

### 3. Model Evaluation
- **Train/Test Split**: 80/20 stratified split
- **Metrics Used**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualization**: Confusion matrices, ROC curves, performance comparisons

### 4. Hyperparameter Optimization
- **GridSearchCV**: Comprehensive parameter tuning for best model
- **Cross-Validation**: 10-fold stratified cross-validation
- **Optimization Target**: F1-Score (balanced for imbalanced dataset)

### 5. Model Persistence
- **Saving**: Best model and scaler saved using joblib and pickle
- **Loading**: Automatic model loading in Streamlit app

## 🌐 Streamlit Web Application

### Features:
- **Interactive Input Form**: User-friendly sliders for all features
- **Real-time Predictions**: Instant diabetes risk assessment
- **Risk Visualization**: Gauge chart showing probability levels
- **Feature Analysis**: Comparison with normal ranges
- **Educational Content**: Detailed information about features and model

### Running the App:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### App Sections:
1. **🎯 Prediction**: Main prediction interface
2. **📊 Analysis**: Detailed feature analysis and comparisons
3. **ℹ️ Information**: Dataset and model information

## 📈 Model Performance

The best model achieves (example metrics from test run):
- **Accuracy**: ~77.9%
- **F1-Score**: ~65.3%
- **ROC-AUC**: ~80%+

*Note: Actual performance metrics will be displayed after running the full analysis notebook*

## 🔬 Key Insights

### Most Important Features:
1. **Glucose Level** - Primary indicator
2. **BMI** - Strong correlation with diabetes
3. **Age** - Risk increases with age
4. **Diabetes Pedigree Function** - Genetic predisposition
5. **Pregnancies** - Relevant for female population

### Data Quality Issues Addressed:
- **Zero values** in impossible fields (glucose, blood pressure, BMI, etc.)
- **Missing data patterns** handled through median imputation
- **Feature scaling** for distance-based algorithms

## 🎓 Educational Value

This project demonstrates:
- **Complete ML Pipeline**: From data exploration to deployment
- **Multiple Algorithms**: Comparison of different ML approaches
- **Hyperparameter Tuning**: Optimization techniques
- **Cross-Validation**: Robust model evaluation
- **Web Deployment**: Interactive application development
- **Best Practices**: Code organization, documentation, testing

## ⚠️ Important Disclaimers

- **Educational Purpose**: This project is for learning machine learning concepts
- **Not Medical Advice**: Results should not replace professional medical consultation
- **Dataset Limitations**: Trained on specific population (Pima Indians)
- **Model Limitations**: Accuracy may vary on different populations

## 🛠️ Troubleshooting

### Common Issues:

1. **Module not found errors**:
   ```bash
   pip install [missing-package-name]
   ```

2. **Model files not found**:
   - Run `python test_workflow.py` first
   - Or execute the Jupyter notebook

3. **Streamlit app won't start**:
   - Check if all packages are installed
   - Ensure you're in the correct directory
   - Verify virtual environment is activated

4. **Virtual environment issues**:
   - Recreate the environment
   - Use `python -m venv` instead of `virtualenv`

## 📚 Learning Outcomes

After completing this project, you will understand:
- **Data preprocessing** techniques for medical datasets
- **Multiple ML algorithms** and their applications
- **Model evaluation** and comparison methods
- **Hyperparameter tuning** with GridSearchCV
- **Cross-validation** for robust assessment
- **Web application development** with Streamlit
- **Model deployment** and persistence
- **Interactive data visualization** with Plotly

## 🚀 Future Enhancements

Potential improvements:
- **Additional algorithms**: XGBoost, LightGBM, Neural Networks
- **Feature selection**: Advanced feature engineering
- **Model stacking**: Ensemble methods
- **Data augmentation**: SMOTE for imbalanced classes
- **Model interpretability**: SHAP values, LIME
- **API deployment**: FastAPI or Flask REST API
- **Database integration**: Store predictions and user data
- **A/B testing**: Compare different model versions

## 📝 Assignment Submission Files

For your formative assessment, submit these files:

1. **diabetes_ml_analysis.ipynb** - Complete analysis notebook
2. **app.py** - Streamlit application
3. **PDF Report** - Summary of findings, model comparisons, and insights

## 🤝 Support

If you encounter any issues:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure dataset file is in the correct location
4. Run the test script to verify setup

---

**Happy Learning! 🎉**

*This project provides a comprehensive introduction to machine learning in healthcare applications while following industry best practices for model development and deployment.*