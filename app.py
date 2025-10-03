import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .positive-prediction {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
    }
    .negative-prediction {
        background: linear-gradient(135deg, #51cf66, #40c057);
        color: white;
    }
    .info-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_scaler():
    """Load the trained model, scaler, and feature names"""
    try:
        # Try loading pre-trained model files first
        model = joblib.load('best_diabetes_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        
        # Load feature names
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
            
        return model, scaler, feature_names
    except FileNotFoundError:
        # If model files don't exist, train a simple model
        st.warning("⚠️ Pre-trained model not found. Training a new model...")
        return train_model_on_startup()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

@st.cache_resource
def train_model_on_startup():
    """Train a model when pre-trained files are not available"""
    try:
        # Load and preprocess data
        df = pd.read_csv('diabetes.csv')
        
        # Handle zero values
        zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in zero_columns:
            if col in df.columns:
                median_value = df[df[col] != 0][col].median()
                df[col] = df[col].replace(0, median_value)
        
        # Prepare features
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        X = df[feature_names]
        y = df['Outcome']
        
        # Train model
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        # Train model
        model = RandomForestClassifier(random_state=42, n_estimators=100)
        model.fit(X_train, y_train)
        
        st.success("✅ Model trained successfully!")
        return model, scaler, feature_names
        
    except Exception as e:
        st.error(f"❌ Error training model: {str(e)}")
        st.stop()

def get_feature_info():
    """Return information about each feature"""
    return {
        'Pregnancies': {
            'description': 'Number of times pregnant',
            'range': (0, 17),
            'normal_range': (0, 8),
            'unit': 'count'
        },
        'Glucose': {
            'description': 'Plasma glucose concentration (2 hours in oral glucose tolerance test)',
            'range': (0, 200),
            'normal_range': (70, 100),
            'unit': 'mg/dL'
        },
        'BloodPressure': {
            'description': 'Diastolic blood pressure',
            'range': (0, 140),
            'normal_range': (60, 80),
            'unit': 'mm Hg'
        },
        'SkinThickness': {
            'description': 'Triceps skin fold thickness',
            'range': (0, 100),
            'normal_range': (10, 30),
            'unit': 'mm'
        },
        'Insulin': {
            'description': '2-Hour serum insulin',
            'range': (0, 900),
            'normal_range': (16, 166),
            'unit': 'mu U/ml'
        },
        'BMI': {
            'description': 'Body Mass Index (weight in kg/(height in m)^2)',
            'range': (0, 70),
            'normal_range': (18.5, 24.9),
            'unit': 'kg/m²'
        },
        'DiabetesPedigreeFunction': {
            'description': 'Diabetes pedigree function (genetic predisposition)',
            'range': (0.0, 2.5),
            'normal_range': (0.0, 0.5),
            'unit': 'score'
        },
        'Age': {
            'description': 'Age in years',
            'range': (21, 81),
            'normal_range': (21, 60),
            'unit': 'years'
        }
    }

def create_input_form():
    """Create the input form for user data"""
    feature_info = get_feature_info()
    
    st.markdown('<div class="sub-header">📝 Enter Patient Information</div>', unsafe_allow_html=True)
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    inputs = {}
    
    with col1:
        st.markdown("**Basic Information**")
        inputs['Pregnancies'] = st.slider(
            "Number of Pregnancies",
            min_value=0, max_value=17, value=1,
            help=feature_info['Pregnancies']['description'],
            key="pregnancies_slider"
        )
        
        inputs['Age'] = st.slider(
            "Age (years)",
            min_value=21, max_value=81, value=25,
            help=feature_info['Age']['description'],
            key="age_slider"
        )
        
        inputs['BMI'] = st.slider(
            "BMI (kg/m²)",
            min_value=0.0, max_value=70.0, value=25.0, step=0.1,
            help=feature_info['BMI']['description'],
            key="bmi_slider"
        )
        
        inputs['DiabetesPedigreeFunction'] = st.slider(
            "Diabetes Pedigree Function",
            min_value=0.0, max_value=2.5, value=0.5, step=0.01,
            help=feature_info['DiabetesPedigreeFunction']['description'],
            key="pedigree_slider"
        )
    
    with col2:
        st.markdown("**Medical Measurements**")
        inputs['Glucose'] = st.slider(
            "Glucose Level (mg/dL)",
            min_value=0, max_value=200, value=120,
            help=feature_info['Glucose']['description'],
            key="glucose_slider"
        )
        
        inputs['BloodPressure'] = st.slider(
            "Blood Pressure (mm Hg)",
            min_value=0, max_value=140, value=80,
            help=feature_info['BloodPressure']['description'],
            key="blood_pressure_slider"
        )
        
        inputs['SkinThickness'] = st.slider(
            "Skin Thickness (mm)",
            min_value=0, max_value=100, value=20,
            help=feature_info['SkinThickness']['description'],
            key="skin_thickness_slider"
        )
        
        inputs['Insulin'] = st.slider(
            "Insulin Level (mu U/ml)",
            min_value=0, max_value=900, value=80,
            help=feature_info['Insulin']['description'],
            key="insulin_slider"
        )
    
    return inputs

def interpret_prediction(probability):
    """Interpret the prediction probability"""
    if probability < 0.3:
        return "Low Risk", "The patient has a low risk of diabetes.", "🟢"
    elif probability < 0.7:
        return "Moderate Risk", "The patient has a moderate risk of diabetes. Monitor closely.", "🟡"
    else:
        return "High Risk", "The patient has a high risk of diabetes. Consider further testing.", "🔴"

def create_risk_gauge(probability):
    """Create a risk gauge visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Diabetes Risk Probability (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig

def create_feature_comparison(inputs):
    """Create a comparison chart of input values vs normal ranges"""
    feature_info = get_feature_info()
    
    features = []
    user_values = []
    normal_mins = []
    normal_maxs = []
    
    for feature, value in inputs.items():
        features.append(feature)
        user_values.append(value)
        normal_mins.append(feature_info[feature]['normal_range'][0])
        normal_maxs.append(feature_info[feature]['normal_range'][1])
    
    fig = go.Figure()
    
    # Add normal range bars
    fig.add_trace(go.Bar(
        y=features,
        x=normal_maxs,
        base=normal_mins,
        orientation='h',
        name='Normal Range',
        marker_color='lightgreen',
        opacity=0.6
    ))
    
    # Add user values
    fig.add_trace(go.Scatter(
        y=features,
        x=user_values,
        mode='markers',
        name='Your Values',
        marker=dict(
            color='red',
            size=12,
            symbol='diamond'
        )
    ))
    
    fig.update_layout(
        title="Your Values vs Normal Ranges",
        xaxis_title="Value",
        yaxis_title="Features",
        height=600,
        showlegend=True,
        barmode='overlay'
    )
    
    return fig

def display_feature_insights(inputs):
    """Display insights about the input features"""
    feature_info = get_feature_info()
    
    st.markdown('<div class="sub-header">🔍 Feature Analysis</div>', unsafe_allow_html=True)
    
    insights = []
    
    for feature, value in inputs.items():
        info = feature_info[feature]
        normal_min, normal_max = info['normal_range']
        
        if value < normal_min:
            status = "Below Normal"
            color = "🔵"
        elif value > normal_max:
            status = "Above Normal"
            color = "🔴"
        else:
            status = "Normal"
            color = "🟢"
        
        insights.append({
            'Feature': feature,
            'Value': f"{value} {info['unit']}",
            'Status': f"{color} {status}",
            'Normal Range': f"{normal_min}-{normal_max} {info['unit']}"
        })
    
    insights_df = pd.DataFrame(insights)
    st.table(insights_df)

def main():
    # Header
    st.markdown('<div class="main-header">🩺 Diabetes Prediction System</div>', unsafe_allow_html=True)
    
    # Load model
    model, scaler, feature_names = load_model_and_scaler()
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("### 📊 About This App")
        st.markdown("""
        This application uses machine learning to predict diabetes risk based on medical measurements.
        
        **Features:**
        - Interactive input form
        - Real-time predictions
        - Risk visualization
        - Feature analysis
        
        **Model Information:**
        - Trained on Pima Indians Diabetes Dataset
        - Uses advanced ML algorithms
        - Optimized with hyperparameter tuning
        """)
        
        st.markdown("### ⚠️ Disclaimer")
        st.markdown("""
        This tool is for educational purposes only and should not replace professional medical advice.
        Always consult healthcare professionals for medical decisions.
        """)
    
    # Initialize session state for inputs if not exists
    if 'current_inputs' not in st.session_state:
        st.session_state.current_inputs = {
            'Pregnancies': 1,
            'Age': 25,
            'BMI': 25.0,
            'DiabetesPedigreeFunction': 0.5,
            'Glucose': 120,
            'BloodPressure': 80,
            'SkinThickness': 20,
            'Insulin': 80
        }
    
    # Input form (only create once)
    inputs = create_input_form()
    st.session_state.current_inputs = inputs
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Analysis", "ℹ️ Information"])
    
    with tab1:
        st.markdown("### 🎯 Make Your Prediction")
        st.markdown("Adjust the values above and click the button below to get your diabetes risk prediction.")
        
        # Prediction button
        if st.button("🔮 Predict Diabetes Risk", type="primary", use_container_width=True):
            try:
                # Prepare input data
                input_df = pd.DataFrame([inputs])
                input_df = input_df[feature_names]  # Ensure correct order
                
                # Scale the input
                input_scaled = scaler.transform(input_df)
                
                # Make prediction
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0][1]
                
                # Display results
                st.markdown("---")
                st.markdown('<div class="sub-header">🎯 Prediction Results</div>', unsafe_allow_html=True)
                
                # Create columns for results
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Prediction box
                    if prediction == 1:
                        st.markdown(f'''
                        <div class="prediction-box positive-prediction">
                            🚨 DIABETES RISK DETECTED<br>
                            Probability: {probability:.1%}
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''
                        <div class="prediction-box negative-prediction">
                            ✅ LOW DIABETES RISK<br>
                            Probability: {probability:.1%}
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    # Risk interpretation
                    risk_level, risk_message, risk_icon = interpret_prediction(probability)
                    st.markdown(f'''
                    <div class="info-box">
                        <strong>{risk_icon} Risk Level:</strong> {risk_level}<br>
                        <strong>Recommendation:</strong> {risk_message}
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col2:
                    # Risk gauge
                    gauge_fig = create_risk_gauge(probability)
                    st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Feature insights
                display_feature_insights(inputs)
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
    
    with tab2:
        st.markdown('<div class="sub-header">📊 Detailed Analysis</div>', unsafe_allow_html=True)
        st.markdown("This analysis uses the values you set in the input form above.")
        
        if st.button("📈 Generate Analysis", use_container_width=True):
            # Use session state inputs
            current_inputs = st.session_state.current_inputs
            
            # Feature comparison chart
            comparison_fig = create_feature_comparison(current_inputs)
            st.plotly_chart(comparison_fig, use_container_width=True)
            
            # Statistical summary
            st.markdown("### 📋 Input Summary")
            input_df = pd.DataFrame([current_inputs]).T
            input_df.columns = ['Your Value']
            input_df['Feature'] = input_df.index
            input_df = input_df[['Feature', 'Your Value']]
            st.table(input_df)
    
    with tab3:
        st.markdown('<div class="sub-header">ℹ️ Dataset Information</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 📊 About the Dataset
        
        This application is trained on the **Pima Indians Diabetes Dataset**, which contains medical information from Pima Indian women.
        
        **Dataset Details:**
        - **Source**: National Institute of Diabetes and Digestive and Kidney Diseases
        - **Population**: Pima Indian women (age 21+)
        - **Sample Size**: 768 instances
        - **Features**: 8 medical predictor variables
        - **Target**: Binary classification (diabetes/no diabetes)
        
        ### 🔬 Feature Descriptions
        """)
        
        feature_info = get_feature_info()
        
        for feature, info in feature_info.items():
            with st.expander(f"📋 {feature}"):
                st.write(f"**Description:** {info['description']}")
                st.write(f"**Unit:** {info['unit']}")
                st.write(f"**Normal Range:** {info['normal_range'][0]} - {info['normal_range'][1]} {info['unit']}")
                st.write(f"**Data Range:** {info['range'][0]} - {info['range'][1]} {info['unit']}")
        
        st.markdown("""
        ### 🧠 Model Information
        
        - **Algorithm**: Advanced machine learning ensemble
        - **Training**: 80% of dataset (614 samples)
        - **Testing**: 20% of dataset (154 samples)
        - **Optimization**: GridSearchCV hyperparameter tuning
        - **Validation**: 10-fold cross-validation
        - **Performance**: Optimized for F1-score and accuracy
        
        ### 🎯 Performance Metrics
        
        The model has been evaluated using multiple metrics:
        - **Accuracy**: Overall prediction correctness
        - **Precision**: Proportion of positive predictions that are correct
        - **Recall**: Proportion of actual positives correctly identified
        - **F1-Score**: Harmonic mean of precision and recall
        - **ROC-AUC**: Area under the receiver operating characteristic curve
        """)

if __name__ == "__main__":
    main()