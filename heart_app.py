import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .positive {
        background-color: #FFE5E5;
        color: #D63031;
        border: 2px solid #FF6B6B;
    }
    .negative {
        background-color: #E5F5E5;
        color: #00B894;
        border: 2px solid #00D084;
    }
    .feature-importance {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = joblib.load('logistic_regression_fe_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file 'logistic_regression_fe_model.pkl' not found. Please ensure the model file is in the same directory as this script.")
        return None

def create_feature_engineering(X):
    """Apply the same feature engineering as in training"""
    X_fe = X.copy()
    
    # Interaction features
    X_fe['age_chol'] = X_fe['age'] * X_fe['chol']
    X_fe['thalach_age'] = X_fe['thalach'] / (X_fe['age'] + 1e-5)
    
    # Binning 'age' and 'chol'
    X_fe['age_bin'] = pd.cut(X_fe['age'], bins=[0, 40, 50, 60, 70, 100], labels=False)
    X_fe['chol_bin'] = pd.cut(X_fe['chol'], bins=[0, 200, 240, 280, 400], labels=False)
    
    # Polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(X_fe[['age', 'chol', 'thalach', 'oldpeak']])
    poly_feature_names = poly.get_feature_names_out(['age', 'chol', 'thalach', 'oldpeak'])
    
    # Combine everything
    X_fe = X_fe.drop(['age', 'chol', 'thalach', 'oldpeak'], axis=1)
    X_fe_poly = pd.DataFrame(poly_features, columns=poly_feature_names)
    X_fe_final = pd.concat([X_fe.reset_index(drop=True), X_fe_poly], axis=1)
    
    # Fill NaNs
    X_fe_final_clean = X_fe_final.fillna(0)
    
    return X_fe_final_clean

def create_visualizations(user_data, prediction, probability):
    """Create various visualizations"""
    
    # Risk factors visualization
    fig1 = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Age vs Max Heart Rate', 'Cholesterol Level', 'Blood Pressure', 'Chest Pain Type'),
        specs=[[{"type": "scatter"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Age vs Max Heart Rate
    color = 'red' if prediction == 1 else 'green'
    fig1.add_trace(
        go.Scatter(x=[user_data['age']], y=[user_data['thalach']], 
                  mode='markers', marker=dict(size=15, color=color),
                  name='Your Data'),
        row=1, col=1
    )
    
    # Cholesterol
    chol_categories = ['Normal (<200)', 'Borderline (200-239)', 'High (≥240)']
    chol_value = user_data['chol']
    if chol_value < 200:
        chol_cat = 0
    elif chol_value < 240:
        chol_cat = 1
    else:
        chol_cat = 2
    
    chol_colors = ['green', 'orange', 'red']
    fig1.add_trace(
        go.Bar(x=chol_categories, y=[1 if i == chol_cat else 0 for i in range(3)],
              marker_color=[chol_colors[i] if i == chol_cat else 'lightgray' for i in range(3)],
              name='Cholesterol Level'),
        row=1, col=2
    )
    
    # Blood Pressure
    bp_categories = ['Normal (<120)', 'Elevated (120-129)', 'High Stage 1 (130-139)', 'High Stage 2 (≥140)']
    bp_value = user_data['trestbps']
    if bp_value < 120:
        bp_cat = 0
    elif bp_value < 130:
        bp_cat = 1
    elif bp_value < 140:
        bp_cat = 2
    else:
        bp_cat = 3
    
    bp_colors = ['green', 'yellow', 'orange', 'red']
    fig1.add_trace(
        go.Bar(x=bp_categories, y=[1 if i == bp_cat else 0 for i in range(4)],
              marker_color=[bp_colors[i] if i == bp_cat else 'lightgray' for i in range(4)],
              name='Blood Pressure'),
        row=2, col=1
    )
    
    # Chest Pain Type
    cp_categories = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']
    cp_colors = ['red', 'orange', 'yellow', 'green']
    fig1.add_trace(
        go.Bar(x=cp_categories, y=[1 if i == user_data['cp'] else 0 for i in range(4)],
              marker_color=[cp_colors[i] if i == user_data['cp'] == i else 'lightgray' for i in range(4)],
              name='Chest Pain Type'),
        row=2, col=2
    )
    
    fig1.update_layout(height=600, showlegend=False, title_text="Risk Factor Analysis")
    
    # Probability gauge
    fig2 = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Heart Disease Risk (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig2.update_layout(height=400)
    
    return fig1, fig2

def main():
    st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction App</h1>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    st.markdown("### Please enter your medical information:")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", min_value=20, max_value=100, value=50)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox("Chest Pain Type", 
                         options=[0, 1, 2, 3], 
                         format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
        trestbps = st.slider("Resting Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
        chol = st.slider("Cholesterol (mg/dl)", min_value=100, max_value=400, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        restecg = st.selectbox("Resting ECG Results", 
                              options=[0, 1, 2], 
                              format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"][x])
    
    with col2:
        thalach = st.slider("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        oldpeak = st.slider("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST Segment", 
                            options=[0, 1, 2], 
                            format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
        ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", 
                           options=[0, 1, 2, 3], 
                           format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x])
    
    # Create prediction button
    if st.button("Predict Heart Disease Risk", type="primary"):
        # Create user data dictionary
        user_data = {
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
            'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
            'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }
        
        # Convert to DataFrame
        user_df = pd.DataFrame([user_data])
        
        # Apply feature engineering
        user_df_fe = create_feature_engineering(user_df)
        
        # Scale features (assuming the same scaler was used in training)
        scaler = StandardScaler()
        # Note: In a real application, you should save and load the scaler used during training
        user_df_scaled = scaler.fit_transform(user_df_fe)
        
        # Make prediction
        prediction = model.predict(user_df_scaled)[0]
        probability = model.predict_proba(user_df_scaled)[0][1]
        
        # Display results
        st.markdown("---")
        st.markdown("### Prediction Results")
        
        if prediction == 1:
            st.markdown(f'<div class="prediction-box positive">⚠️ HIGH RISK: Potential heart disease detected<br>Risk Probability: {probability:.2%}</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="prediction-box negative">✅ LOW RISK: No immediate heart disease risk detected<br>Risk Probability: {probability:.2%}</div>', 
                       unsafe_allow_html=True)
        
        # Create and display visualizations
        fig1, fig2 = create_visualizations(user_data, prediction, probability)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            st.plotly_chart(fig1, use_container_width=True)
        
        # Risk factors explanation
        st.markdown("### Risk Factor Analysis")
        
        risk_factors = []
        if age > 55:
            risk_factors.append(f"Age ({age}) - Higher risk as age increases")
        if chol > 240:
            risk_factors.append(f"High Cholesterol ({chol} mg/dl) - Above 240 is considered high")
        if trestbps > 140:
            risk_factors.append(f"High Blood Pressure ({trestbps} mmHg) - Above 140 is Stage 2 hypertension")
        if thalach < 100:
            risk_factors.append(f"Low Maximum Heart Rate ({thalach}) - May indicate poor cardiovascular fitness")
        if exang == 1:
            risk_factors.append("Exercise Induced Angina - Chest pain during exercise")
        if oldpeak > 2.0:
            risk_factors.append(f"High ST Depression ({oldpeak}) - Indicates possible heart problems")
        if ca > 0:
            risk_factors.append(f"Blocked Vessels ({ca}) - Major vessels with significant blockage")
        
        if risk_factors:
            st.markdown("**Identified Risk Factors:**")
            for factor in risk_factors:
                st.markdown(f"• {factor}")
        else:
            st.markdown("**No major risk factors identified in the provided data.**")
        
        # Recommendations
        st.markdown("### Recommendations")
        if prediction == 1:
            st.markdown("""
            **⚠️ Please consult with a healthcare professional immediately for:**
            - Comprehensive cardiac evaluation
            - ECG and stress testing
            - Blood work and imaging studies
            - Discussion of treatment options
            
            **Lifestyle modifications to consider:**
            - Heart-healthy diet (low sodium, saturated fat)
            - Regular exercise (as approved by doctor)
            - Stress management techniques
            - Smoking cessation if applicable
            - Weight management
            """)
        else:
            st.markdown("""
            **✅ Maintain heart health with:**
            - Regular cardiovascular exercise
            - Balanced, heart-healthy diet
            - Regular health check-ups
            - Stress management
            - Avoiding smoking and excessive alcohol
            
            **Continue monitoring risk factors and consult healthcare providers regularly.**
            """)
        
        # Disclaimer
        st.markdown("---")
        st.markdown("**Disclaimer:** This prediction is based on a machine learning model and should not replace professional medical advice. Always consult with qualified healthcare professionals for medical decisions.")

if __name__ == "__main__":
    main()