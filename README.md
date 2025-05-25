# Heart Disease Prediction App ❤️

A comprehensive web application for predicting heart disease risk using machine learning, built with Streamlit and featuring interactive visualizations.

## 🎯 Overview

This application uses a trained Logistic Regression model with advanced feature engineering to predict the likelihood of heart disease based on 13 medical parameters. The model incorporates polynomial features, interaction terms, and binning techniques to achieve optimal prediction accuracy.

## ✨ Features

- **Interactive Web Interface**: User-friendly form for inputting medical parameters
- **Real-time Predictions**: Instant heart disease risk assessment with probability scores
- **Advanced Visualizations**: 
  - Risk factor analysis charts
  - Probability gauge displaying risk percentage
  - Interactive plots using Plotly
- **Comprehensive Risk Assessment**: Identifies specific risk factors and provides detailed analysis
- **Medical Recommendations**: Tailored advice based on prediction results
- **Professional UI**: Modern, responsive design with medical-grade styling

## 🏗️ Model Architecture

The prediction model includes sophisticated feature engineering:

- **Interaction Features**: Age-cholesterol interaction, heart rate-age ratio
- **Binning**: Age and cholesterol level categorization
- **Polynomial Features**: Second-degree polynomial features for continuous variables
- **Standardization**: Feature scaling for optimal model performance

## 📋 Input Parameters

The application collects 13 medical parameters:

| Parameter | Description | Range/Options |
|-----------|-------------|---------------|
| **Age** | Patient age in years | 20-100 |
| **Sex** | Gender | Male/Female |
| **CP** | Chest pain type | Typical Angina, Atypical Angina, Non-anginal Pain, Asymptomatic |
| **Trestbps** | Resting blood pressure (mmHg) | 80-200 |
| **Chol** | Serum cholesterol (mg/dl) | 100-400 |
| **FBS** | Fasting blood sugar > 120 mg/dl | Yes/No |
| **Restecg** | Resting ECG results | Normal, ST-T Wave Abnormality, Left Ventricular Hypertrophy |
| **Thalach** | Maximum heart rate achieved | 60-220 |
| **Exang** | Exercise induced angina | Yes/No |
| **Oldpeak** | ST depression induced by exercise | 0.0-6.0 |
| **Slope** | Slope of peak exercise ST segment | Upsloping, Flat, Downsloping |
| **CA** | Number of major vessels colored by fluoroscopy | 0-3 |
| **Thal** | Thalassemia | Normal, Fixed Defect, Reversible Defect, Unknown |

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- VS Code (recommended) or any Python IDE
- Git (optional)

### Installation

1. **Clone or download the repository**
   ```bash
   git clone <repository-url>
   cd heart-disease-prediction
   ```

2. **Create a virtual environment**
   ```bash
   # Using conda (recommended)
   conda create -n heart_disease python=3.9
   conda activate heart_disease
   
   # OR using venv
   python -m venv heart_disease_env
   # Windows:
   heart_disease_env\Scripts\activate
   # Mac/Linux:
   source heart_disease_env/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model file is present**
   - Place `logistic_regression_fe_model.pkl` in the project root directory

5. **Run the application**
   ```bash
   streamlit run heart_disease_app.py
   ```

6. **Access the app**
   - Open your browser and navigate to `http://localhost:8501`
   - The app will automatically open in your default browser

## 📁 Project Structure

```
heart-disease-prediction/
├── heart_disease_app.py          # Main Streamlit application
├── logistic_regression_fe_model.pkl  # Trained ML model
├── requirements.txt              # Python dependencies
├── README.md                    # This file
├── .vscode/                     # VS Code configuration (optional)
│   └── launch.json              # Debug configuration
└── screenshots/                 # App screenshots (optional)
```

## 🔧 VS Code Setup

### Required Extensions
- **Python** (by Microsoft)
- **Python Debugger** (by Microsoft)
- **Jupyter** (optional)

### Running in VS Code
1. Open the project folder in VS Code
2. Select Python interpreter (`Ctrl+Shift+P` → "Python: Select Interpreter")
3. Open integrated terminal (`Ctrl+``)
4. Run: `streamlit run heart_disease_app.py`

### Debug Configuration
Use the provided `.vscode/launch.json` to run with F5 or use the debug panel.

## 📊 Model Performance

The model was trained on the Heart Disease UCI dataset with the following performance metrics:

- **Feature Engineering**: Advanced preprocessing with polynomial features and interaction terms
- **Cross-validation**: Robust model validation techniques
- **Scaling**: StandardScaler for feature normalization

## 🎨 Screenshots

*Add screenshots of your application here showing:*
- Main input interface
- Prediction results
- Visualization dashboards
- Risk assessment panel

## ⚠️ Important Notes

### Medical Disclaimer
**This application is for educational and demonstration purposes only. The predictions should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.**

### Model Limitations
- Based on historical data patterns
- May not account for all medical factors
- Requires professional medical validation
- Not approved for clinical use

## 🛠️ Troubleshooting

### Common Issues

**1. ModuleNotFoundError**
```bash
pip install -r requirements.txt
# Ensure virtual environment is activated
```

**2. Model file not found**
- Verify `logistic_regression_fe_model.pkl` is in the project root
- Check file permissions

**3. Port already in use**
```bash
streamlit run heart_disease_app.py --server.port 8502
```

**4. Browser doesn't open automatically**
- Manually navigate to `http://localhost:8501`

### Performance Issues
- Large datasets may require additional memory
- Consider upgrading Python packages for better performance

## 🔄 Development

### Adding New Features
1. Fork the repository
2. Create a feature branch
3. Implement changes
4. Test thoroughly
5. Submit a pull request

### Model Updates
To update the machine learning model:
1. Retrain with new data
2. Save as `logistic_regression_fe_model.pkl`
3. Update feature engineering if needed
4. Test predictions accuracy

## 📚 Dependencies

```
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.3.2
plotly==5.15.0
```


