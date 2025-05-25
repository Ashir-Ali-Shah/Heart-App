Heart Disease Prediction App ❤️
A comprehensive web application for predicting heart disease risk using machine learning, built with Streamlit and featuring interactive visualizations.
🎯 Overview
This application uses a trained Logistic Regression model with advanced feature engineering to predict the likelihood of heart disease based on 13 medical parameters. The model incorporates polynomial features, interaction terms, and binning techniques to achieve optimal prediction accuracy.
✨ Features

Interactive Web Interface: User-friendly form for inputting medical parameters
Real-time Predictions: Instant heart disease risk assessment with probability scores
Advanced Visualizations:

Risk factor analysis charts
Probability gauge displaying risk percentage
Interactive plots using Plotly


Comprehensive Risk Assessment: Identifies specific risk factors and provides detailed analysis
Medical Recommendations: Tailored advice based on prediction results
Professional UI: Modern, responsive design with medical-grade styling

🏗️ Model Architecture
The prediction model includes sophisticated feature engineering:

Interaction Features: Age-cholesterol interaction, heart rate-age ratio
Binning: Age and cholesterol level categorization
Polynomial Features: Second-degree polynomial features for continuous variables
Standardization: Feature scaling for optimal model performance

🔧 VS Code Setup
Required Extensions

Python (by Microsoft)
Python Debugger (by Microsoft)
Jupyter (optional)

Running in VS Code

Open the project folder in VS Code
Select Python interpreter (Ctrl+Shift+P → "Python: Select Interpreter")
Open integrated terminal (`Ctrl+``)
Run: streamlit run heart_disease_app.py

Debug Configuration
Use the provided .vscode/launch.json to run with F5 or use the debug panel.
📊 Model Performance
The model was trained on the Heart Disease UCI dataset with the following performance metrics:

Feature Engineering: Advanced preprocessing with polynomial features and interaction terms
Cross-validation: Robust model validation techniques
Scaling: StandardScaler for feature normalization

🎨 Screenshots
Add screenshots of your application here showing:

Main input interface
Prediction results
Visualization dashboards
Risk assessment panel

⚠️ Important Notes
Medical Disclaimer
This application is for educational and demonstration purposes only. The predictions should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.
Model Limitations

Based on historical data patterns
May not account for all medical factors
Requires professional medical validation
Not approved for clinical use

