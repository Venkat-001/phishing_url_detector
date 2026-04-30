🔐 Phishing URL Detection System
A robust machine learning web application that identifies phishing URLs using advanced feature engineering and ensemble models, delivering real-time predictions with interpretable confidence scores.

📌 Overview

Phishing attacks remain one of the most common cybersecurity threats. This project presents a robust, end-to-end phishing detection system that combines feature engineering with ensemble machine learning models to classify URLs as Safe or Phishing.

🎯 Key Highlights
✅ Dual-model architecture (Random Forest + XGBoost)
⚡ Real-time prediction via Flask web app
📊 Confidence-based output for interpretability
🧪 Built-in sanity checks for validation
🔁 10-Fold cross-validation

🧠 System Architecture
User Input → Feature Extraction → ML Models → Prediction → Web UI

🛠️ Tech Stack
Backend: Python, Flask
ML: scikit-learn, XGBoost
Data: pandas, numpy
Feature Parsing: tldextract
Model Storage: joblib

⚙️ Setup
git clone https://github.com/Venkat-001/phishing_url_detection.git
cd phishing_URL_detection
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py

📊 Example
google.com → ✅ Safe
paypal-secure-login.com → ⚠️ Phishing

🚀 Future Improvements
Deploy to cloud (Render / AWS)
Add deep learning models
Build browser extension

👨‍💻 Author
VENKATA RAMANA PATHAKOTI
