# app.py
from flask import Flask, request, render_template, jsonify
import joblib
from feature_extraction import extract_features_for_prediction

app = Flask(__name__)

# Load both trained models
rf_model = joblib.load("phishing_model_rf.pkl")
xgb_model = joblib.load("phishing_model_xgb.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form.get("url", "").strip()
        model = request.form.get("model", "both")

        if not url:
            return jsonify({"error": "No URL provided"})

        # Normalize URL — add http:// if no prefix given
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "http://" + url

        try:
            features = extract_features_for_prediction(url)
            response = {}

            if model in ["rf", "both"]:
                rf_probs = rf_model.predict_proba(features)[0]
                rf_pred = rf_model.predict(features)[0]
                rf_phishing_prob = int(rf_probs[1] * 100)
                rf_safe_prob = int(rf_probs[0] * 100)
                rf_conf = rf_safe_prob if rf_pred == 0 else rf_phishing_prob
                response["rf_label"] = "Safe" if rf_pred == 0 else "Phishing"
                response["rf_conf"] = rf_conf
                response["rf_phishing_prob"] = rf_phishing_prob
                response["rf_safe_prob"] = rf_safe_prob

            if model in ["xgb", "both"]:
                xgb_probs = xgb_model.predict_proba(features)[0]
                xgb_pred = xgb_model.predict(features)[0]
                xgb_phishing_prob = int(xgb_probs[1] * 100)
                xgb_safe_prob = int(xgb_probs[0] * 100)
                xgb_conf = xgb_safe_prob if xgb_pred == 0 else xgb_phishing_prob
                response["xgb_label"] = "Safe" if xgb_pred == 0 else "Phishing"
                response["xgb_conf"] = xgb_conf
                response["xgb_phishing_prob"] = xgb_phishing_prob
                response["xgb_safe_prob"] = xgb_safe_prob

            return jsonify(response)

        except Exception as e:
            return jsonify({"error": str(e)})

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)