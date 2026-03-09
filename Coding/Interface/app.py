from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix
import os

app = Flask(__name__)
CORS(app)

# ── Serve the HTML interface at http://127.0.0.1:5000 ─────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'interface.html')

# ── Load model, vectorizer & feature list ─────────────────────────────────────
xgb_model      = joblib.load(os.path.join(BASE_DIR, 'xgb_model.pkl'))
tfidf          = joblib.load(os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl'))
model_features = joblib.load(os.path.join(BASE_DIR, 'model_features.pkl'))
print("✅ Model + TF-IDF vectorizer loaded successfully!")

# ── Predict endpoint ───────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # 1. Structured features
    input_df = pd.DataFrame([{
        'employment_type':    data.get('employment_type',    'Unknown'),
        'industry':           data.get('industry',           'Unknown'),
        'department':         data.get('department',         'Unknown'),
        'has_company_logo':   int(data.get('has_company_logo', 0)),
        'telecommuting':      int(data.get('telecommuting',    0)),
        'required_education': data.get('required_education', 'Not Specified'),
        'required_experience':data.get('required_experience','Not Applicable'),
    }])
    input_encoded = pd.get_dummies(input_df, drop_first=True)
    input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)
    X_struct = csr_matrix(input_encoded.astype(float).values)

    # 2. Text features
    description   = data.get('description',  '')
    requirements  = data.get('requirements', '')
    combined_text = description + ' ' + requirements
    X_text = tfidf.transform([combined_text])

    # 3. Combine both
    X_combined = hstack([X_struct, X_text])

    # 4. Predict with tuned threshold
    prob = xgb_model.predict_proba(X_combined)[0][1]
    pred = int(prob >= 0.35)

    # 5. Extract top suspicious words from TF-IDF
    import numpy as np
    tfidf_feature_names = tfidf.get_feature_names_out()
    tfidf_scores = X_text.toarray()[0]                        # TF-IDF score for each word
    xgb_importances = xgb_model.feature_importances_         # model-wide importance per feature
    n_struct = X_struct.shape[1]
    text_importances = xgb_importances[n_struct:]             # only text feature importances

    # Combine TF-IDF presence score × model importance → words that are BOTH present AND important
    combined_scores = tfidf_scores * text_importances
    top_indices = combined_scores.argsort()[::-1][:20]        # top 20

    top_words = [
        {'word': tfidf_feature_names[i], 'score': round(float(combined_scores[i]), 4)}
        for i in top_indices
        if combined_scores[i] > 0
    ]

    return jsonify({
        'verdict':           'FRAUDULENT' if pred == 1 else 'LEGITIMATE',
        'fraud_probability': round(float(prob) * 100, 1),
        'threshold_used':    0.35,
        'top_suspicious_words': top_words
    })

# ── Health check ───────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'XGBoost + TF-IDF'})

if __name__ == '__main__':
    print("Starting Flask server — open http://127.0.0.1:5000 in your browser")
    app.run(debug=True, port=5000)
