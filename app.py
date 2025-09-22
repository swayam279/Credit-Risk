from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
import shap
from datetime import datetime

# Initialize the Flask app
app = Flask(__name__)

# --- Setup: Load models and artifacts ---
BASE_DIR = Path(__file__).resolve(strict=True).parent
MODEL_DIR = BASE_DIR / "models"

try:
    with open(MODEL_DIR / 'imputer.pkl', 'rb') as f:
        imputer = pickle.load(f)
    with open(MODEL_DIR / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(MODEL_DIR / 'train_cols.pkl', 'rb') as f:
        train_cols = pickle.load(f) # The one-hot encoded column names
    with open(MODEL_DIR / 'categorical_cols.pkl', 'rb') as f:
        categorical_cols = pickle.load(f)
    with open(MODEL_DIR / 'final_model_columns.pkl', 'rb') as f:
        final_model_columns = pickle.load(f) # The final full list of columns
    with open(MODEL_DIR / 'xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError as e:
    raise RuntimeError(f"Model artifact not found. Error: {e}")

explainer = shap.TreeExplainer(model)

# --- GLOBAL CONSTANTS ---
FEATURE_MAP = {
    "AMT_GOODS_PRICE": "Price of Goods", "AMT_INCOME_TOTAL": "Total Income",
    "CREDIT_TERM": "Loan Duration", "DAYS_EMPLOYED": "Years Employed",
    "EXT_SOURCE_1": "External Credit Score 1", "DAYS_BIRTH": "Applicant Age",
    "EXT_SOURCE_2": "External Credit Score 2", "EXT_SOURCE_3": "External Credit Score 3",
    "FLAG_OWN_CAR_N": "Not Owning a Car", "FLAG_OWN_CAR_Y": "Owning a Car",
    "NAME_CONTRACT_TYPE_Cash_loans": "Loan Type: Cash Loan",
    "NAME_EDUCATION_TYPE_Secondary___secondary_special": "Education: Secondary",
    "NAME_EDUCATION_TYPE_Higher_education": "Education: Higher",
    "CREDIT_INCOME_RATIO": "Credit to Income Ratio",
    "ANNUITY_INCOME_RATIO": "Annuity to Income Ratio", "AMT_CREDIT": "Loan Amount",
    "AMT_ANNUITY": "Monthly Payment", "CODE_GENDER_M": "Gender: Male", "CODE_GENDER_F": "Gender: Female",
    "NAME_INCOME_TYPE_Working": "Income Type: Working", "FLAG_WORK_PHONE": "Has Work Phone"
}

NON_INTUITIVE_FEATURES = [
    'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
    'FLAG_DOCUMENT_8', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
    'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'FLAG_WORK_PHONE'
]

# --- App Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        
        # --- Create a single row DataFrame from user input ---
        user_input_df = pd.DataFrame([input_data])

        # --- Data Conversion & Feature Engineering ---
        dob = datetime.strptime(user_input_df['DOB'].iloc[0], '%Y-%m-%d')
        user_input_df['DAYS_BIRTH'] = (dob - datetime.now()).days
        
        years_employed = user_input_df.pop('YEARS_EMPLOYED')
        user_input_df['DAYS_EMPLOYED'] = years_employed * -365

        user_input_df.loc[:, 'CREDIT_INCOME_RATIO'] = user_input_df['AMT_CREDIT'] / user_input_df['AMT_INCOME_TOTAL']
        user_input_df.loc[:, 'ANNUITY_INCOME_RATIO'] = user_input_df['AMT_ANNUITY'] / user_input_df['AMT_INCOME_TOTAL']
        user_input_df.loc[:, 'CREDIT_TERM'] = user_input_df['AMT_ANNUITY'] / user_input_df['AMT_CREDIT']
        
        # --- Preprocessing Pipeline ---
        # 1. Start with an empty DataFrame with all expected final columns, filled with 0s
        input_processed = pd.DataFrame(columns=final_model_columns, index=[0]).fillna(0)
        
        # 2. Process and place numerical features
        numeric_cols_original = imputer.feature_names_in_.tolist()
        engineered_features = ['CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO', 'CREDIT_TERM']
        all_numeric_cols = numeric_cols_original + engineered_features

        temp_numeric_df = pd.DataFrame(columns=all_numeric_cols, index=[0])
        temp_numeric_df.update(user_input_df)

        temp_numeric_df[numeric_cols_original] = imputer.transform(temp_numeric_df[numeric_cols_original])
        temp_numeric_df.fillna(0, inplace=True) # Fill NaNs from division by zero
        temp_numeric_df[all_numeric_cols] = scaler.transform(temp_numeric_df[all_numeric_cols])
        
        # 3. Process and place categorical features
        temp_cat_df = pd.DataFrame(columns=categorical_cols, index=[0])
        temp_cat_df.update(user_input_df)
        temp_cat_encoded = pd.get_dummies(temp_cat_df)
        temp_cat_final = temp_cat_encoded.reindex(columns=train_cols, fill_value=0)

        # 4. Combine all parts into the final DataFrame
        final_df_parts = pd.concat([temp_numeric_df, temp_cat_final], axis=1)
        final_df_parts.columns = ["".join (c if c.isalnum() else '_' for c in str(x)) for x in final_df_parts.columns]
        
        # Update the main processed DataFrame
        input_processed.update(final_df_parts)
        
        # --- Prediction and Explanation ---
        prediction_proba = model.predict_proba(input_processed)[:, 1]
        shap_values = explainer.shap_values(input_processed)
        
        contrib_df = pd.DataFrame(shap_values.T, index=final_model_columns, columns=['contribution'])
        contrib_df = contrib_df[~contrib_df.index.isin(NON_INTUITIVE_FEATURES)]
        
        contrib_df['abs_contribution'] = contrib_df['contribution'].abs()
        contrib_df = contrib_df.sort_values(by='abs_contribution', ascending=False)
        
        positive_contrib = contrib_df[contrib_df['contribution'] > 0]
        negative_contrib = contrib_df[contrib_df['contribution'] < 0]

        user_provided_features_raw = list(user_input_df.columns)

        def format_explanation(contrib_series, user_input_raw):
            explanation_list = []
            user_selected_ohe = []
            for col, val in user_input_raw.items():
                if isinstance(val, str):
                    sanitized_val = val.replace(' / ', '_').replace(' ', '_')
                    user_selected_ohe.append(f"{col}_{sanitized_val}")

            filtered_contrib = contrib_series[
                contrib_series.index.isin(numeric_cols_original) |
                contrib_series.index.isin(engineered_features) |
                contrib_series.index.str.contains('|'.join(user_selected_ohe), na=False)
            ]

            for feature, value in filtered_contrib.head(5).to_dict().items():
                is_user_provided = any(user_feature in feature for user_feature in user_provided_features_raw)
                explanation_list.append({
                    "feature": FEATURE_MAP.get(feature, feature.replace('_', ' ').title()),
                    "value": f"{value:.4f}",
                    "is_user_provided": is_user_provided
                })
            return explanation_list

        summary = (f"The model predicts a {'High' if prediction_proba[0] > 0.5 else 'Low'} risk of default with a probability of {prediction_proba[0]:.2%}. ")
        
        response = {
            "summary": summary,
            "default_probability": f"{float(prediction_proba[0]):.4f}",
            "explanation": {
                "factors_increasing_risk": format_explanation(positive_contrib['contribution'], input_data),
                "factors_decreasing_risk": format_explanation(negative_contrib['contribution'], input_data)
            }
        }
        
        return jsonify(response)

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True)