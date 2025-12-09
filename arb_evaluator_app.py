import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load preprocessed data
preprocessed_file = r'C:\Users\anitr\AAI590_Capstone\AAI590_Capstone_AH\Data\ea_modelset\eamodelset\dataset\preprocessed_models.csv'
model_path = r'C:\Users\anitr\AAI590_Capstone\AAI590_Capstone_AH\Models\tuned_and_hybrid_models\rf_tuned.pkl'
metrics = r'C:\Users\anitr\AAI590_Capstone\AAI590_Capstone_AH\Models\tuned_and_hybrid_models\model_metrics.csv'

# Use template columns as features, drop known non-feature columns
df_template = pd.read_csv(preprocessed_file)
X_columns = df_template.columns.drop(['arb_outcome', 'name', 'id'], errors='ignore')

# Load models
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Could not load RF model: {e}")
    model = None

label_mapping = {0: 'Approve', 1: 'Needs Work', 2: 'Reject'}

# Streamlit App Set Up
st.set_page_config(page_title="ARB Architecture Model Evaluator", layout="wide")
st.title("ARB Architecture Model Evaluator")

st.markdown(
    """
    This application predicts the likely **Architecture Review Board (ARB)** outcomes for an enterprise architecture model
    based on high-level metrics provided by the user.

    Use the sidebar to input metrics such as view count, element count, relationship count, and other relevant features.
    Upon clicking the "Predict ARB Outcome" button, the model will analyze the inputs and provide:

    - The predicted ARB outcome category: Approve, Needs Work, or Reject.
    - The probabilities associated with each possible outcome.
    - A local feature importance explanation plot illustrating feature contributions to the prediction.
    - A model comparison section displaying performance metrics of different models.

    *Note: This tool is intended for preliminary assessments and should not replace comprehensive reviews.*
    """
)

st.sidebar.header("Input Architecture Model Metrics")

# Sidebar inputs
viewCount = st.sidebar.number_input("View Count (number of views)", min_value=0, value=10)
elementCount = st.sidebar.number_input("Element Count (number of elements)", min_value=0, value=50)
relationshipCount = st.sidebar.number_input("Relationship Count (number of relationships)", min_value=0, value=80)
duplicateCount = st.sidebar.number_input("Duplicate Count (number of duplicate elements)", min_value=0, value=0)
hasWarnings = st.sidebar.checkbox("Has Warnings?", value=False)
hasDuplicates = st.sidebar.checkbox("Has Duplicates?", value=False)

# Options for source and language
source_options = ['GitHub', 'GenMyModel', 'Other', 'Unknown']
language_options = ['en', 'es', 'pt', 'de', 'fr', 'Other']

source = st.sidebar.selectbox("Source", options=source_options)
language = st.sidebar.selectbox("Language", options=language_options)

st.sidebar.markdown("---")
predict_button = st.sidebar.button("Predict ARB Outcome")

# Function to build feature row
def build_feature_row():
    row = pd.DataFrame(0, index=[0], columns=X_columns)

    # Numeric features
    for col, value in [
        ("viewCount", viewCount),
        ("elementCount", elementCount),
        ("relationshipCount", relationshipCount),
        ("duplicateCount", duplicateCount),
    ]:
        if col in row.columns:
            row[col] = value

    # Flag features
    if "hasWarning" in row.columns:
        row["hasWarning"] = int(hasWarnings)
    if "hasWarnings" in row.columns:
        row["hasWarnings"] = int(hasWarnings)
    if "hasDuplicate" in row.columns:
        row["hasDuplicate"] = int(hasDuplicates)
    if "hasDuplicates" in row.columns:
        row["hasDuplicates"] = int(hasDuplicates)

    # Ratios
    if "rel_elem_ratio" in row.columns:
        if elementCount > 0:
            row["rel_elem_ratio"] = relationshipCount / elementCount
        else:
            row["rel_elem_ratio"] = 0

    if "view_elem_ratio" in row.columns:
        if elementCount > 0:
            row["view_elem_ratio"] = viewCount / elementCount
        else:
            row["view_elem_ratio"] = 0

    # Source 
    for col in row.columns:
        if col.startswith("source_"):
            row[col] = 0
    source_col = "source_GitHub" if source == "GitHub" else f"source_{source}"
    if source_col in row.columns:
        row[source_col] = 1

    # Language 
    for col in row.columns:
        if col.startswith("language_"):
            row[col] = 0
    language_col = "language_en" if language == "en" else f"language_{language}"
    if language_col in row.columns:
        row[language_col] = 1

    return row

# Tabs for prediction explanation and model comparison
tab_pred, tab_compare = st.tabs(["Prediction Explanation", "Model Comparison"])

with tab_pred:

    if predict_button:
        if model is None:
            st.error("Prediction model not loaded.")
        else:
            row = build_feature_row()
            prediction = model.predict(row)[0]
            probabilities = model.predict_proba(row)[0]

            pred_label = label_mapping.get(prediction, str(prediction))

            color_map = {'Approve': 'green', 'Needs Work': 'orange', 'Reject': 'red'}

            st.markdown(
                f"### Predicted ARB Outcome: "
                f"<span style='color: {color_map.get(pred_label, 'black')}; font-weight: bold;'>{pred_label}</span>",
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)
            
            with col1:
                prob_df = pd.DataFrame({
                    "ARB Outcome": [label_mapping[i] for i in range(len(probabilities))],
                    "Probability": probabilities
                }).set_index("ARB Outcome")

                st.write("**Prediction Confidence (Class Probabilities):**")
                st.bar_chart(prob_df)
                st.caption("Note: Higher probability indicates greater confidence in the prediction.")

            with col2:
                st.write("**Feature Importance (Random Forest Model):**")

                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feat_importance = pd.Series(importances, index=X_columns).sort_values(ascending=False)
                    top_features = feat_importance.head(10)

                    st.write("Top 5 features contributing to the prediction:")
                    st.bar_chart(top_features)
                    st.caption("Note: Feature importance is based on the Random Forest model's internal metrics.")
                else:
                    st.warning("Model does not provide feature importances.")

with tab_compare:
    st.subheader("Model Performance Comparison")

    if os.path.exists(metrics):
        metrics_df = pd.read_csv(metrics)
        st.write("Summary of model performance metrics:")
        st.dataframe(metrics_df.style.format({"Accuracy": "{:.2f}", "F1-Score": "{:.2f}"}))
        try:
            sorted_df = metrics_df.sort_values(by='Accuracy', ascending=False)
            st.markdown("#### Accuracy by Model")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(sorted_df['Model'], sorted_df['Accuracy'])
            ax.set_xlabel("Accuracy")
            ax.set_xlim(0.0, 1.01)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not plot metrics: {e}")
    else:
        st.warning("Model metrics file not found.")
