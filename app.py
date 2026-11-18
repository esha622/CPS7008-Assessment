
"""
POWERCO CHURN PREDICTION DASHBOARD

An optional Streamlit frontend for the
CPS7008 machine learning project.

Run:
python -m streamlit run app.py
"""

import json
import joblib
import pandas as pd
import streamlit as st

from ml_utils import (
    OUTPUTS,
    MODELS,
    FEATURES,
    load_customers
)


# CONFIGURE THE APPLICATION

st.set_page_config(
    page_title="PowerCo Churn Dashboard",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ PowerCo Customer Churn Prediction")

st.caption(
    "CPS7008 educational demonstration. "
    "Predictions are not verified business decisions."
)


# CHECK WHETHER A MODEL EXISTS

model_file = MODELS / "churn_pipeline.joblib"
settings_file = MODELS / "settings.joblib"

if not model_file.exists() or not settings_file.exists():

    st.warning(
        "No trained model found. "
        "Run preprocessing, model training "
        "and optimisation first."
    )

    st.stop()


# LOAD THE TRAINED MODEL

@st.cache_resource
def load_model():

    model = joblib.load(model_file)

    settings = joblib.load(settings_file)

    return model, settings


model, settings = load_model()

customers = load_customers()


# CREATE THREE DASHBOARD TABS

tab1, tab2, tab3 = st.tabs([
    "Overview",
    "Predict Customers",
    "Model Results"
])


# TAB 1: CUSTOMER OVERVIEW

with tab1:

    st.subheader("Customer Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Customers",
        f"{len(customers):,}"
    )

    col2.metric(
        "Churned Customers",
        f"{int(customers['churn'].sum()):,}"
    )

    col3.metric(
        "Observed Churn Rate",
        f"{customers['churn'].mean():.1%}"
    )

    st.subheader("Churn Distribution")

    churn_counts = customers[
        "churn"
    ].value_counts().sort_index()

    st.bar_chart(churn_counts)

    st.info(
        "The customer dataset is imbalanced. "
        "Accuracy alone cannot determine "
        "whether churners are detected well."
    )


# TAB 2: CHURN PREDICTION

with tab2:

    st.subheader("Predict Customer Churn Risk")

    st.write(
        "Select an existing customer or "
        "upload a CSV containing the model features."
    )

    uploaded = st.file_uploader(
        "Upload customer CSV",
        type="csv"
    )

    if uploaded is not None:

        try:

            new_data = pd.read_csv(uploaded)

            missing = [
                column
                for column in FEATURES
                if column not in new_data.columns
            ]

            if missing:

                st.error(
                    "Missing columns: "
                    + ", ".join(missing)
                )

            else:

                if st.button("Predict Uploaded Customers"):

                    probabilities = model.predict_proba(
                        new_data[FEATURES]
                    )[:, 1]

                    results = new_data.copy()

                    results["churn_probability"] = (
                        probabilities
                    )

                    results["at_risk"] = (
                        probabilities >= settings["threshold"]
                    ).astype(int)

                    st.dataframe(results.head(100))

                    st.download_button(
                        "Download Predictions",
                        data=results.to_csv(index=False),
                        file_name="powerco_predictions.csv",
                        mime="text/csv"
                    )

        except Exception as error:

            st.error(
                f"Could not process the CSV: {error}"
            )

    else:

        customer_number = st.number_input(
            "Select Customer Row",
            min_value=0,
            max_value=len(customers) - 1,
            value=0,
            step=1
        )

        if st.button("Predict Customer"):

            customer = customers.iloc[
                [int(customer_number)]
            ][FEATURES]

            probability = float(
                model.predict_proba(customer)[0, 1]
            )

            st.metric(
                "Predicted Churn Probability",
                f"{probability:.1%}"
            )

            if probability >= settings["threshold"]:

                st.warning("Customer flagged as at risk")

            else:

                st.success("Customer not flagged")

            st.caption(
                "Validation threshold: "
                f"{settings['threshold']:.2f}"
            )

            st.dataframe(customer.T)


# TAB 3: MODEL EVALUATION

with tab3:

    st.subheader("Model Evaluation Results")

    metrics_file = OUTPUTS / "final_metrics.json"

    if metrics_file.exists():

        with open(
            metrics_file,
            encoding="utf-8"
        ) as file:

            metrics = json.load(file)

        st.json(metrics)

    else:

        st.info(
            "Run 06_model_evaluation.py "
            "to generate evaluation metrics."
        )


    result_files = [

        (
            "model_comparison.csv",
            "Cross-Validation Model Comparison"
        ),

        (
            "feature_importance.csv",
            "Feature Importance"
        ),

        (
            "segment_metrics.csv",
            "Customer Segment Evaluation"
        )
    ]


    for filename, heading in result_files:

        path = OUTPUTS / filename

        if path.exists():

            st.subheader(heading)

            st.dataframe(
                pd.read_csv(path)
            )


    confusion_image = (
        OUTPUTS / "confusion_matrix.png"
    )

    if confusion_image.exists():

        st.image(
            str(confusion_image),
            caption="Final Test Confusion Matrix"
        )


st.divider()

st.caption(
    "This is a retrospective coursework demonstration. "
    "Prediction-time data availability has not been "
    "fully established for operational deployment."
)