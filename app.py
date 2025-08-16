import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Women's Literacy Predictor",
    page_icon="👩‍🎓",
    layout="wide"
)

# --- Load Data and Model ---
@st.cache_data
def load_data():
    with open('app_data.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

app_data = load_data()
model = app_data['model']
df = app_data['data']
features = app_data['features']

# --- Helper function for labels ---
def get_label(feature_name):
    labels = {
        "Female population age 6 years and above who ever attended school (%)": "Females Who Attended School (%)",
        "Women (age 15-49)  with 10 or more years of schooling (%)": "Women with 10+ Years of Schooling (%)",
        "Women (age 15-49)  who have ever used the internet (%)": "Women Who Used Internet (%)",
        "Population living in households with electricity (%)": "Households with Electricity (%)",
        "Households using clean fuel for cooking3 (%)": "Households with Clean Cooking Fuel (%)",
        "Population living in households that use an improved sanitation facility2 (%)": "Households with Improved Sanitation (%)"
    }
    return labels.get(feature_name, feature_name)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Literacy Predictor", "Model Insights", "Data Explorer", "Batch Prediction"])

# =============================================
# PAGE 1: LITERACY PREDICTOR
# =============================================
if page == "Literacy Predictor":
    st.title("👩‍🎓 Women's Literacy Rate Predictor")
    st.markdown("Use the input boxes to set regional data and predict the literacy rate for women aged 15-49.")

    st.header("Input Features")
    col1, col2 = st.columns(2)
    user_inputs = {}

    # Dynamically create number inputs
    for i, feature in enumerate(features):
        container = col1 if i < len(features) / 2 else col2
        # --- MODIFICATION: Changed st.slider to st.number_input ---
        user_inputs[feature] = container.number_input(
            label=get_label(feature),
            min_value=0.0, max_value=100.0, value=50.0, step=0.1, format="%.1f"
        )

    if st.button("Predict Literacy Rate", type="primary"):
        input_df = pd.DataFrame([user_inputs])

        all_tree_predictions = [tree.predict(input_df)[0] for tree in model.estimators_]
        mean_prediction = np.mean(all_tree_predictions)
        std_dev = np.std(all_tree_predictions)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=mean_prediction,
            title={'text': "Predicted Women's Literacy Rate"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "red"}}
        ))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.metric(
            label="Model Confidence Range",
            value=f"{mean_prediction:.2f}%",
            delta=f"± {std_dev:.2f}%",
            delta_color="off"
        )
        st.info("The `±` value represents the standard deviation of predictions from all trees in the model, indicating the model's certainty.")

# =============================================
# PAGE 2: MODEL INSIGHTS
# =============================================
elif page == "Model Insights":
    st.title("🧠 Model Insights")
    st.markdown("Understand which factors are the most important for predicting literacy rates.")

    st.header("Feature Importance")
    st.markdown("This chart shows the relative importance of each feature in the model's predictions. Higher values mean the feature has a greater impact.")

    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': [get_label(f) for f in features],
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)

    fig = px.bar(
        feature_importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Feature Importance Chart"
    )
    fig.update_layout(yaxis_title="", xaxis_title="Importance Score")
    st.plotly_chart(fig, use_container_width=True)

# =============================================
# PAGE 3: DATA EXPLORER
# =============================================
elif page == "Data Explorer":
    st.title("📊 Data Explorer")
    st.markdown("View the cleaned dataset that was used to train the model.")
    st.dataframe(df)

# =============================================
# PAGE 4: BATCH PREDICTION
# =============================================
elif page == "Batch Prediction":
    st.title("📁 Batch Prediction")
    st.markdown("Upload a CSV file with the required columns to get predictions for multiple regions at once.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:")
            st.dataframe(input_df.head())

            if all(col in input_df.columns for col in features):
                predictions = model.predict(input_df[features])
                result_df = input_df.copy()
                result_df['Predicted_Literacy_Rate'] = predictions

                st.write("Predictions:")
                st.dataframe(result_df)

                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name='predictions.csv',
                    mime='text/csv',
                )
            else:
                st.error(f"Error: The uploaded CSV must contain the following columns: {', '.join(features)}")
        except Exception as e:
            st.error(f"An error occurred: {e}")