import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from src.predict import predict_student

# Load model
model = joblib.load("models/model.pkl")
explainer = joblib.load("models/explainer.pkl")

st.set_page_config(layout="wide", page_title="Student Performance Dashboard")

# =========================
# HEADER
# =========================
st.title("🎓 Student Performance Prediction System")
st.caption("AI-powered academic risk & performance analytics")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("📥 Student Input")

study_hours = st.sidebar.slider("Study Hours", 0.0, 10.0, 5.0)
attendance = st.sidebar.slider("Attendance (%)", 0.0, 100.0, 75.0)
assignments = st.sidebar.slider("Assignments", 0.0, 100.0, 60.0)
quizzes = st.sidebar.slider("Quizzes", 0.0, 100.0, 60.0)

input_data = {
    "study_hours": study_hours,
    "attendance": attendance,
    "assignments": assignments,
    "quizzes": quizzes
}

# =========================
# TOP METRICS
# =========================
st.markdown("### 📊 Prediction Summary")

c1, c2, c3 = st.columns(3)

if st.button("Run Prediction"):
    result = predict_student(input_data)

    c1.metric("Prediction", result["prediction"])
    c2.metric("Probability", f"{result['probability']:.2f}")
    c3.metric("Risk", result["risk_level"])

    st.markdown("**Recommended Actions:**")
    for i in result["interventions"]:
        st.write(f"- {i}")

# =========================
# INSIGHTS SECTION
# =========================
st.markdown("---")
st.markdown("### 🔍 Model Insights")

col1, col2 = st.columns(2)

# -------- SHAP --------
with col1:
    df = pd.DataFrame([input_data])
    shap_values = explainer(df)

    values = shap_values.values
    if len(values.shape) == 3:
        values = values[0, :, 1]
    elif len(values.shape) == 2:
        values = values[0]

    values = values.flatten()

    shap_df = pd.DataFrame({
        "Feature": df.columns,
        "Impact": values
    }).sort_values(by="Impact", key=abs)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.barh(shap_df["Feature"], shap_df["Impact"])
    ax.set_title("Feature Impact", fontsize=10)
    ax.tick_params(labelsize=8)

    st.pyplot(fig, use_container_width=False)

# -------- WHAT IF --------
with col2:
    sim_hours = list(range(1, 11))
    probs = []

    for h in sim_hours:
        temp = input_data.copy()
        temp["study_hours"] = h
        res = predict_student(temp)
        probs.append(res["probability"])

    fig2, ax2 = plt.subplots(figsize=(4, 3))
    ax2.plot(sim_hours, probs, marker='o')
    ax2.set_title("Study Hours vs Probability", fontsize=10)
    ax2.tick_params(labelsize=8)

    st.pyplot(fig2, use_container_width=False)

# =========================
# BATCH
# =========================
st.markdown("---")
st.markdown("### 📂 Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df_batch = pd.read_csv(uploaded_file)

    results = []
    for _, row in df_batch.iterrows():
        res = predict_student(row.to_dict())
        results.append(res)

    result_df = pd.DataFrame(results)
    final_df = pd.concat([df_batch, result_df], axis=1)

    st.dataframe(final_df, height=250)

    # Summary
    c1, c2, c3 = st.columns(3)
    c1.metric("Students", len(final_df))
    c2.metric("High Risk", (final_df["risk_level"] == "High Risk").sum())
    c3.metric("Pass Rate", f"{(final_df['prediction'] == 'Pass').mean():.2f}")

    # Risk chart (compact)
    risk_counts = final_df["risk_level"].value_counts()

    fig3, ax3 = plt.subplots(figsize=(4, 3))
    ax3.bar(risk_counts.index, risk_counts.values)
    ax3.set_title("Risk Distribution", fontsize=10)
    ax3.tick_params(labelsize=8)

    st.pyplot(fig3, use_container_width=False)

    st.download_button(
        "Download Results",
        final_df.to_csv(index=False),
        "results.csv"
    )

# =========================
# MODEL OVERVIEW
# =========================
st.markdown("---")
st.markdown("### 📈 Model Overview")

df = pd.read_csv("data/student_data.csv")
X = df.drop(["pass", "final_score"], axis=1)
y = df["pass"]

preds = model.predict(X)
accuracy = (preds == y).mean()

m1, m2, m3 = st.columns(3)
m1.metric("Accuracy", f"{accuracy:.2f}")
m2.metric("Dataset", len(df))
m3.metric("Pass Rate", f"{y.mean():.2f}")

