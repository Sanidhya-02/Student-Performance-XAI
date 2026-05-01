import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import shap

df = pd.read_csv("data/student_data.csv")

X = df.drop(["pass", "final_score"], axis=1)
y = df["pass"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

# Save model
joblib.dump(model, "models/model.pkl")

# ✅ NEW SHAP (FINAL)
explainer = shap.Explainer(model, X_train)
joblib.dump(explainer, "models/explainer.pkl")