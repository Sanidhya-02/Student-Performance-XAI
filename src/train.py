
import os
import pandas as pd
import joblib
import shap

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/student_data.csv")

X = df.drop(["pass", "final_score"], axis=1)
y = df["pass"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate on unseen test data
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print(f"Test Accuracy: {accuracy:.2f}")

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save trained model
joblib.dump(model, "models/model.pkl")

# Create and save SHAP explainer
explainer = shap.Explainer(model, X_train)
joblib.dump(explainer, "models/explainer.pkl")

print("Model and SHAP explainer saved successfully.")
