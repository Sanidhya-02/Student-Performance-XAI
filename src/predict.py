import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

def get_risk(prob):
    if prob < 0.4:
        return "High Risk"
    elif prob < 0.7:
        return "Medium Risk"
    else:
        return "Low Risk"

def get_intervention(data):
    suggestions = []

    if data["attendance"] < 75:
        suggestions.append("Improve class attendance")

    if data["study_hours"] < 4:
        suggestions.append("Increase daily study hours")

    if data["assignments"] < 60:
        suggestions.append("Focus on assignment completion")

    if data["quizzes"] < 60:
        suggestions.append("Revise concepts and practice quizzes")

    return suggestions

def predict_student(input_dict):
    df = pd.DataFrame([input_dict])

    prob = model.predict_proba(df)[0][1]
    pred = model.predict(df)[0]

    return {
        "prediction": "Pass" if pred == 1 else "Fail",
        "probability": float(prob),
        "risk_level": get_risk(prob),
        "interventions": get_intervention(input_dict)
    }


# test
if __name__ == "__main__":
    sample = {
        "study_hours": 3,
        "attendance": 65,
        "assignments": 55,
        "quizzes": 50
    }

    print(predict_student(sample))