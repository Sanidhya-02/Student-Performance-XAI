from fastapi import FastAPI
from src.predict import predict_student

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Student Performance API Running"}

@app.post("/predict")
def predict(study_hours: float, attendance: float, assignments: float, quizzes: float):

    input_data = {
        "study_hours": study_hours,
        "attendance": attendance,
        "assignments": assignments,
        "quizzes": quizzes
    }

    return predict_student(input_data)