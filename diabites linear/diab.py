import pandas as pd
import joblib

# Load the trained model
loaded_model = joblib.load('diabetes_prediction_model.joblib')

def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    # Example data to match the format of the original dataset
    data = {
        'Pregnancies': [Pregnancies],
        'Glucose': [Glucose],
        'BloodPressure': [BloodPressure],
        'SkinThickness': [SkinThickness],
        'Insulin': [Insulin],
        'BMI': [BMI],
        'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
        'Age': [Age],
    }

    # Create a DataFrame
    df_input = pd.DataFrame(data)

    # Make prediction
    prediction = loaded_model.predict(df_input)

    return prediction[0]

# Example usage
result = predict_diabetes(6, 148, 72, 35, 0, 33.6, 0.627, 50)

print(f"For the given input values, the predicted diabetes outcome is: {result}")
