# train_diabetes_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Step 1: Load the Dataset
file_path = "archive\diabetes.csv"  # Replace with the actual path to your CSV file
df = pd.read_csv(file_path)

# Step 2: Preprocess the Data (if needed)
# Handle missing values and other preprocessing steps as required

# Step 3: Split the Dataset into Features (X) and Target Variable (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Step 4: Split the Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Choose a Machine Learning Model
model = LogisticRegression()

# Step 6: Train the Model
model.fit(X_train, y_train)

# Step 7: Evaluate the Model (Optional)
# Evaluate the model performance on the test set and make adjustments if needed

# Step 8: Save the Trained Model
joblib.dump(model, 'diabetes_prediction_model.joblib')

print("Model training and saving completed.")
