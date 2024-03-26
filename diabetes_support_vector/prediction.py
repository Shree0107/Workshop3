import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

diabetes_df = pd.read_csv('diabetes.csv')

# Visualizing missing values
sns.heatmap(diabetes_df.isnull(), cmap='viridis', cbar=False)
plt.title('Missing Values')
plt.show()

# Splitting predictors and target variable
X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']

# Creating training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training SVM model
svm_model = SVC(kernel='linear', probability=True)  # Using linear kernel
svm_model.fit(X_train, y_train)

# Predictions on test data
y_pred = svm_model.predict(X_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# ROC curve and AUC
y_prob = svm_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)


# Plotting ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Accuracy visualization
accuracy_values = [accuracy, 1 - accuracy]  
labels = ['Correct Predictions', 'Incorrect Predictions']
plt.figure(figsize=(8, 6))
plt.bar(labels, accuracy_values, color=['green', 'red'])
plt.title('Accuracy of Diabetes Prediction Model')
plt.xlabel('Prediction Outcome')
plt.ylabel('Proportion of Predictions')
plt.show()


# Confusion matrix visualization
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual No Diabetes', 'Actual Diabetes'], columns=['Predicted No Diabetes', 'Predicted Diabetes'])
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()