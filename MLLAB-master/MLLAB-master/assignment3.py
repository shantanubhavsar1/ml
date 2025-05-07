# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, roc_curve, roc_auc_score

# Define column names
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

# Load dataset and ensure correct header handling
pima = pd.read_csv("diabetes.csv", header=0, names=col_names)

# Convert all values to numeric (handling any non-numeric data)
pima = pima.apply(pd.to_numeric, errors='coerce')

# Handle missing values (drop or fill)
pima.dropna(inplace=True)  # OR use pima.fillna(pima.mean(), inplace=True) to fill missing values

# Split dataset into features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
X = pima[feature_cols]  # Features
y = pima['label']  # Target variable

# Split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

# Instantiate and fit the Logistic Regression model
logreg = LogisticRegression(random_state=16, max_iter=200)  # Increased max_iter to avoid convergence warnings
logreg.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = logreg.predict(X_test)

# Compute confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.title('Confusion Matrix', fontsize=14)
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.xticks(ticks=[0.5, 1.5], labels=['Without Diabetes', 'With Diabetes'])
plt.yticks(ticks=[0.5, 1.5], labels=['Without Diabetes', 'With Diabetes'])
plt.show()

# Print classification report
target_names = ['Without Diabetes', 'With Diabetes']
print(classification_report(y_test, y_pred, target_names=target_names))

# Compute ROC curve and AUC
y_pred_proba = logreg.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

# Plot ROC curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label="AUC = " + str(round(auc, 2)))
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal reference line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc=4)
plt.show()
