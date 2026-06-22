import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("D:/heart.csv")

# Data Preprocessing

print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.describe())

# Showing the Histogram

plt.figure(figsize=(8,5))
plt.hist(df['age'], bins=10)
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.title("Age Distribution of Patients")
plt.show()

# Disease vs No Disease Count Plot

plt.figure(figsize=(6,5))
df['target'].value_counts().plot(kind='bar')
plt.xlabel("Heart Disease")
plt.ylabel("Count")
plt.title("Heart Disease Distribution")
plt.xticks([0,1], ['No Disease','Disease'])
plt.show()

# Heatmap Representation

correlation = df.corr()

plt.figure(figsize=(12,8))
sn.heatmap(
correlation,
annot=True,
cmap="coolwarm",
fmt=".2f"
)
plt.title("Correlation Heatmap of Heart Disease Dataset")
plt.show()

# Splitting the data

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)

# Logistic Regression Model


log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

print("Values of y_test values:")
print(y_test.values[:30])

print("Predicted values by Logistic Regression:")
print(y_pred_log[:30])

print("Logistic Regression Accuracy:")
print(accuracy_score(y_test, y_pred_log))

cm_log = confusion_matrix(y_test, y_pred_log)

print("Confusion Matrix (Logistic Regression):")
print(cm_log)

plt.figure(figsize=(7,5))
sn.heatmap(cm_log, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

print("Classification Report (Logistic Regression)")
print(classification_report(y_test, y_pred_log))

# Random Forest Model

rf_model = RandomForestClassifier(
n_estimators=100,
random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("Predicted values by Random Forest:")
print(y_pred_rf[:30])

print("Random Forest Accuracy:")
print(accuracy_score(y_test, y_pred_rf))

cm_rf = confusion_matrix(y_test, y_pred_rf)

print("Confusion Matrix (Random Forest):")
print(cm_rf)

plt.figure(figsize=(7,5))
sn.heatmap(cm_rf, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Confusion Matrix - Random Forest')
plt.show()

print("Classification Report (Random Forest)")
print(classification_report(y_test, y_pred_rf))

# Sample Predictions
sample = pd.DataFrame(
[[63,1,3,145,233,1,0,150,0,2.3,0,0,1]],
columns=X.columns
)

sample1 = pd.DataFrame(
[[52,1,0,125,212,0,1,168,0,1.0,2,2,3]],
columns=X.columns
)
# Logistic Regression Predictions
print("Logistic Regression Prediction:")
print(log_model.predict(sample))
print(log_model.predict(sample1))

# Random Forest Predictions

print("Random Forest Prediction:")
print(rf_model.predict(sample))
print(rf_model.predict(sample1))

prediction = rf_model.predict(sample)

if prediction[0] == 1:
    print("Heart Disease Present")
else:
    print("No Heart Disease")

prediction1 = rf_model.predict(sample1)

if prediction1[0] == 1:
    print("Heart Disease Present")
else:
    print("No Heart Disease")
