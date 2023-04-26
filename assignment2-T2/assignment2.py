# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# step 1 
# Load the dataset
data = pd.read_csv("D:/CDU/ML/assignment2/Datasets/D2/healthcare-dataset-stroke-data.csv")
print(data.head())
print(data.info())
print(data.describe())

#Data Pre-Processing
data["bmi"].fillna(data["bmi"].mean(), inplace=True)
data["smoking_status"].replace(
    "Unknown", data["smoking_status"].mode()[0], inplace=True
)
# EDA
#Exploratory Diagram Analysis
sns.countplot(x='stroke', data=data)
plt.title('Count of Stroke Cases')
plt.show()


sns.boxplot(x='stroke', y='age', data=data)
plt.title('Age Distribution by Stroke')
plt.show()

#step 2
# Feature Engineering
# Convert categorical variables to dummy variables
data = pd.get_dummies(data, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])
# Create X and y variables
X = data.drop(['stroke'], axis=1)
y = data['stroke']

# Create Train and Test dfsets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Apply at least 4 algorithms (Training and Testing)
# Algorithm 1: Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Algorithm 2: Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Algorithm 3: Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Algorithm 4: Support Vector Machine
from sklearn.svm import SVC
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
# Generate at least 4 Evaluation Metrics on each algorithm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Evaluation Metrics for Logistic Regression
print('Logistic Regression Metrics:')
print('Accuracy:', accuracy_score(y_test, y_pred_lr))
print('Precision:', precision_score(y_test, y_pred_lr))
print('Recall:', recall_score(y_test, y_pred_lr))
print('F1 Score:', f1_score(y_test, y_pred_lr))
print()

# Evaluation Metrics for Decision Tree
print('Decision Tree Metrics:')
print('Accuracy:', accuracy_score(y_test, y_pred_lr))
