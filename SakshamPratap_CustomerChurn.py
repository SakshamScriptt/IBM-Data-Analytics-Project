import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the Dataset
# We are using a standard public dataset URL for Customer Churn
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

print("Data Analytics with AI - Customer Churn Project")
print("------------------------------------------------")
print(f"Dataset loaded successfully. Shape: {df.shape}\n")

# 2. Data Cleaning & Preprocessing
# Convert TotalCharges to numeric, dropping errors
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Convert Churn to binary (1 = Yes, 0 = No)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 3. Exploratory Data Analysis (EDA)
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Churn')
plt.title('Customer Churn Distribution')
plt.savefig('churn_distribution.png')
print("EDA complete. Visualization saved as 'churn_distribution.png'.\n")

# 4. Feature Engineering for AI Model
# Selecting key features for the prediction model
features = ['tenure', 'MonthlyCharges', 'TotalCharges']
X = df[features]
y = df['Churn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. AI / Machine Learning Modeling
print("Training Random Forest AI Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Model Evaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))
print("\nProject execution completed successfully.")