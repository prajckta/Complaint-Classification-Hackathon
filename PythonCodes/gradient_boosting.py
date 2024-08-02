# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:06:47 2024

@author: Shreeya
"""
#gradient boosting
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the CSV dataset
df = pd.read_csv('filtered_tokens_complaints_chunk1.csv')

# Separate features and target variable
X_text = df['tokens']
y = df['product']

# Convert text data into numerical format using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X_text)

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier()

# Fit the model on the training data
gb_clf.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = gb_clf.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the calculated metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Function to classify a complaint
def classify_complaint(complaint):
    complaint_vec = vectorizer.transform([complaint])
    prediction = gb_clf.predict(complaint_vec)
    return prediction[0]

# Example usage
user_complaint = input("Enter your complaint: ")
predicted_product = classify_complaint(user_complaint)
print(f'Predicted Product Type: {predicted_product}')