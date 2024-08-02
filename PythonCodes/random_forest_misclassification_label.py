#Random forest with sub category sentiment and misclassification label 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.stem import PorterStemmer
import re
from textblob import TextBlob

# Load the CSV dataset
data = pd.read_csv('merged_file1.csv')

data['tokens'] = data['tokens'].apply(lambda x: ' '.join(eval(x)))

X = data['tokens']
y = data['product']

#Stemming function for input complaint 
stemmer = PorterStemmer()

def stem_words(text):
    words = re.findall(r'\w+', text)
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

X_stemmed = X.apply(stem_words)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X_stemmed)

# Split the data into training and testing sets(70-30)
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)
print(f'Training data shape: {X_train.shape}, {y_train.shape}')
print(f'Testing data shape: {X_test.shape}, {y_test.shape}')
print("-" * 50)

# Train a Random Forest Classifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

#evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the evaluation metrics
print("Evaluation Metrics:")
metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1}
for i, (metric, value) in enumerate(metrics.items(), start=1):
    print(f"{i}. {metric}: {value}")
print("-" * 50)

# Input complaint from the user
complaint = input("Enter your complaint: ")

# Stem the input complaint
stemmed_complaint = stem_words(complaint)

# Check if at least 5 stemmed words from the complaint are in the stemmed words from the CSV file
complaint_words = stemmed_complaint.split()
matching_words = set(complaint_words) & set(X_stemmed.str.split().explode().unique())

if len(matching_words) >= 5:
    # Vectorize the input complaint
    complaint_vectorized = vectorizer.transform([stemmed_complaint])

    # Predict the product type
    predicted_product = classifier.predict(complaint_vectorized)
    print("-" * 50)
    print(f'Predicted Product Type: {predicted_product[0]}')
    
    # Perform sentiment analysis using TextBlob
    sentiment = TextBlob(complaint).sentiment

    # Print the sentiment analysis results
    print(f'Sentiment: {sentiment.polarity}')  # Polarity ranges from -1 (negative) to 1 (positive)

    if sentiment.polarity < 0:
        print("Negative Sentiment")
    elif sentiment.polarity == 0:
        print("Neutral Sentiment")

    # Check for subcategories
    if any(word in complaint.lower() for word in ['angry', 'annoyed', 'frustrated', 'irritated']):
        print("Subcategory: Anger")
    elif any(word in complaint.lower() for word in ['disappointed', 'let down', 'unhappy', 'unsatisfied']):
        print("Subcategory: Disappointment")
    elif any(word in complaint.lower() for word in ['helpless', 'powerless', 'hopeless', 'despair']):
        print("Subcategory: Helplessness")
    else:
        print("Subcategory: Other")
else:
    print("Invalid Input: Please provide a valid complaint")


