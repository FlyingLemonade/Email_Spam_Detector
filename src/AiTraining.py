import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'email': [
        'Free money offer just for you',
        'Hey, how have you been?',
        'Lowest price on new phones',
        'Lunch tomorrow at our place?',
        'Win a brand new car'
    ],
    'label': ['spam', 'not spam', 'spam', 'not spam', 'spam']
}

# Create DataFrame
df = pd.DataFrame(data)

# Feature extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['email'])

# Labels
y = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Example of checking a new email
new_email = ["Congratulations, you have won a lottery!"]
new_email_transformed = vectorizer.transform(new_email)
prediction = knn.predict(new_email_transformed)
print(f'The new email is: {prediction[0]}')
