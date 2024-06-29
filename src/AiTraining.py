import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Function to clean email text
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove special characters and numbers
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Read data from CSV file
df = pd.read_csv('mail_data.csv')

# Clean email text
df['email'] = df['email'].apply(clean_text)

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=10000)  # Increased vocabulary size
tokenizer.fit_on_texts(df['email'])
X = tokenizer.texts_to_sequences(df['email'])
X = pad_sequences(X, padding='post', maxlen=100)  # Increased maxlen

# Labels
y = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=100))  # Increased embedding size
model.add(LSTM(128, return_sequences=True))  # Increased LSTM units
model.add(Dropout(0.5))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with a lower learning rate
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy}')

# Example of checking a new email
new_email = ["Congratulations, you have won a lottery!"]
new_email_clean = [clean_text(email) for email in new_email]
new_email_seq = tokenizer.texts_to_sequences(new_email_clean)
new_email_pad = pad_sequences(new_email_seq, padding='post', maxlen=100)
prediction = model.predict(new_email_pad)
print(f'The new email is: {"spam" if prediction[0][0] > 0.5 else "not spam"}')
