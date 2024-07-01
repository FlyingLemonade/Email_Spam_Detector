import numpy as np
import pandas as pd
import tensorflow as tf
from tabulate import tabulate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, SpatialDropout1D, Dropout, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
import pickle
import nltk
from nltk.corpus import stopwords

folder_path = "./LSTM_Model/Model_7/"
stop_words = set(stopwords.words('english'))

# Load the entire model
file_json = folder_path + 'model_bidirectional.json'
file_token = folder_path + 'tokenizer.pkl'
file_weights = folder_path + 'model_bidirectional.weights.h5'

with open(file_json, 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
loaded_model.load_weights(file_weights)

# Load the tokenizer
with open(file_token, 'rb') as handle:
    tokenizer = pickle.load(handle)

print("Loaded model and tokenizer from disk.")

def preprocess_input(text, tokenizer, stop_words):
# Tokenize the text
    words = nltk.word_tokenize(text)
    filtered_sentence = [word.lower() for word in words if word.lower() not in stop_words]

    # Convert back to text after filtering stop words
    filtered_text = " ".join(filtered_sentence)

    # Tokenize the filtered text
    sequences = tokenizer.texts_to_sequences([filtered_text])
    sequence_length = 15
    padded_sequences = pad_sequences(sequences, maxlen=sequence_length, padding='post')
    return padded_sequences

# Example input text
input_text = """
Subject: Weekly Newsletter - July Edition
Hi [Recipient's Name],

We hope this email finds you well. Here's our latest newsletter for July:

1. Featured Article: Tips for Summer Travel
2. Events: Local Community Day this Saturday
3. Product Spotlight: New Arrivals in our Store

Stay tuned for more updates and feel free to reach out if you have any questions!

Best Regards,
[Your Company Name]


"""

# input_text = lines

preprocessed_input = preprocess_input(input_text, tokenizer, stop_words)

# Make a prediction
prediction = (loaded_model.predict(preprocessed_input) > 0.5).astype(int)

# Output the result
if prediction == 0:
    print("spam",prediction[0])
elif prediction == 1 :
    print("Normal",prediction[0])
else:
    print("Unknown")
    # return lines  # Return or process the manipulated email content