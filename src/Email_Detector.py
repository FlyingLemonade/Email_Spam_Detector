import mailparser

# Load an email file

email = mailparser.parse_from_file('../dataset/New announcement_ _Perhatian_ 1. UAS Kecerdasan Buatan,…_.eml')

text_email = ''.join(email.text_plain)
lines = text_email.replace("\n", "")
# lines = "Perkenalkan nama saya adalah joni, anda dapat melihat portofolio saya. "

print(lines)   


import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('punkt')

folder_path = "./LSTM_Model/Model_4/"
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

Sorry to be a pain. Is it ok if we meet another night? I spent late afternoon in casualty and that means i haven't done any of y stuff42moro and that includes all my time sheets and that. Sorry.

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