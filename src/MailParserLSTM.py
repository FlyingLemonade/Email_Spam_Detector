import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the entire model
with open('./LSTM_Model/Model_1/model_bidirectional.json', 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
loaded_model.load_weights('./LSTM_Model/Model_1/model_bidirectional.weights.h5')

# Load the tokenizer
with open('./LSTM_Model/Model_1/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

print("Loaded model and tokenizer from disk.")

def preprocess_input(text, tokenizer):
    # Tokenize the text
    sequences = tokenizer.texts_to_sequences([text])
    sequence_length = len(sequences[0])
    padded_sequences = pad_sequences(sequences, maxlen=sequence_length, padding='post')
    return padded_sequences

# Example input text
input_text = "Aku Senang Kamu Datang Besok"

preprocessed_input = preprocess_input(input_text, tokenizer)

# Make a prediction
prediction = (loaded_model.predict(preprocessed_input) > 0.5).astype(int)

print(prediction)
# Output the result
if prediction == 0:
    print("spam",prediction[0])
else:
    print("Normal",prediction[0])
