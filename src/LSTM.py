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

df = pd.read_csv('../dataset/email_spam_indo.csv')

data = df.where((pd.notnull(df)),"")

data.shape

data.loc[data['Kategori'] == 'spam' , 'Kategori'] = 0 # data shown as 0 if it is a spam
data.loc[data['Kategori'] == 'ham', 'Kategori'] = 1 # data shown as 1 if it is not a spam

X = data['Pesan']
Y = data['Kategori']


tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(X)
max_sequence_length = max(len(seq) for seq in sequences)
sequences = pad_sequences(sequences, maxlen=max_sequence_length,padding='post')


sm = SMOTE(random_state=42)
Y = Y.astype(int)
x_resampled, y_resampled = sm.fit_resample(sequences,Y)


class_count_after_oversampling = Counter(y_resampled)
print("Class Counts After Oversampling:")
for label, count in class_count_after_oversampling.items():
   print(f"Class {label}: {count} samples")

X_train, X_test, Y_train, Y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42)

# Training Model

# model_lstm = Sequential()
# model_lstm.add(Embedding(input_dim=vocab_size, output_dim= 128,
#                          input_length=max_sequence_length))
# model_lstm.add(SpatialDropout1D(0.2))
# model_lstm.add(LSTM(64))
# model_lstm.add(Dropout(0.2))
# model_lstm.add(Dense(1, activation='sigmoid'))
# model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model_bidirectional = Sequential()
model_bidirectional.add(Embedding(input_dim=vocab_size, output_dim= 128,
                         input_length=max_sequence_length))
model_bidirectional.add(SpatialDropout1D(0.2))
model_bidirectional.add(Bidirectional(LSTM(64, return_sequences=True)))
model_bidirectional.add(Bidirectional(LSTM(64)))
model_bidirectional.add(Dropout(0.2))
model_bidirectional.add(Dense(1, activation='sigmoid'))
model_bidirectional.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

history_bidirectional = model_bidirectional.fit(X_train, Y_train, epochs=1, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

y_pred_bidirectional = (model_bidirectional.predict(X_test) > 0.5).astype(int)

accuracy_bidirectional = accuracy_score(Y_test,y_pred_bidirectional)

report_bidirectional = classification_report(Y_test, y_pred_bidirectional, target_names=["ham","spam"])
# After model training

model_json = model_bidirectional.to_json()
with open("./LSTM_Model/Model_1/model_bidirectional.json", "w") as json_file:
    json_file.write(model_json)

# Save model weights to HDF5 file
model_bidirectional.save_weights("./LSTM_Model/Model_1/model_bidirectional.weights.h5")

# Save the tokenizer
with open('./LSTM_Model/Model_1/tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("LSTM Model:")
print(f"Accuracy: {accuracy_bidirectional}")
print(report_bidirectional)
print("Saved model and tokenizer to disk.")