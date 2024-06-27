import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer # Feature of extraction tech to use nlp
from sklearn.linear_model import LogisticRegression # Popular regression for binary class or 2 class (True or False)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score # Evaluate Model Performance
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_csv('../dataset/email_spam_indo.csv')

data = df.where((pd.notnull(df)),"")

data.shape

data.loc[data['Kategori'] == 'spam' , 'Kategori',] = 0 # data shown as 0 if it is a spam
data.loc[data['Kategori'] == 'ham', 'Kategori',] = 1 # data shown as 1 if it is not a spam

X = data['Pesan']
Y = data['Kategori']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Define Indonesian stop words
indonesian_stop_words = [
    'dan', 'di', 'yang', 'untuk', 'dengan', 'pada', 'ada', 'dalam', 'itu', 'ini', 'ke', 'dari', 
    'atau', 'akan', 'tersebut', 'oleh', 'sebagai', 'pada', 'juga', 'kami', 'tetapi', 'lebih', 
    'lagi', 'saat', 'bahwa', 'harus', 'semua', 'dia', 'jadi', 'seperti', 'karena', 'sudah', 
    'mereka', 'sangat', 'bukan', 'namun', 'setelah', 'bisa', 'banyak', 'masih', 'hingga', 
    'saat', 'bisa', 'kalau', 'tidak', 'maupun'
]

# Initialize TfidfVectorizer
feature_extraction = TfidfVectorizer(min_df=1, stop_words=indonesian_stop_words, lowercase=True)

# Transform the text data
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode the labels
Y_train = label_encoder.fit_transform(Y_train)
Y_test = label_encoder.transform(Y_test)

model = KNeighborsClassifier(n_neighbors=2)
model.fit(X_train_features, Y_train)

prediction_on_training_data=model.predict(X_train_features)
accuracy_on_training_data=accuracy_score(Y_train,prediction_on_training_data)
print("Accuracy on training data:",accuracy_on_training_data)

prediction_on_test_data=model.predict(X_test_features)
accuracy_on_test_data=accuracy_score(Y_test,prediction_on_test_data)
print("Accuracy on test data:",accuracy_on_test_data)

inputs=""" Tes apa saja yang dapat diambil di luar PCU dan diakui untuk
menggantikan EPT?

Tes yang dapat menggantikan EPT adalah *IELTS, Duolingo*, atau tes di bawah
lisensi Educational Testing Service (ETS) seperti *TOEFL ITP, TOEFL iBT*,
dan *TOEIC*. Info lokasi pelaksanaan tes tsb terlampir (*file FAQ EPT
Terbaru 2023*).

** Sesuai Memo Dinas 0675/UKP/2024, *sejak semester gasal 2024/2025, PCU
tidak mengakomodasi TOEIC sebagai konversi non-EPT.*
"""
input_mail=[str(inputs)]

input_data_features=feature_extraction.transform(input_mail)

print("input_data_features:",input_data_features)

prediction=model.predict(input_data_features)
print("prediction:",prediction)

if prediction[0]==1:
   print("Normal mail",prediction[0])
elif  prediction[0]==0:
   print("spam mail",prediction[0])
else:
   print("unknown condition")

