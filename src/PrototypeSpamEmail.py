import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer # Feature of extraction tech to use nlp
from sklearn.linear_model import LogisticRegression # Popular regression for binary class or 2 class (True or False)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score # Evaluate Model Performance
import matplotlib.pyplot as plt

df = pd.read_csv('../dataset/mail_data.csv')

data = df.where((pd.notnull(df)),"")

data.shape

data.loc[data['Category'] == 'spam' , 'Category',] = 0 # data shown as 0 if it is a spam
data.loc[data['Category'] == 'ham', 'Category',] = 1 # data shown as 1 if it is not a spam

X = data['Message']
Y = data['Category']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)

X_train_features=feature_extraction.fit_transform(X_train)
X_test_features=feature_extraction.transform(X_test)

Y_train=Y_train.astype('int')
Y_test=Y_test.astype('int')


model = KNeighborsClassifier(n_neighbors=2)
model.fit(X_train_features, Y_train)

prediction_on_training_data=model.predict(X_train_features)
accuracy_on_training_data=accuracy_score(Y_train,prediction_on_training_data)
print("Accuracy on training data:",accuracy_on_training_data)

prediction_on_test_data=model.predict(X_test_features)
accuracy_on_test_data=accuracy_score(Y_test,prediction_on_test_data)
print("Accuracy on test data:",accuracy_on_test_data)

inputs="""Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"""
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
