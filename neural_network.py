import numpy 
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from numpy import array
from keras.models import model_from_json
import os
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import MinMaxScaler




numpy.random.seed(7)
dataset = numpy.loadtxt("MACCS.csv", delimiter=",")

X = dataset[:,0:166]
Y = dataset[:,166]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



model = Sequential()

model.add(Dense(12, input_dim=166, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size = 10, nb_epoch = 15)

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)
print(cm)

scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save("model.h5")
print("Saved model to disk")







