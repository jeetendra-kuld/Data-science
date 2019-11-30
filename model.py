
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
import csv
import pandas as pd


dataframe = pd.read_csv("test.csv")
dataset = dataframe.values
X = dataset[:,0:166]
out_csv = 'new-predictions1.csv'


json_file = open('model.json', 'r')

loaded_model_json = json_file.read()


json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")

print("Loaded model from disk")



predictions = loaded_model.predict(X)

rounded = [round(x[0]) for x in predictions]
print(rounded)


df = pd.DataFrame(predictions)

df.to_csv(out_csv, index=False, header=False)

print("Predictions saved to disk: {0}".format(out_csv))
