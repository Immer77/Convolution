# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:41:07 2021

@author: sila
"""

# import panda library and a few others we will need.
import pandas as pd
import matplotlib.pyplot as plt

col_names = ["Id", "Churn", "Line", "Grade", "Age", "Distance", "StudyGroup"]

data = pd.read_csv('studentchurn.csv', sep=';', header=0, names=col_names)

data.dropna(inplace=True)

# show the data
print(data.describe(include='all'))
# the describe is a great way to get an overview of the data
print(data.values)

print(data.columns)
print(data.shape)

#data[['Id', 'Churn', 'Line', 'Grade', 'Age', 'Distance', 'StudyGroup']] = data[
#    'Id;Churn;Line;Grade;Age;Distance;StudyGroup'].str.split(';', expand=True)

# Replace Data churn from Completed/stopped to 1/0
data['Churn'].replace(['Completed', 'Stopped'], [1, 0], inplace=True)

#print(data['Churn'])
#print(data.describe())

x = data["Line"]
y = data["Age"]
plt.figure()
plt.scatter(x.values, y.values, color='black', s=20)
plt.show()

x = data["Grade"]
y = data["Age"]
plt.figure()
plt.scatter(x.values, y.values, color='black', s=20)
plt.show()

x = data["Line"]
y = data["Grade"]
plt.figure()
plt.scatter(x.values, y.values, color='black', s=20)
plt.show()

x = data["Grade"]
y = data["Churn"]
plt.figure()
plt.scatter(x.values, y.values, color='black', s=20)
plt.show()

# Dropping the ID column since it is not neeeded when training our model
data.drop('Id', axis=1, inplace=True)
data.drop('Line', axis=1, inplace=True)
yvalues = pd.DataFrame(dict(Churn=[]), dtype=int)
yvalues["Churn"] = data["Churn"].copy()
print(yvalues)
data.drop('Churn', axis=1, inplace=True)
print(data)
# Now we need to exclude the rows where there is an empty value


# Now we need to prepare the data by normalizing it
# We will use the standard scaler from sklearn
x_train = data.head(868)
x_test = data.tail(217)

y_train = yvalues.head(868)
y_test = yvalues.tail(217)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)

xtrain = scaler.transform(x_train)
xtest = scaler.transform(x_test)

# Now we need to train our model
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

mlp = MLPClassifier(hidden_layer_sizes=(5, 10), max_iter=1000, random_state=1)
mlp.fit(xtrain, y_train.values)
# Predicting the value Brug af x test
predictions = mlp.predict(xtest)
matrix = confusion_matrix(predictions, y_test)
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print("Prediction:" + str(predictions))
print("Confusion Matrix")
print(matrix)
print("Classification")
print(report)
print(f"Accuracy: {accuracy * 100:.2f}%")






