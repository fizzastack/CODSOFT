# Iris Flower Classification Project
# CodSoft Internship Task 3

# importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# loading iris dataset
data = sns.load_dataset("iris")

# showing first rows
print("Iris Dataset\n")
print(data.head())

# checking dataset size
print("\nShape of dataset:")
print(data.shape)

# checking null values
print("\nMissing Values:")
print(data.isnull().sum())

# graph for understanding data
sns.pairplot(data, hue="species")
plt.show()

# input and output data
x = data.drop("species", axis=1)
y = data["species"]

# splitting dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2
)

# creating model
knn = KNeighborsClassifier()

# training model
knn.fit(x_train, y_train)

# predicting output
prediction = knn.predict(x_test)

# checking accuracy
acc = accuracy_score(y_test, prediction)

print("\nAccuracy of Model:")
print(acc * 100)

# giving sample flower data
sample_flower = [[5.1, 3.5, 1.4, 0.2]]

# predicting flower name
result = knn.predict(sample_flower)

print("\nPredicted Flower is:")
print(result[0])