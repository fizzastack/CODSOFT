# Titanic Survival Prediction
# CodSoft Internship Task 1

# importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading dataset
data = pd.read_csv("Titanic-Dataset.csv")

# showing first 5 rows
print("Titanic Dataset\n")
print(data.head())

# selecting useful columns
data = data[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]

# removing empty values
data = data.dropna()

# converting male and female into numbers
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# visualization 1
sns.countplot(x='Survived', data=data)

plt.title("Survival Count")

plt.xlabel("0 = Not Survived , 1 = Survived")

plt.ylabel("Passengers")

plt.show()

# visualization 2
sns.histplot(data['Age'])

plt.title("Age Distribution")

plt.xlabel("Age")

plt.show()

# calculating survival percentage
survived = data['Survived'].sum()

total = len(data)

percentage = (survived / total) * 100

print("\nSurvival Percentage:")
print(round(percentage, 2), "%")

# input and output
x = data[['Pclass', 'Sex', 'Age', 'Fare']]
y = data['Survived']

# splitting dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2
)

# creating model
model = LogisticRegression()

# training model
model.fit(x_train, y_train)

# prediction
y_pred = model.predict(x_test)

# model accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")

# custom prediction
sample = [[1, 1, 22, 100]]

result = model.predict(sample)

print("\nPrediction Result:")

if result[0] == 1:
    print("Passenger Survived")
else:
    print("Passenger Did Not Survive")