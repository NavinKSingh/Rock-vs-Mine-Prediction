import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sonar_data = pd.read_csv('D:\Rock vs Mine Predction\Copy of sonar data.csv',header=None)

sonar_data.head()

sonar_data.shape

sonar_data.size

sonar_data.describe()

sonar_data[60].value_counts()

sonar_data[60].value_counts().sum()

sonar_data.groupby(60).mean()

x = sonar_data.drop(columns=60,axis=1)
y = sonar_data[60]

print(x)

print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,stratify=y,random_state=1)

print(x.shape,x_train.shape,x_test.shape)

print(x_train)
print(y_train)

model = LogisticRegression()

model.fit(x_train,y_train)

x_train_predction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_predction,y_train)

print("Accuracy Score of Training data is:",training_data_accuracy)

x_test_predction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_predction,y_test)

print("Accuracy Score of Test data is: ",test_data_accuracy)

input_data = input("Enter data (comma-separated): ")
input_data_list = [float(x) for x in input_data.split(',')]
input_data_as_numpy_array = np.asarray(input_data_list)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)
if prediction[0] == 'R':
    print("Object is Rock")
else:
    print("Object is Mine")