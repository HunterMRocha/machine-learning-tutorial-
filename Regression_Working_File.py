import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")
# print(data.head()) #prints out all the data

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# print(data.head()) #prints out only the above 6 now

#predict is also known as the label .
predict = "G3"

#all of our features/attr
x = np.array(data.drop([predict], 1))

#all of our labels
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(f"Accuracy: {acc}")
print(f"Coefficients:\n {linear.coef_}")
print(f"Intercept:\n {linear.intercept_}")

predictions = linear.predict(x_test)
print(predictions)

