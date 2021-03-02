#Here I am using Decision Tree Classifier as it gives the best result for this given dataset out of the other Classifiers.


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("C:/Users/Predator/Downloads/ACME-HappinessSurvey2020.csv")

df1 = df
y = np.array(df.pop('Y'))
print(y.shape)

x = np.array(df)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state=47)
print(x_train.shape)
print(x_test.shape)

model = DecisionTreeClassifier(criterion='gini', random_state=43).fit(x_train, y_train)
print("Model Accuracy: ",model.score(x_train,y_train))

prediction = model.predict(x_test)
print(prediction)
