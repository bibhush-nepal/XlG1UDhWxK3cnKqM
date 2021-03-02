#Here I am using Decision Tree Classifier as it gives the best result for this given dataset out of the other Classifiers.


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/Predator/Downloads/ACME-HappinessSurvey2020.csv")

df1 = df
y = np.array(df.pop('Y'))
print(y.shape)
x = np.array(df)
print(x.shape)

#to view the correlation between the features and the outcomes to determine the best features for the next survey
names = ['X1','X2','X3','X4','X5','X6']
corr_df =[]
for i in names:
    corr = np.corrcoef(df[i], y)
    corr_df.append(corr[0][1])
print(corr_df)

#Looking at the output, we can eliminate the survey question that has a negative correlation with the happiness outcome. As it negatively affects the happiness of the customer
#with the increase in the rating
#the most important question affecting the happiness is the one that has the highest positive value of Pearson's constant.
#       X1                      X2                    X3                  X4                  X5                 X6
#[0.28016013727360045, -0.024274179540955645, 0.15083836544066362, 0.0644150769251196, 0.2245224256809451, 0.1676693237607623]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state=47)
print(x_train.shape)
print(x_test.shape)

model = DecisionTreeClassifier(criterion='gini', random_state=43).fit(x_train, y_train)
print("Model Accuracy: ",model.score(x_train,y_train))

prediction = model.predict(x_test)
print(prediction)
