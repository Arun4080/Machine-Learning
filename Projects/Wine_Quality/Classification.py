import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

a=pd.read_csv("wineQualityWhites.csv")
#print(a.head())
array=a.values
x = array[ : , 1:12]
y = array[ : , 12]
validation_size = 0.20
seed = 3
X_train , X_Validation , Y_train , Y_Validation = train_test_split(x , y , test_size=validation_size , random_state=seed)

#training and tesing using SVM
'''
SVC=SVC()
SVC.fit(X_train,Y_train)
prediction=SVC.predict(X_Validation)
print(accuracy_score(Y_Validation, prediction))'''

#training and tesing using SVM

KNN=KNeighborsClassifier(n_jobs=-1)
KNN.fit(X_train,Y_train)
prediction=KNN.predict(X_Validation)
print(accuracy_score(Y_Validation, prediction))


#X=a[["fixed.acidity","volatile.acidity","citric.acid","residual.sugar","chlorides","free.sulfur.dioxide","total.sulfur.dioxide","density","pH","sulphates","alcohol"]]
#Y=a["quality"]


