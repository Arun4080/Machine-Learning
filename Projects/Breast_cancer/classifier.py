import pandas as pd
import sklearn as sl

from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

#dataSet
names=['code_id','Clump_Thickness','Cell_size','Cell_Shape','Marginal_Adhesion','singleEpithelialCellSize','bareNuclei','blandChromatin','normalNucleoli','Mitoses','Class']
dataset=pd.read_csv('breast-cancer.csv',names=names)

dataset['bareNuclei']=dataset['bareNuclei'].replace("?",0)

array=dataset.values
x=array[:,1:10]
y=array[:,10]
validation_size=0.20
seed=7
X_train,X_validation,Y_train,Y_validation=sl.model_selection.train_test_split(x,y, test_size=validation_size, random_state=seed)

#By using SVC
svc=SVC()
svc.fit(X_train,Y_train)
predictions = svc.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

#By using KNN
KNN=KNeighborsClassifier()
KNN.fit(X_train,Y_train)
predictions = KNN.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
