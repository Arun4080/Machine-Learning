import pandas as pd
import sklearn as sl

from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
#from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.naive_bayes import GaussianNB
#from pandas.plotting import scatter_matrix 
#import matplotlib.pyplot as plt

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
a=pd.read_csv("iris.csv", names=names)

#un-hash for graphs
#a.plot(kind="box", subplots=True, layout=(2,2),sharex=False, sharey=False)
#scatter_matrix(a)
#plt.show()
#plt.clf()

#split out validation dataset
array=a.values
x=array[:,0:4]
y=array[:,4]
validation_size=0.20
seed=7
scoring='accuracy'
X_train, X_validation, Y_train, Y_validation = sl.model_selection.train_test_split(x, y, test_size=validation_size, random_state=seed)

# Spot Check Algorithms
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVM', SVC()))
#other models to try unhash them and their libraries
#models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))

# evaluate each model in turn

'''
results = []
names = []
for name, model in models:
	kfold = sl.model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = sl.model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)'''

#As we can see k Neighbours and SVM gives best result so lets train using them both

#Using KNN
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
#print(predictions)
print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

#Using SVM
SVC = SVC()
SVC.fit(X_train, Y_train)
predictions = SVC.predict(X_validation)
#print(predictions)
print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))