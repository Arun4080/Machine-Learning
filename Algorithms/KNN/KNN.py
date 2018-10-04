import numpy as np
import random
import pandas as pd
from collections import Counter

def KNN(data, predict, k=3):
	if len(data) >= k:
		print("len of data is less than value of k")
	distance=[]
	for group in data:
		for features in data[group]:
			#euclidean_distance=np.sqrt(np.sum((np.array(features)-np.array(predict))**2))
			#or change it like 
			euclidean_distance=np.linalg.norm(np.array(features)-np.array(predict))
			distance.append([euclidean_distance, group])
	votes=[i[1] for i in sorted(distance)[:k]]
	result=Counter(votes).most_common(1)[0][0]
	return result

#names=['code_id','Clump_Thickness','Cell_size','Cell_Shape','Marginal_Adhesion','singleEpithelialCellSize','bareNuclei','blandChromatin','normalNucleoli','Mitoses','Class']
dataset=pd.read_csv('breast-cancer.csv')
dataset.drop(['id'], 1, inplace=True)
dataset=dataset.values.tolist()
random.shuffle(dataset)
train=dataset[int(len(dataset)*0.2):]
test=dataset[:int(len(dataset)*0.2)]

train_set={2:[], 4:[]}
test_set={2:[], 4:[]}

[train_set[i[-1]].append(i[:-1]) for i in train]
[test_set[i[-1]].append(i[:-1]) for i in test]

correct, total =0,0
for group in test_set:
	for data1 in test_set[group]:
		vote=KNN(train_set, data1, k=5)
		if group == vote:
			correct+=1
		total +=1
print('Accuracy: ', correct/total)
