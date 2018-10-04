import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression

a=pd.read_csv('data/train.csv')
#b=pd.read_csv('data/test.csv')

#convert String data into int data refer readme
a.fillna(0,inplace=True)
def convert(data,list1):
	result=[]
	for i in data:
		result.append(list1[i])
	return result

list1={'low fat':1, 'LF':1, 'Low Fat':1, 'reg':2, 'Regular':2}
list2={'Dairy':1, 'Breakfast':2, 'Canned':3, 'Fruits and Vegetables':4, 'Soft Drinks':5, 'Frozen Foods':6, 'Breads':7, 'Household':8, 'Meat':9, 'Baking Goods':10, 'Health and Hygiene':11, 'Starchy Foods':12, 'Seafood':13, 'Hard Drinks':14, 'Snack Foods':15, 'Others':16}
list3={'Small':1, 'Medium':2, 'High':3}
list4 ={'Tier 1':1, 'Tier 2':2, 'Tier 3':3}
list5={'Grocery Store':1, 'Supermarket Type1':2, 'Supermarket Type2':3, 'Supermarket Type3':4}
a["Item_Fat_Content"]=convert(a["Item_Fat_Content"],list1)
a["Item_Type"]=convert(a["Item_Type"],list2)
#a["Outlet_Size"]=convert(a["Outlet_Size"],list3)
a["Outlet_Location_Type"]=convert(a["Outlet_Location_Type"],list4)
a["Outlet_Type"]=convert(a["Outlet_Type"],list5)
#a["Item_Outlet_Sales"]=[int(i) for i in a["Item_Outlet_Sales"]]
#a["Item_MRP"]=[int(i) for i in a["Item_MRP"]]

'''
train_X=a[["Item_Weight","Item_Fat_Content","Item_Visibility","Item_Type","Item_MRP","Outlet_Establishment_Year","Outlet_Location_Type","Outlet_Type"]]
test_X=b[["Item_Weight","Item_Fat_Content","Item_Visibility","Item_Type","Item_MRP","Outlet_Establishment_Year","Outlet_Location_Type","Outlet_Type"]]
train_Y=a["Item_Outlet_Sales"]
train_X["Item_Fat_Content"]
'''
train_X,test_X,train_Y,test_Y=cross_validation.train_test_split(a[["Item_Fat_Content","Item_Visibility","Item_Type","Item_MRP","Outlet_Establishment_Year","Outlet_Location_Type","Outlet_Type"]],a["Item_Outlet_Sales"],test_size=0.20)

clf=LinearRegression(n_jobs=-1)
clf.fit(train_X,train_Y)
accuracy=clf.score(test_X,test_Y)
print(accuracy)