# import pandas and numpy
import pandas as pd
import numpy as np
#from numpy import genfromtxt

#Import necessary modules for machine learning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#Read a csv file downloaded from kaggle
missing_values = ["n/a", "na", "--","unknown"]
dataset = pd.read_csv("cardio_train.csv",delimiter=';', na_values = missing_values) 
dataset.dropna()
print(dataset.shape)
print(dataset.head)
#IODO: CLEANING CODE GOES HERE!!!
dataset["gender"]=dataset["gender"].apply(lambda val: 0 if val==1 else 1)
print(dataset["gender"].value_counts())
#Create an array to store the values from the file
array = dataset.values #Assign values to dataset
x = array[:,1:12]
y = array[:,12]
x_train, x_test, y_train,y_test = train_test_split(x,y, test_size=0.20, random_state=1)
#Instantiate a k-NN classifier:knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classfier to the training dataset
knn.fit(x_train, y_train)

#Predict the labels of the test data:y_pred
y_pred = knn.predict(x_test)

#generate the confusion matrix and classification classification_report
print("******CONFUSION Matrix**")
cm=confusion_matrix(y_test,y_pred)
print(cm)
print("TRUE POSITIVES : %d" %(cm[0,0]))
print("TRUE NEGATIVES : %d" %(cm[1,1]))
print("FALSE POSITIVES : %d" %(cm[1,0]))
print("FALSE NEGATIVES : %d" %(cm[0,1]))



print("*************")

print("******CONFUSION Matrix**")
print(confusion_matrix(y_test,y_pred))
print("*************")
print("******ACCURACY SCORE**")
print(knn.score(x_test, y_test))
print("**************")

