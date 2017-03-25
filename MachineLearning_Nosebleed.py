# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 14:52:08 2017

@author: Vikneshan

Predicting pollen count using machine learning

Reference:
    1. http://machinelearningmastery.com/machine-learning-in-python-step-by-step/
    2. http://stackoverflow.com/questions/17063458/reading-an-excel-file-in-python-using-pandas
    3. http://stackoverflow.com/questions/25985120/numpy-1-9-0-valueerror-probabilities-do-not-sum-to-1
    4. http://machinelearningmastery.com/basic-concepts-in-machine-learning/

"""
#importing modules
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#importing data from excel spreadsheet
xl=pd.ExcelFile("Nosebleed1.xls")
df=xl.parse("Data")

print(df.shape) # data size
print(df.head(20)) #preview first 20 lines.
print(df.describe()) #descriptions

#visualization of input data
df.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False) # box and whisker plots
plt.show()

#scatter plot matrix
scatter_matrix(df)
plt.show()

# Split-out validation dataset
array = df.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20 #using 20% of data for cross validation
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
scoring = 'accuracy'

# Build Models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model in turn
results = []
names = []
highest_val=0
highest_name=""

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    if highest_val <=cv_results.mean():
        highest_val=cv_results.mean()
        highest_name=name

print("Model with highest correlation: %s, with val: %f" % (highest_name,highest_val))

#Make Predictions (Automatically choose model with highest correlation score), Visualizate 20% of validation data 
if highest_name=='LR':
    best_model = LogisticRegression()
elif highest_name=='LDA':
    best_model = LinearDiscriminantAnalysis()
elif highest_name=='KNN':
    best_model = KNeighborsClassifier()
elif highest_name=='CART':
    best_model = DecisionTreeClassifier()
elif highest_name=='NB':
    best_model = GaussianNB()    
else:
    best_model=SVC()

best_model.fit(X_train, Y_train)
predictions = best_model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

fig2 = plt.figure()
plt.plot(range(0,len(predictions)),predictions,'ro',label=highest_name)
plt.plot(range(0,len(predictions)),Y_validation,'b+',label="Actual")         
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


#Make Predictions (Automatically choose model with highest correlation score) , Visualizate 80% of training data 
predictions = best_model.predict(X_train)
print(accuracy_score(Y_train, predictions))
print(confusion_matrix(Y_train, predictions))
print(classification_report(Y_train, predictions))

fig3 = plt.figure()
plt.plot(range(0,len(predictions)),predictions,'ro',label=highest_name)
plt.plot(range(0,len(predictions)),Y_train,'b+',label="Actual")         
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

#Make Predictions (Automatically choose model with highest correlation score) , Visualizate 100% of predictions with 80% of data for training
predictions = best_model.predict(X)
print(accuracy_score(Y, predictions))
print(confusion_matrix(Y, predictions))
print(classification_report(Y, predictions))

fig4 = plt.figure()
plt.plot(range(0,len(predictions)),predictions,'ro',label=highest_name)
plt.plot(range(0,len(predictions)),Y,'b+',label="Actual")         
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


     