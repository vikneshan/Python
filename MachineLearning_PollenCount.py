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
xl=pd.ExcelFile("PollenCount_Data.xls")
df=xl.parse("PollenData")

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
seed = 14
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

## Compare Algorithms using a boxplot
#fig = plt.figure()
#fig.suptitle('Algorithm Comparison')
#ax = fig.add_subplot(111)
#plt.boxplot(results)
#ax.set_xticklabels(names)
#plt.show()

##Make Predictions (Automatically choose model with highest correlation score)
#if highest_name=='LR':
#    best_model = LogisticRegression()
#elif highest_name=='LDA':
#    best_model = LinearDiscriminantAnalysis()
#elif highest_name=='KNN':
#    best_model = KNeighborsClassifier()
#elif highest_name=='CART':
#    best_model = DecisionTreeClassifier()
#elif highest_name=='NB':
#    best_model = GaussianNB()    
#else:
#    best_model=SVC()
#
#best_model.fit(X_train, Y_train)
#predictions = best_model.predict(X_validation)
#print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))
#
#fig2 = plt.figure()
#plt.plot(range(0,len(predictions)),predictions,'r',label="Model")
#plt.plot(range(0,len(predictions)),Y_validation,'b',label="Actual")         
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

#Make Predictions (Run all models)
LR = LogisticRegression()
LDA = LinearDiscriminantAnalysis()
KNN = KNeighborsClassifier()
CART = DecisionTreeClassifier()
NB = GaussianNB()    
SVC=SVC()

LR.fit(X_train, Y_train)
predictions_LR = LR.predict(X_validation)

LDA.fit(X_train, Y_train)
predictions_LDA = LDA.predict(X_validation)

KNN.fit(X_train, Y_train)
predictions_KNN = KNN.predict(X_validation)

CART.fit(X_train, Y_train)
predictions_CART = CART.predict(X_validation)

NB.fit(X_train, Y_train)
predictions_NB = NB.predict(X_validation)

SVC.fit(X_train, Y_train)
predictions_SVC = SVC.predict(X_validation)


fig2 = plt.figure()
plt.plot(range(0,len(predictions_LR)),predictions_LR,'r:',label="LR")
plt.plot(range(0,len(predictions_LDA)),predictions_LDA,'g:',label="LDA")
plt.plot(range(0,len(predictions_KNN)),predictions_LDA,'c:',label="KNN")
plt.plot(range(0,len(predictions_CART)),predictions_CART,'m:',label="CART")
plt.plot(range(0,len(predictions_NB)),predictions_NB,'y:',label="NB")
plt.plot(range(0,len(predictions_SVC)),predictions_SVC,'k:',label="SVC")
plt.plot(range(0,len(Y_validation)),Y_validation,'b',label="Actual")        
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel('Pollen Count (grains/m^3 air)')
plt.xlabel('Data Count')

     