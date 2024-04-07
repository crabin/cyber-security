from warnings import simplefilter

import numpy as np
import pandas as pd
import sklearn
from numpy import genfromtxt
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

simplefilter(action='ignore', category=FutureWarning)
###############################################################


feature=genfromtxt('label_feature_IOT.csv',delimiter=',',usecols=(i for i in range(1,19)),dtype=int,skip_header=1)
target=genfromtxt('label_feature_IOT.csv',delimiter=',',usecols=(0),dtype=str,skip_header=1)
for c in range(-5,0):
    for i in range(len(feature[:,c])):
        feature[:,c][i] = int(str(feature[:,c][i]),16)
labels = LabelEncoder().fit_transform(target)
feature_std = StandardScaler().fit_transform(feature)
x_train, x_test, y_train, y_test = train_test_split(feature_std, labels, test_size=0.25, random_state=0)

print("Begin:__________________________________")
###################################################
## print stats

def print_stats_metrics(y_test, y_pred):
    print('Accuracy: %.2f' % accuracy_score(y_test,y_pred) )
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print ("confusion matrix")
    print(confmat)
    print (pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred,average='weighted'))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred,average='weighted'))
    print('F1-measure: %.3f' % f1_score(y_true=y_test, y_pred=y_pred,average='weighted'))


########################Logistic Regression##############################
print("########################Logistic Regression##############################")
clfLog = LogisticRegression()
clfLog.fit(x_train,y_train)
predictions = clfLog.predict(x_test)
print_stats_metrics(y_test, predictions)

########################Random Forest##############################
print("########################Random Forest##############################")
clfRandForest = RandomForestClassifier()
clfRandForest.fit(x_train,y_train)
predictions = clfRandForest.predict(x_test)
print_stats_metrics(y_test, predictions)
#######################Decision Tree#######################
print("#######################Decision Tree#######################")
clfDT = DecisionTreeRegressor()
clfDT.fit(x_train,y_train)
predictions = clfDT.predict(x_test)
print_stats_metrics(y_test, predictions)
#######################Naive Bayes#######################
print("#######################Naive Bayes#######################")
clfNB = GaussianNB()
clfNB.fit(x_train,y_train)
predictions = clfNB.predict(x_test)
print_stats_metrics(y_test, predictions)

