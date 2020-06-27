from bs4 import BeautifulSoup
from time import sleep
import pandas as pd
import requests
import urllib.request
import csv
import time
import re
import os
from numpy import loadtxt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier 
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier 
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from scipy import stats
from sklearn.metrics import roc_auc_score


#Script for logistic, random forest, dummmy classifier and prediction scores 

#df = pd.read_csv('streamlit_woodentoys.csv')
df = pd.read_csv('cleaned_data_earrings_subsample.csv')
#df = pd.read_csv('streamlit_candles.csv')

#define test and training data
X = df[["currentprice", "freeshipping2","foreign","etsysince2","dayssincelastreview","daysuntilship","numberofreviews2"]]
#X_2079 = df[["currentprice", "freeshipping2","foreign","numberofreviews2"]]
y = df["bestseller2"]
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=10,shuffle=True)

clf1 = DummyClassifier(strategy="most_frequent")
clf2 = LogisticRegression(max_iter=1000)
clf3 = RandomForestClassifier(max_depth = 8, random_state=10)


# predict class probabilities for all classifiers
probas = [c.fit(X_train, y_train).score(X_train,y_train) for c in (clf1, clf2, clf3)]

print(probas)
# get class probabilities
class1 = [pr for pr in probas]
complement = [1-pr for pr in probas]


# plotting
N = 3  # number of groups
ind = np.arange(N)  # group positions
width = 0.20  # bar width

fig, ax = plt.subplots()
p1 = ax.bar(ind, class1, width, label = 'Non-bestseller')
p2 = ax.bar(ind + width, complement, width, label = 'Bestseller')

#graph labels.
ax.set_xticks(ind+.15)
ax.set_xticklabels(['Dummy','Logistic',
                    'Random Forest'],
                   rotation=0,
                   ha='right')
plt.ylim([0, 1])
#plt.title('Prediction score for earring sellers by classifier')
#plt.title('Prediction score for candle sellers by classifier')
plt.title('Prediction score for toy sellers by classifier')
plt.legend(bbox_to_anchor=(1.01, .8),loc = 'center left', borderaxespad=0.)
plt.tight_layout()
plt.show()



#Script to generate ROC curves
df_earrings = pd.read_csv('cleaned_data_earrings_subsample.csv')
X = df_earrings[["currentprice", "freeshipping2","foreign","etsysince2","dayssincelastreview","daysuntilship","numberofreviews2"]]
y = df_earrings["bestseller2"]
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=10)
lg = LogisticRegression(max_iter=10000).fit(X_train,y_train)
clf = RandomForestClassifier(max_depth = 8, random_state = 10).fit(X_train,y_train)

df_candles = pd.read_csv('streamlit_candles.csv') 
X = df_candles[["currentprice", "freeshipping2","foreign","etsysince2","dayssincelastreview","daysuntilship","numberofreviews2"]]
y = df_candles["bestseller2"]
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=10)
lg = LogisticRegression(max_iter=10000).fit(X_train,y_train)
clf = RandomForestClassifier(max_depth = 8, random_state = 10).fit(X_train,y_train)

df_toys = pd.read_csv('streamlit_woodentoys.csv')
X = df_toys[["currentprice", "freeshipping2","foreign","etsysince2","dayssincelastreview","daysuntilship","numberofreviews2"]]
y = df_toys["bestseller2"]
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=10)
lg = LogisticRegression(max_iter=10000).fit(X_train,y_train)
clf = RandomForestClassifier(max_depth = 8, random_state = 10).fit(X_train,y_train)


# method I: logistic plt
probs = lg.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)


plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', color = 'blue', label = 'Logistic ROC (area = %0.2f)' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


 
# method I: random forest plt
probs = clf.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)



plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', color = 'green', label = 'Random forest ROC (area = %0.2f)' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


#OLS regression

X=df_earrings[["currentprice", "freeshipping2","foreign","etsysince2","dayssincelastreview","daysuntilship","numberofreviews2"]]
#X=df_candles[["currentprice", "freeshipping2","foreign","etsysince2","dayssincelastreview","daysuntilship","numberofreviews2"]]
#X=df_toys[["currentprice", "freeshipping2","foreign","etsysince2","dayssincelastreview","daysuntilship","numberofreviews2"]]
y = df_earrings['bestseller2']
#y = df_candles['bestseller2']
#y = df_toys['bestseller2']
lm = linear_model.LinearRegression()
model = lm.fit(X,y)
model_prediction = lm.predict(X)
print("Coefficients: \n", lm.coef_)
print("Mean squared error: %.2f" 
     %mean_squared_error(y, model_prediction))
print("Coefficient of determination: %.2f"
      %r2_score(y, model_prediction))

#T-tests corresponding to box plots

df_no = df_earrings[df_earrings['bestseller2'] != 1]
df_yes = df_earrings[df_earrings['bestseller2'] == 1] 
list = ['sales','currentprice','numberofreviews','dayssincelastreview']
ttest = []
for x in list:
    ttest.append(stats.ttest_ind(df_no[x],df_yes[x]))
print(ttest)












