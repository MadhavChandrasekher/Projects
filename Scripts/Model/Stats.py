
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from numpy import loadtxt

#import streamlit as st
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy import stats
from pandas import DataFrame

from sklearn.ensemble import RandomForestClassifier 
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
from numpy import where
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline



#CANDLES
data = pd.read_csv('streamlit_candles.csv')
list(data.columns)
data=data.rename(columns={'bestseller2':'bestseller', 'sales2':'sales','numberofreviews2':'numberofreviews'})
data.head()
X = data[["bestseller","sales","currentprice", "freeshipping2","foreign","etsysince2","dayssincelastreview","daysuntilship","numberofreviews"]]
X.head()


plt.ylim(0,35000)
boxplot1 = sns.boxplot(x = X["bestseller"], y = X["sales"], data = X, fliersize=0)
plt.suptitle('CANDLES',fontsize=24, y=1.05)
plt.title('Sales by Bestseller Badge',fontsize=16)

plt.ylim(0,42)
boxplot1 = sns.boxplot(x = X["bestseller"], y = X["currentprice"], data = X, fliersize=0)
plt.suptitle('CANDLES',fontsize=24, y=1.05)
plt.title('Price by Bestseller Badge',fontsize=16)

plt.ylim(0,6200)
boxplot1 = sns.boxplot(x = X["bestseller"], y = X["numberofreviews"], data = X, fliersize=0)
plt.suptitle('CANDLES',fontsize=24, y=1.05)
plt.title('Number of Reviews by Bestseller Badge',fontsize=16)

plt.ylim(0,450)
boxplot1 = sns.boxplot(x = X["bestseller"], y = X["dayssincelastreview"], data = X, fliersize=0)
plt.suptitle('CANDLES',fontsize=24, y=1.05)
plt.title('Days Since Last Review by Bestseller Badge',fontsize=16)

plt.ylim(0,42)
boxplot1 = sns.boxplot(x = X["bestseller"], y = X["currentprice"], data = X, fliersize=0)
plt.suptitle('CANDLES',fontsize=24, y=1.05)
plt.title('Price by Bestseller Badge',fontsize=16)


#EARRINGS
data = pd.read_csv('cleaned_data_earrings_subsample.csv')
list(data.columns)
data=data.rename(columns={'bestseller2':'bestseller', 'sales2':'sales','numberofreviews2':'numberofreviews'})
data.head()
X = data[["bestseller","sales","currentprice", "freeshipping2","foreign","etsysince2","dayssincelastreview","daysuntilship","numberofreviews"]]
X.head()

plt.ylim(0,120000)
boxplot1 = sns.boxplot(x = X["bestseller"], y = X["sales"], data = X, fliersize=0)
plt.suptitle('EARRINGS',fontsize=24, y=1.05)
plt.title('Sales by Bestseller Badge',fontsize=16)



plt.ylim(0,70)
boxplot1 = sns.boxplot(x = X["bestseller"], y = X["currentprice"], data = X, fliersize=0)
plt.suptitle('EARRINGS',fontsize=24, y=1.05)
plt.title('Price by Bestseller Badge',fontsize=16)

plt.ylim(0,20000)
boxplot1 = sns.boxplot(x = X["bestseller"], y = X["numberofreviews"], data = X, fliersize=0)
plt.suptitle('EARRINGS',fontsize=24, y=1.05)
plt.title('Number of Reviews by Bestseller Badge',fontsize=16)

plt.ylim(0,800)
boxplot1 = sns.boxplot(x = X["bestseller"], y = X["dayssincelastreview"], data = X, fliersize=0)
plt.suptitle('EARRINGS',fontsize=24, y=1.05)
plt.title('Days Since Last Review by Bestseller Badge',fontsize=16)

plt.ylim(0,42)
boxplot1 = sns.boxplot(x = X["bestseller"], y = X["currentprice"], data = X, fliersize=0)
plt.suptitle('EARRINGS',fontsize=24, y=1.05)
plt.title('Price by Bestseller Badge',fontsize=16)


#WOODEN NAMED PUZZLE TOYS
data = pd.read_csv('streamlit_woodentoys.csv')
list(data.columns)
data=data.rename(columns={'bestseller2':'bestseller', 'sales2':'sales','numberofreviews2':'numberofreviews'})
data.head()
X = data[["bestseller","sales","currentprice", "freeshipping2","foreign","etsysince2","dayssincelastreview","daysuntilship","numberofreviews"]]
X.head()

plt.ylim(0,80000)
boxplot1 = sns.boxplot(x = X["bestseller"], y = X["sales"], data = X, fliersize=0)
plt.suptitle('WOODEN NAME PUZZLE TOYS',fontsize=24, y=1.05)
plt.title('Sales by Bestseller Badge',fontsize=16)

plt.ylim(0,70)
boxplot1 = sns.boxplot(x = X["bestseller"], y = X["currentprice"], data = X, fliersize=0)
plt.suptitle('WOODEN NAME PUZZLE TOYS',fontsize=24, y=1.05)
plt.title('Price by Bestseller Badge',fontsize=16)

plt.ylim(0,12000)
boxplot1 = sns.boxplot(x = X["bestseller"], y = X["numberofreviews"], data = X, fliersize=0)
plt.suptitle('WOODEN NAME PUZZLE TOYS',fontsize=24, y=1.05)
plt.title('Number of Reviews by Bestseller Badge',fontsize=16)

plt.ylim(0,300)
boxplot1 = sns.boxplot(x = X["bestseller"], y = X["dayssincelastreview"], data = X, fliersize=0)
plt.suptitle('WOODEN NAME PUZZLE TOYS',fontsize=24, y=1.05)
plt.title('Days Since Last Review by Bestseller Badge',fontsize=16)

plt.ylim(0,42)
boxplot1 = sns.boxplot(x = X["bestseller"], y = X["currentprice"], data = X, fliersize=0)
plt.suptitle('WOODEN NAME PUZZLE TOYS',fontsize=24, y=1.05)
plt.title('Price by Bestseller Badge',fontsize=16)



#DEALING WITH IMBALANCED SUB-SAMPLE OF JUST-BESTSELLERS and JUST-NON-BESTSELLERS

# collapse dataset on z_scores of price and sales
df_earrings = pd.read_csv('streamlit_earrings_almostbest.csv')
df = df_earrings
X = df[["z_currentprice","z_sales2","bestseller2"]].drop_duplicates()
y = X['bestseller2']
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=10,shuffle=True)


 
# plot imbalanced data
sns.lmplot( x="z_sales2", y="z_currentprice", data=X, fit_reg=False, hue='bestseller2', legend=False)
plt.xlabel("z-score sales")
plt.ylabel("z-score price")
# Move the legend to an empty part of the plot
plt.legend(loc='lower right')

        


# balance sample and summarize class distribution
counter = Counter(y)
print(counter)

##both oversample just bestsellers and undersampling just non-bestsellers
#over = SMOTE(sampling_strategy=0.1)
#under = RandomUnderSampler(sampling_strategy=0.5)
#steps = [('o', over), ('u', under)]
#pipeline = Pipeline(steps=steps)
#X, y = pipeline.fit_resample(X, y)
#counter = Counter(y)
#print(counter)

##only oversampling bestsellers
oversample = SMOTE()

X_train, y_train = oversample.fit_resample(X_train, y_train)
# summarize the new class distribution
counter = Counter(y)
print(counter)
 
#plot oversampled data
sns.lmplot( x="z_sales2", y="z_currentprice", data=X_train, fit_reg=False, hue='bestseller2', legend=False)
 
#move legend
plt.legend(loc='lower right')



