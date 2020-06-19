

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import math
import pandas as pd
import csv
from numpy import loadtxt

import streamlit as st
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 

from scipy import stats
 

st.title("EtsyEdge")

#@st.cache
#def load_earrings(path):
#    return pd.read_csv(path)
#@st.cache
#def load_toys(path):
#    return pd.read_csv(path)
df_candles = st.cache(pd.read_csv)('streamlit_candles.csv') 
df_candles= df_candles.sort_values(by=['company','product'])
d=list(df_candles['betabestseller2'])
y_c=d[0]
df_earrings = st.cache(pd.read_csv)('cleaned_data_earrings_subsample.csv')
df_earrings = df_earrings.sort_values(by=['company','product'])
d=list(df_earrings['betabestseller2'])
y_e=d[0]
df_toys = st.cache(pd.read_csv)('streamlit_woodentoys.csv')
df_toys = df_toys.sort_values(by=['company','product'])
d=list(df_toys['betabestseller2'])
y_t=d[0]
#sns.set(rc={"axes.facecolor":"#ccddff",
#            "axes.grid":False,
#            'axes.labelsize':30,
#            'figure.figsize':(30.0, 10.0),
#           'xtick.labelsize':25,
#            'ytick.labelsize':20})
#sns.set(rc={'figure.figsize': (20.0,15.0)})


def main():
    st.sidebar.title('User inputs')
            
    category = st.sidebar.selectbox('Select a category', ('Candles', 'Earrings', 'Wooden name puzzle toys'))
    
    if category == 'Wooden name puzzle toys':
        df = df_toys
        df_saved = df
        X = df[["currentprice", "freeshipping2","foreign","etsysince2","dayssincelastreview","daysuntilship","numberofreviews2"]]
        y = df["bestseller2"]
        X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=10)
        lg = LogisticRegression(solver='lbfgs',max_iter=10000).fit(X_train,y_train)
        clf = RandomForestClassifier(max_depth = 8, random_state = 10).fit(X_train,y_train)
#Code snippet suggested by Streamlit team
        for column in df[['company','product']].columns:
            options = pd.Series([""]).append(df[column], ignore_index=False).unique()
            choice = st.sidebar.selectbox("Select a {}".format(column), options)
    
            if choice != "":
                df = df[df[column] == choice].drop(columns = column)
        dfnew = df
        #st.table(dfnew)
        
        
        
        if st.sidebar.checkbox('Days since last review'):
            review_cut = st.slider('Select a reduction in review time',0.0,1.0,0.1)
            st.write('You selected a', round(100*review_cut,2), 'percent reduction in review time')
            arg = dfnew[["currentprice", "freeshipping2","foreign","etsysince2","dayssincelastreview","daysuntilship","numberofreviews2"]]
            
            x_0=lg.predict_proba(arg)[:,1]
            dfnew[["dayssincelastreview"]] = dfnew[["dayssincelastreview"]]*(1-review_cut)
            arg = dfnew[["currentprice", "freeshipping2","foreign","etsysince2","dayssincelastreview","daysuntilship","numberofreviews2"]]
            x=lg.predict_proba(arg)[:,1]
            b=x-x_0
            b = b*y_t
            b=math.floor(b)
                   
           
            #b=math.floor(b)
            st.write('This reduction gives you a', x, 'chance of attaining the bestseller badge and an increase of', b, 'units of store-level sales!')
                    
            
            plt.ylim(0,200)
            sns.boxplot(y=df_saved['dayssincelastreview'],fliersize=0,width=0.5).set(ylabel=None)
            plt.hlines(dfnew['dayssincelastreview'],-0.35,0.35, color = 'r')
            plt.title("Days since last review")
            st.pyplot()
            
                        
            
            
        if st.sidebar.checkbox('Price'):
            price_cut = st.slider('Select a reduction in price',0.0,1.0,0.1)
            st.write('You selected a', round(100*price_cut,2), 'percent reduction in price')
            arg = dfnew[["currentprice", "freeshipping2","foreign","etsysince2","dayssincelastreview","daysuntilship","numberofreviews2"]]
            x_0=lg.predict_proba(arg)[:,1]
            dfnew[["currentprice"]] = dfnew[["currentprice"]]*(1-price_cut)
            arg = dfnew[["currentprice", "freeshipping2","foreign","etsysince2","dayssincelastreview","daysuntilship","numberofreviews2"]]
            x=lg.predict_proba(arg)[:,1]
            b=x-x_0
            b = b*y_t
            b=math.floor(b)
                   
              
            #b=math.floor(b)
            st.write('This reduction gives you a', x, 'chance of attaining the bestseller badge and an increase of', b, 'units of store-level sales!')
               
            plt.ylim(0,200)
            sns.boxplot(y=df_saved['currentprice'],fliersize=0,width=0.5).set(ylabel=None)
            plt.hlines(dfnew['currentprice'],-0.35,0.35, color = 'r')
            plt.title("Relative price")
            st.pyplot()
            
        
        if st.sidebar.checkbox('Shipping'):
            ship_cut = st.slider('Select a reduction in shipping time',0.0,1.0,0.1)
            st.write('You selected a', round(100*ship_cut,2), 'percent reduction in shipping time')
            arg = dfnew[["currentprice", "freeshipping2","foreign","etsysince2","dayssincelastreview","daysuntilship","numberofreviews2"]]
            x_0=lg.predict_proba(arg)[:,1]
           
            dfnew[["daysuntilship"]] = dfnew[["daysuntilship"]]*(1-ship_cut)
            arg = dfnew[["currentprice", "freeshipping2","foreign","etsysince2","dayssincelastreview","daysuntilship","numberofreviews2"]]
            x=lg.predict_proba(arg)[:,1]
            b=x-x_0
            b = b*y_t
            b=math.floor(b)
                   
              
            #b=math.floor(b)
            st.write('This reduction gives you a', x, 'chance of attaining the bestseller badge and an increase of', b, 'units of store-level sales!')
            
            plt.ylim(0,30)
            sns.boxplot(y=df_saved['daysuntilship'],fliersize=0,width=0.5).set(ylabel=None)
            plt.hlines(dfnew['daysuntilship'],-0.35,0.35, color = 'r')
            plt.title("Days until ship")
            st.pyplot()
             
            
    
    elif category == 'Earrings':
        df = df_earrings
        df_saved = df
        X = df[["currentprice", "freeshipping2","foreign","etsysince2","dayssincelastreview","daysuntilship","numberofreviews2"]]
        y = df["bestseller2"]
        X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=10)
        lg = LogisticRegression(max_iter=1000).fit(X_train,y_train)
        clf = RandomForestClassifier(max_depth = 8, random_state = 10).fit(X_train,y_train)

        for column in df[['company','product']].columns:
            options = pd.Series([""]).append(df[column], ignore_index=False).unique()
            choice = st.sidebar.selectbox("Select a {}".format(column), options)
    
            if choice != "":
                df = df[df[column] == choice].drop(columns = column)
        dfnew = df
        #st.table(dfnew)
        
        
        
        if st.sidebar.checkbox('Days since last review'):
            review_cut = st.slider('Select a reduction in review time',0.0,1.0,0.1)
            st.write('You selected a', round(100*review_cut,2), 'percent reduction in review time')
            arg = dfnew[["currentprice", "freeshipping2","foreign","etsysince2","dayssincelastreview","daysuntilship","numberofreviews2"]]
            x_0=lg.predict_proba(arg)[:,1]
           
            dfnew[["dayssincelastreview"]] = dfnew[["dayssincelastreview"]]*(1-review_cut)
            arg = dfnew[["currentprice", "freeshipping2","foreign","etsysince2","dayssincelastreview","daysuntilship","numberofreviews2"]]
            x=lg.predict_proba(arg)[:,1]
            b=x-x_0
            b = b*y_e
            b=math.floor(b)
                   
            #b=math.floor(b)
            st.write('This reduction gives you a', x, 'chance of attaining the bestseller badge and an increase of', b, 'units of store-level sales!')
               
            
            plt.ylim(0,300)
            sns.boxplot(y=df_saved['dayssincelastreview'],fliersize=0,width=0.5).set(ylabel=None)
            plt.hlines(dfnew['dayssincelastreview'],-0.35,0.35, color = 'r')
            plt.title("Days since last review")
            st.pyplot()
            
                        
            
            
        if st.sidebar.checkbox('Price'):
            price_cut = st.slider('Select a reduction in price',0.0,1.0,0.1)
            st.write('You selected a', round(100*price_cut,2), 'percent reduction in price')
            arg = dfnew[["currentprice", "freeshipping2","foreign","etsysince2","dayssincelastreview","daysuntilship","numberofreviews2"]]
            x_0=lg.predict_proba(arg)[:,1]
           
            dfnew[["currentprice"]] = dfnew[["currentprice"]]*(1-price_cut)
            arg = dfnew[["currentprice", "freeshipping2","foreign","etsysince2","dayssincelastreview","daysuntilship","numberofreviews2"]]
            x=lg.predict_proba(arg)[:,1]
            b=x-x_0
            b = b*y_e
            b=math.floor(b)
                   
           
            #b=math.floor(b)
            st.write('This reduction gives you a', x, 'chance of attaining the bestseller badge and an increase of', b, 'units of store-level sales!')
            
            plt.ylim(0,100)
            sns.boxplot(y=df_saved['currentprice'],fliersize=0,width=0.5).set(ylabel=None)
            plt.hlines(dfnew['currentprice'],-0.35,0.35, color = 'r')
            plt.title("Relative price")
            st.pyplot()
            
        
        if st.sidebar.checkbox('Shipping'):
            ship_cut = st.slider('Select a reduction in shipping time',0.0,1.0,0.1)
            st.write('You selected a', round(100*ship_cut,2), 'percent reduction in shipping time')
            arg = dfnew[["currentprice", "freeshipping2","foreign","etsysince2","dayssincelastreview","daysuntilship","numberofreviews2"]]
            x_0=lg.predict_proba(arg)[:,1]
           
            dfnew[["daysuntilship"]] = dfnew[["daysuntilship"]]*(1-ship_cut)
            arg = dfnew[["currentprice", "freeshipping2","foreign","etsysince2","dayssincelastreview","daysuntilship","numberofreviews2"]]
            x=lg.predict_proba(arg)[:,1]
            b=x-x_0
            b = b*y_e
            b=math.floor(b)
                   
           
            #b=math.floor(b)
            st.write('This reduction gives you a', x, 'chance of attaining the bestseller badge and an increase of', b, 'units of store-level sales!')
                  
            plt.ylim(0,30)
            sns.boxplot(y=df_saved['daysuntilship'],fliersize=0,width=0.5).set(ylabel=None)
            plt.hlines(dfnew['daysuntilship'],-0.35,0.35, color = 'r')
            plt.title("Days until ship")
            st.pyplot()
     

    elif category == 'Candles':
            df = df_candles
            df_saved = df
            X = df[["currentprice", "freeshipping2","foreign","etsysince2","dayssincelastreview","daysuntilship","numberofreviews2"]]
            y = df["bestseller2"]
            X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=10)
            lg = LogisticRegression(max_iter=1000).fit(X_train,y_train)
            clf = RandomForestClassifier(max_depth = 8, random_state = 10).fit(X_train,y_train)
    
            for column in df[['company','product']].columns:
                options = pd.Series([""]).append(df[column], ignore_index=False).unique()
                choice = st.sidebar.selectbox("Select a {}".format(column), options)
        
                if choice != "":
                    df = df[df[column] == choice].drop(columns = column)
            dfnew = df
            #st.table(dfnew)
            
            
            
            if st.sidebar.checkbox('Days since last review'):
                review_cut = st.slider('Select a reduction in review time',0.0,1.0,0.1)
                st.write('You selected a', round(100*review_cut,2), 'percent reduction in review time')
                arg = dfnew[["currentprice", "freeshipping2","foreign","etsysince2","dayssincelastreview","daysuntilship","numberofreviews2"]]
                x_0=lg.predict_proba(arg)[:,1]
                dfnew[["dayssincelastreview"]] = dfnew[["dayssincelastreview"]]*(1-review_cut)
                arg = dfnew[["currentprice", "freeshipping2","foreign","etsysince2","dayssincelastreview","daysuntilship","numberofreviews2"]]
                x=lg.predict_proba(arg)[:,1]
                b=x-x_0
                b = b*y_c
                b=math.floor(b)
                   
           
                #b=math.floor(b)
                st.write('This reduction gives you a', x, 'chance of attaining the bestseller badge and an increase of', b, 'units of store-level sales!')
            
                
                plt.ylim(0,300)
                sns.boxplot(y=df_saved['dayssincelastreview'],fliersize=0,width=0.5).set(ylabel=None)
                plt.hlines(dfnew['dayssincelastreview'],-0.35,0.35, color = 'r')
                plt.title("Days since last review")
                st.pyplot()
                
                            
                
                
            if st.sidebar.checkbox('Price'):
                price_cut = st.slider('Select a reduction in price',0.0,1.0,0.1)
                st.write('You selected a', round(100*price_cut,2), 'percent reduction in price')
                arg = dfnew[["currentprice", "freeshipping2","foreign","etsysince2","dayssincelastreview","daysuntilship","numberofreviews2"]]
                x_0=lg.predict_proba(arg)[:,1]
              
                dfnew[["currentprice"]] = dfnew[["currentprice"]]*(1-price_cut)
                arg = dfnew[["currentprice", "freeshipping2","foreign","etsysince2","dayssincelastreview","daysuntilship","numberofreviews2"]]
                x=lg.predict_proba(arg)[:,1]
                b=x-x_0
                b = b*y_c
                b=math.floor(b)
                   
           
                #b=math.floor(b)
                st.write('This reduction gives you a', x, 'chance of attaining the bestseller badge and an increase of', b, 'units of store-level sales!')
              
                plt.ylim(0,100)
                sns.boxplot(y=df_saved['currentprice'],fliersize=0,width=0.5).set(ylabel=None)
                plt.hlines(dfnew['currentprice'],-0.35,0.35, color = 'r')
                plt.title("Relative price")
                st.pyplot()
                
            
            if st.sidebar.checkbox('Shipping'):
                
                ship_cut = st.slider('Select a reduction in shipping time',0.0,1.0,0.1)
                st.write('You selected a', round(100*ship_cut,2), 'percent reduction in shipping time')
                arg = dfnew[["currentprice", "freeshipping2","foreign","etsysince2","dayssincelastreview","daysuntilship","numberofreviews2"]]
                x_0=lg.predict_proba(arg)[:,1]
                
                dfnew[["daysuntilship"]] = dfnew[["daysuntilship"]]*(1-ship_cut)
                arg = dfnew[["currentprice", "freeshipping2","foreign","etsysince2","dayssincelastreview","daysuntilship","numberofreviews2"]]
                x= lg.predict_proba(arg)[:,1]
                b= x - x_0
                b = b*y_c
                b=math.floor(b)
                #b=math.floor(b)
                st.write('This reduction gives you a', x, 'chance of attaining the bestseller badge and an increase of', b, 'units of store-level sales!')
            
                plt.ylim(0,30)
                sns.boxplot(y=df_saved['daysuntilship'],fliersize=0,width=0.5).set(ylabel=None)
                plt.hlines(dfnew['daysuntilship'],-0.35,0.35, color = 'r')
                plt.title("Days until ship")
                st.pyplot()

if __name__ == "__main__":
    main()
