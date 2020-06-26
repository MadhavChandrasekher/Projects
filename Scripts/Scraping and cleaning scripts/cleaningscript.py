from bs4 import BeautifulSoup
from time import sleep
import pandas as pd
import requests
import urllib.request
import csv
import time
import re
import os
import numpy as np
from numpy import loadtxt

#df= pd.read_csv('C:/Users/mccha/Downloads/Record 1 to 2080 minus 4 that dont work.csv')
woodentoys = pd.read_csv('etsy_woodentoys_ALL.csv')
woodentoys=woodentoys.drop('12',axis=1)
woodentoys.columns
woodentoys=woodentoys.rename(columns={'0': 'Number_of_reviews','1': 'Number_of_stars','2': 'Average_rating',
                           '3': 'Product','4': 'Sales','5': 'Company','6': 'Location','7': 'Etsy_since',
                           '8': 'Time_to_Ship','9': 'Recent_review','10':'Handmade',
                           '11': 'Price','13':'Raves','14':'Free_shipping','15':'Bestseller'})
woodentoys=woodentoys.drop('Handmade',axis=1)
df=woodentoys

#FORMAT NUMBER OF REVIEWS
df['Item unavailable'] = np.nan
df['Number_of_reviews'] = df['Number_of_reviews'].where(df['Number_of_reviews'] != 'Item Unavailable', df['Item unanvailable'])
df_list = df['Number_of_reviews'].tolist()
len(df_list) 
df_new = []
for i in range(0,2079):
    x_i = re.sub('[^0-9]','',df_list[i])    
    df_new.append(x_i)
df_new=pd.DataFrame(df_new)


df_new.head()
df['Number_of_reviews'] = df_new
df['Zeroes'] = 0
df['Zeros'] = df['Zeroes'].apply(str)
list(df['Zeroes'])
df['Number_of_reviews']= df['Number_of_reviews'].where(df['Number_of_reviews'] != '', df['Zeros'])
print(df['Number_of_reviews'])    
#FORMAT NUMBER OF STARS
df['Ones'] = 1
df['Twos'] = 2
df['Three'] = 3
df['Four'] = 4
df['Four mid'] = 4.5
df['Five'] = 5
df['Never rated'] = np.nan
df['Number_of_stars'] = df['Number_of_stars'].where(df['Number_of_stars'] != '0 out of 5 stars'
                                                    , df['Zeroes'])
df['Number_of_stars'] = df['Number_of_stars'].where(df['Number_of_stars'] != '1 out of 5 stars'
                                                    , df['Ones'])
df['Number_of_stars'] = df['Number_of_stars'].where(df['Number_of_stars'] != '2 out of 5 stars'
                                                    , df['Twos'])
df['Number_of_stars'] = df['Number_of_stars'].where(df['Number_of_stars'] != '3 out of 5 stars'
                                                    , df['Three'])
df['Number_of_stars'] = df['Number_of_stars'].where(df['Number_of_stars'] != '4 out of 5 stars'
                                                    , df['Four'])
df['Number_of_stars'] = df['Number_of_stars'].where(df['Number_of_stars'] != '4.5 out of 5 stars'
                                                    , df['Four mid'])
df['Number_of_stars'] = df['Number_of_stars'].where(df['Number_of_stars'] != '5 out of 5 stars'
                                                    , df['Five'])
df['Number_of_stars'] = df['Number_of_stars'].where(df['Number_of_stars'] != 'NR'
                                                    , df['Never rated'])
#FORMAT AVERAGE RATING
df['Average_rating'] = df['Average_rating'].where(df['Average_rating'] != 'No Values'
                                                  , df['Never rated'])
#FORMAT SALES
df['No Sales'] = np.nan
df['Sales'] = df['Sales'].where(df['Sales]') != 'No Sales', df['No Sales'])
df_sales = df['Sales'].tolist()
df_new = []
for i in range(0,2079):
    x_i = re.sub('[^0-9]','',df_sales[i])    
    df_new.append(x_i)
df_new=pd.DataFrame(df_new)
df['Sales'] = df_new


#FORMAT LOCATION
df['No Location'] = 'No Location'
df['Location'] = df['Location'].where(df['Location'] != 'No Location', df['No Location'])
df['United_States'] = 'United States'
df_location = df['Location'].tolist()
df_new = []
for i in range(0,2079): 
    if re.match(',',df_location[i]) is not None: 
        x_i = 'United States'                                      
    else:
        x_i = df_location[i] 
    df_new.append(x_i)
df['Location'] = df_new 
df['Location'].head(100)                                
df.drop(['Zeroes', 'Ones', 'Twos', 'Three', 'Four', 'Four mid', 'Five'
         , 'Never rated'], axis=1 )                                      
                                      
#FORMAT US INDICATOR
df['United States'] = 'United States'
df['US_indicator'] = df['Location'].where(df['Location'] != 'United States', df['United States'])
df.drop(df['United States'], axis = 1)

#FORMAT STORE HISTORY
df['Empty'] = ''
df['Etsy_since'] = df['Etsy_since'].where(df['Etsy_since'] != 'No historical information'
                                          , df['Empty'])
df.drop(['Empty'],axis=1)


#FORMAT TIME_TO_SHIP
df_timetoship = df['Time_to_Ship'].tolist() 
df_new = []
for i in range(0,2079):
    x = df_timetoship[i]
    #print(x)
    y = re.search('weeks',x)
    c = re.findall('[0-9]+\S[0-9]+',x)
    if len(c) == 0:
        
        d = re.findall('[0-9]+',x)
        if len(d) == 0:
            d = 'No shipping info'
        else:
            d = float(d[0])
    else:
        a = re.findall('([0-9]+)\S[0-9]+',x)
        a = float(a[0])
        b = re.findall('[0-9]+\S([0-9]+)',x)
        b = float(b[0])
        d = b-a/2   
    if y is None:
       z = d
    else:
        z = d*7
    df_new.append(z)
df_new=pd.DataFrame(df_new)
df['Time_to_Ship']=df_new


#FORMAT PRICE
df_pricelist = df['Price'].tolist() 
df_new = []
print(df_pricelist[1])
for i in range(0,2079):
    x = re.findall('^Price:\S([0-9.]+).*',df_pricelist[i])  
    if len(x) == 0:
        x = re.findall('.([0-9.]+).+',df_pricelist[i])
    df_new.append(x)
df_new=pd.DataFrame(df_new)
df_new.head()


