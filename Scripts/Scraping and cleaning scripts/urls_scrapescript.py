
from bs4 import BeautifulSoup
from time import sleep
import pandas as pd
import requests
import urllib.request
import csv
import json


# url_list contains all of the individual products
# Works by iterating through all 44 product pages and populating url_list with 
# urls.  url_list will contain 2054 entries
url_list = []
page_list = ['https://www.etsy.com/c/toys-and-games/toys/baby-and-toddler-toys?ref=catnav-11049']
for i in range(2, 250):
    page_url = 'https://www.etsy.com/c/toys-and-games/toys/baby-and-toddler-toys?ref=pagination&page={}'.format(i)
    page_list.append(page_url)
    
for page_url in page_list:
    page = requests.get(page_url)
    soup = BeautifulSoup(page.content, 'html.parser')

    page = soup.find_all(class_="responsive-listing-grid wt-grid wt-grid--block justify-content-flex-start pl-xs-0")[0]
    page_elements = page.find_all('li')
    for elem in page_elements:
        for a in elem.find_all('a', href=True):
            url_list.append(a['href'])

#print(url_list)
#type(url_list)

#Example
#st = '12,12'
#s = st.replace(',','\r\n')
#print(s)


length=len(url_list)
print(length)

s = str(url_list)
snew = s.replace(',','\r\n')
print(snew)
f = open('BASE_URL_babytoys.txt', 'w+')
json.dump(snew, f)
f.close()

