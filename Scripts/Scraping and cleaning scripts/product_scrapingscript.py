# -*- coding: utf-8 -*-

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


#Load empty array for product info
websites = []
#Loop through our URLS loaded above
with open('BASE_URL_babytoys.txt','r', encoding="utf8") as links:
    for b in links:
   
        page = requests.get(b)
        soup = BeautifulSoup(page.content, 'html.parser')
        intro_block = soup.find('div',class_ = "wt-display-flex-xs wt-align-items-center")
        intro_block2=intro_block.find('p',class_="wt-text-body-01")
        #print(intro_block2)


        if intro_block2!=None:
            reviews="Item Unavailable"
            stars="Item Unavailable"
            values="Item Unavailable"
            product="Item Unavailable"                
            sales="Item Unavailable"
            brand="Item Unavailable"
            country="Item Unavailable"
            etsysince="Item Unavailable"
            shipping="Item Unavailable"
            mostrecentreview="Item Unavailable"
            handmade="Item Unavailable"
            price="Item Unavailable"
            domesticfreeshipping="Item Unavailable"
            buyersareraving="Item Unavailable"
            freeshipping="Item Unavailable"
            bestseller="Item Unavailable"
            
            all=[reviews,stars,values,product,sales,brand,country,etsysince,shipping,mostrecentreview,handmade,price,domesticfreeshipping,#itemreview,
             buyersareraving,freeshipping, bestseller]
            print(all)
            websites.append(all)
        
        
        else:
            #Stars, Reviews, Values
            stars, reviews, values = None, None, None
            start = time.time()
            while values is None and time.time() - start < 60:
                page = requests.get(b)
                soup = BeautifulSoup(page.content, 'html.parser')
                full_block = soup.find_all('h3',class_ = "wt-text-body-03")
                if len(full_block) == 0:
                    stars= None
                    reviews= None
                    values= None
    
                else:
                    text1 = full_block[0].get_text()
                    text1=text1.split('\n')
                    #print(text1)
                    text2=[x for x in text1 if x != '']
                    text = full_block[0].get_text().split("\n")
                    reviews=text2[0]
                    stars=text2[1]
                    #reviews = text[0]
                    #stars = text[3]
                    hiddenInputs = full_block[0].select('input[type=hidden]')
                    for i in hiddenInputs:
                        if i.get('name')== 'initial-rating':
                            values = i.get('value')
                    break    
    
            #Product
            start = time.time()
            product1=soup.find('h1', class_ = "wt-text-body-03 wt-line-height-tight wt-break-word wt-mb-xs-1")
            while True and time.time()-start < 60:
                if product1 is None:        
                   product1=soup.find('h1', class_ = "wt-text-body-03 wt-line-height-tight wt-break-word wt-mb-xs-1") 
                else:
                   break
            if product1 is None:
                product = 'None'
            else:
                product = product1.get_text().strip()
            #Sales
            sales_help=soup.find('a', class_="wt-text-link-no-underline wt-display-inline-flex-xs wt-align-items-center")
            if sales_help is None:
                sales = 'No sales'
            else:
                sales = sales_help.find('span', class_ = "wt-screen-reader-only").get_text().strip()
        
            #Get Brand Name
            brand_help=soup.find('p',class_="wt-text-body-01 wt-mr-xs-1")
            brand=brand_help.find('span').get_text().strip()
            
            #Get Country
            country_help=soup.find('div', class_="js-ships-from wt-grid__item-xs-6 wt-mb-xs-3 wt-pr-xs-2")
            if country_help is None:
                country='No country infomation'
            else:
                country=country_help.find('p',class_="wt-text-body-03 wt-mt-xs-1 wt-line-height-tight").get_text().strip()
            
            
            #On Etsy Since
            etsysince_help = soup.find('div', class_ = "wt-display-flex-xs wt-flex-wrap wt-position-relative")
            if etsysince_help is None:
                etsysince = 'No historical information'
            else:
                etsysince_list = [tag for tag in etsysince_help]
                etsysince = etsysince_list[3].get_text().strip()
                etsysince = etsysince.replace('\n',' ')
            #print('Since'+'='+etsysince)
           
            
            #p = list(x for x in soup.find_all('div',"wt-grid__item-xs-6 wt-mb-xs-3 wt-pr-xs-2"))
            #d=p[0].get_text().strip('\n')
            #d.replace('\n',' ')
            
            #Ready to Ship or Not in X days
            shipping_help = soup.find('div', class_ ="wt-grid__item-xs-6 wt-mb-xs-3 wt-pr-xs-2")
            #print(shipping_help)
            if shipping_help is None:
                shipping = 'No Ready to Ship in X Days Info'
            else:
                shipping = shipping_help.find('p',class_ ="wt-text-body-03 wt-mt-xs-1 wt-line-height-tight").get_text().strip()
            #print('Shipping'+'='+shipping)
            
            #Free Shipping
            freeshipping1 = soup.find_all('p', class_= "wt-text-body-03 wt-mt-xs-1 wt-line-height-tight")    
            freeshipping1 = list(freeshipping1)
            if len(freeshipping1) < 3:
                freeshipping = 'No free shipping'
            else:
                freeshipping = 'Free shipping'
            #print('Free_shipping'+'='+freeshipping)
            
            '''
            #Number of people who have favorited the item
            favorites_help = soup.find_all('li', class_ = "list-inline-item")
            list(favorites_help)
                <a href="/listing/787587099/tabletop-role-playing-nameplate-dice/favoriters?ref=l2-collection-count">
                        6 favorites
                    </a>
                </li>)
            if favorites_help is None:
                favorites = 'No favorites'
            else: 
                favorites = favorites_help.get_text().strip()
            print('Favorites'+'='+favorites)
            '''
            #Date of most recent review
            mostrecentreview_help = soup.find('div', class_ = "wt-display-flex-xs wt-align-items-center wt-mb-xs-1")
            if mostrecentreview_help is None:
                mostrecentreview = 'No reviews'
            else:
                mostrecentreview = mostrecentreview_help.find('p',class_ = "wt-text-caption wt-text-gray").get_text().strip()    
            #mostrecentreview = soup.find_all('p', class_ = "wt-text-caption wt-text-gray").get_text().strip()            
            #print('Recent'+'='+mostrecentreview)
            #Special information 
            buyersareraving_help = soup.find('div',class_="ml-lg-4 wt-display-inline-flex-xs wt-align-items-center shop-highlight wt-pt-xs-2 wt-pt-sm-0")
            if buyersareraving_help is None:
                buyersareraving = 'No raves'
            else:
                buyersareraving1 = buyersareraving_help.find('p',class_="wt-text-caption-title wt-pl-xs-1")
                buyersareraving = buyersareraving1.get_text().strip()
            #print('Raves'+'='+buyersareraving)
            #Handmade or not
            handmade_help = soup.find('div', class_ ="wt-display-flex-xs wt-align-items-center wt-mb-lg-3")
            if handmade_help is None:
                handmade = 'Not handmade'
            else:
                handmade = handmade_help.find('p').get_text().strip()
            #print('Handmade'+'='+handmade)
            #Best-seller or not
            bestseller_help = soup.find('div',class_="wt-display-flex-xs wt-align-items-center")
            bestseller1 = bestseller_help.find('span', class_="wt-display-inline-block")
            if bestseller1 is None:
                bestseller = 'Not a bestseller'
            else:
                bestseller = bestseller1.get_text().strip()
            #print('Bestseller'+'='+bestseller)
                
            '''
            #Number of reviews for that product
            itemreview_help = soup.find(<'button',id = "same-listing-reviews-tab",class_ ="wt-tab__item", aria_controls ="same-listing-reviews-panel",aria_selected ="true", role = "tab", tabindex="0",data_wt_tab="")
            if itemreview_help is None:
                itemreview = 'None'
            else:
                itemreview1 = itemreview_help.find('span', class_="wt-badge wt-badge--status-02 wt-ml-xs-2")
                itemreview = itemreview1.get_text().strip()
           '''               
           #Price
            price_help = soup.find_all('div',class_="wt-display-flex-xs wt-align-items-center")
            price_help = list(price_help)
            #print(price_help)
            if bestseller == 'Not a bestseller':
                price1 = price_help[0].get_text().strip()
                price1 = price1.strip().replace('\n','').replace(' ','')
                price1=price1.replace('Loading','') 
                price=price1
            else:
                price1 = price_help[1].get_text().strip()
                price2 = str(price1)
                #print(price2)
                #price1 = price_help.find('p', class_ ="wt-text-title-03 wt-mr-xs-2").get_text().strip()
                #price2 = str(price1).replace('\n','').replace(' ','')
                #print(price2)
                price = re.search('\$\S+',price2).group()
                #print(price)
                price=str(price)
            
                #print(price)
           #FreeshippingwithinUS
            USfreeshipping_help = soup.find('div', class_="wt-display-flex-xs wt-align-items-center")
            domesticfreeshipping1 = USfreeshipping_help.find('p',class_="wt-position-relative wt-text-caption")
            if domesticfreeshipping1 is None:
                domesticfreeshipping = 'No domestic free shipping'
            else:
                domesticfreeshipping = domesticfreeshipping1.get_text().strip()
            #print(domesticfreeshipping)
            #print(reviews + '\n' + stars.rstrip() + '\n' + values + '\n' + product + '\n' + sales + '\n' + brand + '\n' + country + '\n' + etsysince + '\n' + shipping + '\n' + '\n' #+ mostrecentreview 
           #       + '\n' + handmade + '\n' + price+ '\n' + domesticfreeshipping + '\n' + '\n' + buyersareraving + '\n' + freeshipping + '\n' + bestseller)
        all=[reviews,stars,values,product,sales,brand,country,etsysince,shipping,mostrecentreview,handmade,price,domesticfreeshipping,#itemreview,
             buyersareraving,freeshipping, bestseller]
        print(all)
        #print(etsysince)
        websites.append(all)
            
#print(websites)


