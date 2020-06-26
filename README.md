# EtsyEdge: A predictive tool for sellers on the Etsy platform
This repository collects the data and scripts used to assemble my Insight Data Science project. The completed product is viewable [here](http://www.insightetsy.site) and slides which accompany the product demonstration are [here](https://docs.google.com/presentation/d/1TveW6OcEcbtEsvoqgeiciuSwDoYenJMXrOXGLqTVfEE/edit).

## Overview
Etsy is the single largest platform for sellers of handmade goods, with a growing seller base that numbers in the millions. EtsyEdge is a tool which allows an individual seller to see how his or her products measure up against the competition. This information feeds into a machine learning model which makes recommendations to increase future sales.

### Step 1: Identify competitive gaps

![Picture1](https://github.com/MadhavChandrasekher/Projects/blob/master/Pictures/Picture1.png)















### Step 2: Predict sales increases












## Data
I collected data by scraping seller pages on the Etsy website using *Beautiful Soup*. I did not use Etsy's proprietary API since it comes with both restrictions on content and the number of daily requests. Sellers are grouped by product category and I scraped four categories: candles, earrings, wooden name puzzle toys, and baby toys. Categories were chosen on the basis of two criteria. First, they had to be large enough to admit reasonable statistical inference. Second, items in a category had to be similar enough so that they could be considered competitors. Raw data and scraping scripts are available in the following two folders,

* #### ~/Raw 
  - contains raw data csvs from scraped product categories and text files of url lists of Etsy seller pages.

* #### ~/Scripts/Scraping and cleaning scripts 
  - script which creates text list of seller pages
  - script which scrapes individual seller pages for product data
  - script which cleans the raw data csv files
  
## Analysis
The model and analysis are stored in the following folder,
* #### ~/SCripts/Models
	- script `Analysis.py` which generates results from ML algorithms and related regression results.   
	- script `Stats.py' which generates all descriptive statistics and box plots appearing in the slide deck.
	
## Running the app
The web app, EtsyEdge, is scripted in *Streamlit*. The main projects folder contains the four documents needed to run the app. `Streamlitapp.py` runs the app and `streamlit_candles.csv`, `streamlit_earrings.csv`, `cleaned_data_logit_subsample.csv` are the (cleaned) data files used for the app.
  
	


 

