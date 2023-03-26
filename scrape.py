import requests
import time
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urlencode
import csv
from random import randint

def get_reviews():
    pdt_url = 'https://www.amazon.com/Beats-Fit-Pro-Cancelling-Built/product-reviews/B09JL41N9C/ref=cm_cr_getr_d_paging_btm_next_{}?ie=UTF8&reviewerType=all_reviews&pageNumber={}'
    review_list = []
    HEADERS = ({'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
            AppleWebKit/537.36 (KHTML, like Gecko) \
            Chrome/90.0.4430.212 Safari/537.36',
            'Accept-Language': 'en-US, en;q=0.5'})
    current_page = 1
    while True:
        print("Current Page:", current_page)
        current_url = pdt_url.format(current_page, current_page)
        time.sleep(randint(10,100))
        page = requests.get(current_url, headers = HEADERS)
        soup = BeautifulSoup(page.content, 'html.parser')
        #print(soup)
        reviews = soup.find_all('div',{'data-hook':'review'})
        #print(reviews)

        for review in reviews:
           rating = float(review.find('i', {'data-hook': 'review-star-rating'}).get_text().replace('out of 5 stars', '').strip())
           verified = review.find('span', {'data-hook': 'avp-badge'}).get_text().strip()
           title = review.find('a', {'data-hook': 'review-title'}).get_text().strip()
           body = review.find('span', {'data-hook': 'review-body'}).get_text().strip()

           review_list.append({'RATING':rating, 'VERIFIED_PURCHASE': verified, 'REVIEW_TITLE':title, 'REVIEW_TEXT':body})
        
        
        
        next_button = soup.find('li', {'class': 'a-last'})
        if next_button is not None:
            current_page += 1 
        else:
            print("Missing next button")
            break

    review_list = [r for r in review_list if r]
    df_amazon =  pd.DataFrame(review_list)
    #df_amazon.to_csv('pdt_amazon_reviews.csv', index = False)
    with open('beats_amzn_reviews.csv', 'a') as f:
        df_amazon.to_csv(f, header=False, index=False)

get_reviews()
     


#Archive

# for item in soup.find_all("span", {"data-hook": "review-body"}):
    #   data_string = data_string + item.get_text()
    #   reviews.append(data_string)
    #   data_string = ""