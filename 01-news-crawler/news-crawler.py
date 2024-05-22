#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:24:37 2024

@author: yunbai
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime
import os
import time

def get_news_article_content(url):
    '''
    find news articles from the news url (not the news list url)
    url: the link to the full news
    '''
    # set http request
    response = requests.get(url)
    # analysis contents with beautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    # find titles
    title_tag = soup.find('h1', class_='article__title')
    title = title_tag.get_text(strip=True) if title_tag else None
    
    # find all the paragraphs
    paragraphs = soup.find_all('p', class_='article__paragraph')
    # concat article with pararaphs
    article_body = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])
    
    return title,article_body
        
def save_article_content(date_str, article_content, article_index):
    '''
    save the news article in txt
    date_str: French date with "%d-%m-%Y", 
    article_content: full article of news, 
    article_index: index of a piece of news in a day
    '''
    date = datetime.strptime(date_str, "%d-%m-%Y")
    year, month, day = date.strftime("%Y"), date.strftime("%m"), date.strftime("%d")
    
    base_dir = "news_articles"
    dir_path = os.path.join(base_dir, year, month, day)
    os.makedirs(dir_path, exist_ok=True)
    
    file_path = os.path.join(dir_path, f"article_{article_index}.txt")
    
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(article_content)

def generate_date_strings(start_year, end_year):
    '''
    create the date strings from start year to end year
    '''
    date_strings = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            day = 1
            while True:
                try:
                    date = datetime(year, month, day)
                    date_strings.append(date.strftime('%d-%m-%Y'))
                    day += 1
                except ValueError:
                    break
    return date_strings
        
def main_crawler():
    date_strings = generate_date_strings(2016, 2023)
    for date_str in date_strings:
        print("Processing date:", date_str)
        url = f'https://www.lemonde.fr/archives-du-monde/{date_str}/'
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            news_links = soup.select('a.teaser__link')
        except requests.exceptions.RequestException as e:
            print(f"Failed to retrieve news list for {date_str}: {e}")
            continue
        
        for idx, link in enumerate(news_links):
            try:
                title, article_body = get_news_article_content(link['href'])
                if article_body and title:
                    save_content = title + '\n' + article_body
                    save_article_content(date_str, save_content, idx)
            except requests.exceptions.RequestException as e:
                print(f"Failed to retrieve article from {link['href']}: {e}")
                continue
        time.sleep(1)
            
main_crawler()    
