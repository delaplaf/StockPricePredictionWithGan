#!/usr/bin/env python
# coding: utf-8

"""
You need to be in web_scrapping folder.
Script to scrape the news of a desired company on seeking alpha. 
Requires the chromedriver.exe file in the folder. 
Modify this file as explained in point number 2 of this link :
https://piprogramming.org/articles/How-to-make-Selenium-undetectable-and-stealth--7-Ways-to-hide-your-Bot-Automation-from-Detection-0000000017.html
"""

import os
import time
import pandas as pd
import random

from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service


def get_driver(n):
    """
    Obtain the n-th page of news from the website seeking alpha for the desired company

    Parameters
    ----------
    n : int
        n-th page

    Returns
    -------
    chrome driver 
    """
    options = get_options()   
    ser = Service(PATH)

    driver = webdriver.Chrome(service=ser, options=options)
    change_property(driver)
    
    url = 'https://seekingalpha.com/symbol/FB/news?page=' + str(n)
    driver.get(url)
    
    return driver


def get_options():
    """
    Driver options especially to avoid being detected as a bot

    Returns
    -------
    class Options for driver 
    """
    options = Options()
    
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('--disable-blink-features=AutomationControlled')
    
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    
    options.add_argument('--start-maximized')
    
    return options


def change_property(driver):
    """
    Driver property especially to avoid being detected as a bot
    """
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source":
            "const newProto = navigator.__proto__;"
            "delete newProto.webdriver;"
            "navigator.__proto__ = newProto;"
    })


def get_html(driver):
    """
    Get html code from driver

    Parameters
    ----------
    driver 

    Returns
    -------
    string html code 
    """
    html = driver.page_source
    soup = BeautifulSoup(html, "html5lib")
    return soup


def get_titles(soup):
    """
    Get titles of news articles from html code

    Parameters
    ----------
    soup : string
        html code

    Returns
    -------
    list of titles 
    """
    title = soup.find_all('h3',{'data-test-id':'post-list-item-title'})
    title_list=[]
    for i in title:
        d=i.get_text()
        title_list.append(d)
    return title_list


def get_dates(soup):
    """
    Get dates of news articles from html code

    Parameters
    ----------
    soup : string
        html code

    Returns
    -------
    list of dates 
    """
    date = soup.find_all('span',{'data-test-id':'post-list-date'})
    date_list=[]
    for i in date:
        d=i.get_text()
        date_list.append(d)
    return date_list


def isEmpty(soup):
    """
    Check if there is a next page for news

    Parameters
    ----------
    soup : string
        html code

    Returns
    -------
    boolean 
    """
    empty = soup.find_all('div',{'data-test-id':'empty-state-message'})
    if empty:
        return True
    return False


def test():
    """
    Checks that the code works for a single page.
    Sometimes don't work because of add-blocker.
    """
    driver = get_driver(1)
    soup = get_html(driver)
    titles = get_titles(soup)
    dates = get_dates(soup)
    assert(len(titles) > 0)
    assert(len(dates) > 0)
    time.sleep(max(3, random.random()*5))
    driver.quit()
    print("Test OK")


def get_all_articles():
    """
    Obtain the titles and dates of all the articles on the site 
    concerning the desired company. 
    Saved the results to a csv file.
    """
    allTitles = []
    allDates = []

    n = 1
    while n==1 or not(isEmpty(soup)):
        driver = get_driver(n)
        time.sleep(max(10, random.random()*60))
        
        soup = get_html(driver)
        titles = get_titles(soup)
        dates = get_dates(soup)
        
        allTitles.extend(titles)
        allDates.extend(dates)

        driver.quit()
        time.sleep(max(10, random.random()*60))
        
        if len(titles) > 1:
            print("First article of the page", n, ":", dates[0], titles[0])
        n += 1

    print("Number of pages:", n-2)
    print("Number of titles:",len(allTitles))
    print("Number of dates:", len(allDates))

    df_show_info = pd.DataFrame({'Article_Title': allTitles, 'Article_Date': allDates})
    df_show_info.to_csv('articles.csv')


if __name__ == '__main__':
    PATH = os.path.abspath("chromedriver.exe")
    test()
    get_all_articles()