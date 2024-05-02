# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:46:23 2024

@author: sebst
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import time
import re

chrome_driver_path = r'C:\Users\sebst\OneDrive\Documents\Apps+Games\chromedriver-win64\chromedriver-win64\chromedriver.exe'  # Replace this with the actual path to chromedriver

# Create a service object
# service = Service(chrome_driver_path)

# Launch a browser instance
driver = webdriver.Chrome(executable_path=chrome_driver_path)  # Example for Chrome, you can use other drivers like Firefox, etc.
# driver.get("https://nisat.prio.org/Trade-Database/Researchers-Database/")
driver.get("https://nisatapps.prio.org/Query_SQL.aspx")


imports_check=driver.find_element(By.ID,"chkImpExp_0")
imports_check.click()
dropdown = driver.find_element(By.ID, "cmbCountry1")
dropdown.click() 
select1=Select(dropdown)
numbers=[]
for idx,option in enumerate(select1.options):
    if idx!=0:
        text = option.text.strip()
        # print(text)
        number = re.search(r'\[(\d+)\]$', text).group(1)
        numbers.append(number)
        # print(number)# selected_option1 = select1.options[1]
import requests
from bs4 import BeautifulSoup
import csv
# URL of the website you want to access
numbers=numbers[2:]
for number in numbers:
    print(number)
    url = f'https://nisatapps.prio.org/Results_SQL.aspx?C1={number}&C2=-2&p=Imports&Dep1=0&Dep2=False&r=False&W=100&dtl=3&Y=All%20Years&d=99&t=3&dls=False&csv=True&EY=All%20Years&scp=3'
    
    # Send a GET request to the website
    response = requests.get(url)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Get the HTML content of the website
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
    
        # Find all font tags
        font_tags = soup.find_all('font')
    
        # Search for the font tag containing the text "Dataset follows:"
        for font_tag in font_tags:
            if 'Reporter_Code' in font_tag.get_text():
                start_index = font_tag.prettify().find('Reporter_Code')
                end_condition = '<br/>\n<br/>\n<a href="Query_SQL.aspx">\n <img src="NewQuery.bmp"/>\n</a>\n</font>'
                end_index = font_tag.prettify().find(end_condition)
    
                # Print the prettified output between the start and end indices
                new_html=font_tag.prettify()[start_index:end_index-len(end_condition)-6]
                soup = BeautifulSoup(new_html, 'html.parser')
                # print(soup)
                lines=soup.get_text().split('\n \n ')
                # print(lines)
    
                # print(font_tag.prettify()[start_index:])    # print(html_content)
    else:
        print('Failed to retrieve the website content. Status code:', response.status_code)
    with open('output.csv', 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        print('writing')
        for idx,line in enumerate(lines):
            if idx!=0:
                columns = line.split(',')
            
            # Write each row to the CSV file
                writer.writerow(columns)