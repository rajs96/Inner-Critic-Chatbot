import requests
from bs4 import BeautifulSoup
import urllib.request
import os
import re

url = 'https://zenodo.org/record/3941387'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
links = soup.find_all('a', class_='filename')

base_url = 'https://zenodo.org'
csv_links = [base_url + link.get('href') for link in links]

pattern = r'/([^/]+)\?download=1'
data_dir = 'data/mental_health_data'
for csv_link in csv_links:
    match = re.search(pattern, csv_link)
    write_path = os.path.join(data_dir, match.group(1))
    if not os.path.exists(write_path):
        urllib.request.urlretrieve(csv_link, os.path.join(data_dir, match.group(1)))  
