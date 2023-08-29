from typing import List
import re
from os.path import join
import urllib.request
import requests
from bs4 import BeautifulSoup

from config import REDDIT_INSTANCE_GENERATION_CONFIG


def get_csv_links(csv_url: str, base_url: str) -> List[str]:
    res = requests.get(csv_url)
    soup = BeautifulSoup(res.text, "html.parser")
    links = soup.find_all("a", class_="filename")
    csv_links = [base_url + link.get("href") for link in links]
    return csv_links


def download_csv_links(csv_links: List[str], output_dir: str):
    pattern = r"/([^/]+)\?download=1"
    for csv_link in csv_links:
        match = re.search(pattern, csv_link)
        urllib.request.urlretrieve(csv_link, join(output_dir, match.group(1)))


if __name__ == "__main__":
    csv_links = get_csv_links(
        csv_url=REDDIT_INSTANCE_GENERATION_CONFIG["reddit_data_csv_url"],
        base_url=REDDIT_INSTANCE_GENERATION_CONFIG["reddit_data_base_url"],
    )
    download_csv_links(
        csv_links=csv_links,
        output_dir=REDDIT_INSTANCE_GENERATION_CONFIG["reddit_data_folder_path"],
    )