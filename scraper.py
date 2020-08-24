import requests
import urllib.request
from bs4 import BeautifulSoup
import time

ROOT = "https://www.vgmusic.com/music/console/"

pages = []

response = requests.get(ROOT)
soup = BeautifulSoup(response.text, "html.parser")

print(soup.findAll('a')[5:])


