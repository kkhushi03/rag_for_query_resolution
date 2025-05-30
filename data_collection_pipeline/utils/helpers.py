from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


# Should return a BeautifulSoup object created from Selenium-rendered HTML.
def use_selenium_to_get_links(url):
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    html = driver.page_source
    driver.quit()
    return BeautifulSoup(html, "html.parser")

# Simple regex to extract a 4-digit year from the file name.
def infer_year_from_filename(filename):
    for part in filename.split("_"):
        if part.isdigit() and len(part) == 4:
            return part
    return "Unknown"

# Can use keyword matching or NLP (e.g., just return cleaned file name as a placeholder).
def infer_topic_from_filename(filename):
    return filename.replace("_", " ").split(".")[0][:50]