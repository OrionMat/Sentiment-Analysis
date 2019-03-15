# NEWS WEB SCRAPING
import sys
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import re 




#%% New York Times: 

# scrapes a NYT article and its headline
def NYT_article_scrape(url):
    try:
        # gets a webpage
        response = requests.get(url)
    except:
        print("no response from webpage. Error: ", sys.exc_info()[0])
        raise
    soup = BeautifulSoup(response.text, 'html.parser')
    # get article title
    title = soup.find('h1')
    headline = title.get_text() 
    # get all paragraphs in the article body
    articlebody = soup.find(attrs={"name": "articleBody"})
    if articlebody == None: # catches videos
        return "",""
    all_paragraphs = articlebody.find_all('p')
    # get the text of all paragraphs and flatten them into an article
    article = ''
    for paragraph in all_paragraphs:
        text = paragraph.get_text()
        article = article + text
    return headline, article

# scrapes all article links on a NYT page
def NYT_link_scrape(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    article_links = soup.find_all('article') 
    hyp_link_list = []
    for link in article_links:
        url = link.find('a')
        hyperlink = url.get('href')
        hyp_link_list = hyp_link_list + [hyperlink]
    return hyp_link_list

# gets a list of NYT articles and their corresponing headlines from a page url
def NYT_page_scrape(url):
    hyp_link_list = NYT_link_scrape(url)
    title_list = []
    article_list = []
    for hyp_link in hyp_link_list:
        url = 'https://www.nytimes.com'+ hyp_link
        title, article = NYT_article_scrape(url)
        if title and article:
            title_list = title_list + [title]
            article_list = article_list + [article]
    return title_list, article_list

# gets NYT links form google with article dates
def google_NYT_links(query):
    NYT_query = "new york times + " + query
    article_links = search(NYT_query, tld="co.in", num=15, stop=10, pause=2)    # gets 10 top links from google
    url_list = []
    date_list = []
    for link in article_links:
        if re.search(r"www.nytimes.com/\d{4}/\d{2}/\d{2}", link) != None:
            date = re.search(r"\d{4}/\d{2}/\d{2}", link)
            url_list = url_list + [link]
            date_list = date_list + [date.group()]
    return url_list, date_list

# gets a list of NYT articles and their corresponing headlines from a list of links
def NYT_links_scrape(link_list):
    title_list = []
    article_list = []
    for link in link_list:
        title, article = NYT_article_scrape(link)
        if title and article:
            title_list = title_list + [title]
            article_list = article_list + [article]
    return title_list, article_list








query = "Ethiopian Airlines plane crashed"
url_list, date_list = google_NYT_links(query)
title_list, article_list = NYT_links_scrape(url_list)

#for idx in range(len(title_list)):
#    print(date_list[idx], "  :  ", title_list[idx], "  :  ", url_list[idx])
