# NEWS WEB SCRAPING
import sys
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import re 
import data_saving as save_to_CSV





#%% New York Times: 

# gets NYT links from google with article dates
def google_NYT_links(query):
    NYT_query = "new york times + " + query
    article_links = search(NYT_query, tld="com", num=15, stop=10, pause=2)    # gets 15 top links from google
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
    if title == None:
        print("NYT article title not found")
        return "", ""
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





#%% BBC:

# gets BBC links from google with article dates
def google_BBC_links(query):
    BBC_query = "BBC news + " + query
    article_links = search(BBC_query, tld="co.uk", num=15, stop=10, pause=2)    # gets 15 top links from google
    url_list = []
    for link in article_links:
        if re.search(r"www.bbc.co.uk/news/[^ av]", link) != None:
            url_list = url_list + [link]
    return url_list

# gets a list of BBC articles and their corresponing headlines and dates from a list of links
def BBC_links_scrape(link_list):
    title_list = []
    article_list = []
    date_list = []
    for link in link_list:
        title, article, date = BBC_article_scrape(link)
        if title and article and date:
            title_list = title_list + [title]
            article_list = article_list + [article]
            date_list = date_list + [date]
    return title_list, article_list, date_list

# scrapes a BBC article, its headline and its date
def BBC_article_scrape(url):
    # get webpage
    try:
        response = requests.get(url)
    except:
        print("no response from webpage. Error: ", sys.exc_info()[0])
        raise
    soup = BeautifulSoup(response.text, 'html.parser')
    # get article title
    title = soup.find('h1')
    if title == None:
        print("BBC article title not found")
        return "", "", ""
    headline = title.get_text() 
    # get all paragraphs in the article body
    articlebody = soup.find(attrs={"property": "articleBody"})
    if articlebody == None: # catches videos
        return "","", ""
    all_paragraphs = articlebody.find_all('p')
    # get the text of all paragraphs and flatten them into an article
    article = ''
    for paragraph in all_paragraphs:
        text = paragraph.get_text()
        article = article + text
    # get article date
    date_section = soup.find(attrs={"class": "mini-info-list__item"})
    date_location = date_section.find('div')
    date = date_location.get_text()
    return headline, article, date




query = "Ethiopian Airlines plane crashed"
# Shares in Boeing fell by 12.9% on Monday in the wake of the crash





#%% NYT CSV

url_list, date_list = google_NYT_links(query)
title_list, article_list = NYT_links_scrape(url_list)
#for idx in range(len(title_list)):
#    print(date_list[idx], "  :  ", title_list[idx], "  :  ", url_list[idx])
#for idx in range(2):
#    print(url_list[idx], '\n', article_list[idx], '\n\n\n\n')
news_dicList = save_to_CSV.lists_to_dictList('NYT', title_list, date_list, article_list, url_list)
print(news_dicList)
csv_file_path = 'consensus_data.csv'
csv_columns = ['agency', 'title', 'date', 'article', 'link']
save_to_CSV.initiate_csv(csv_file_path, csv_columns)
save_to_CSV.append_csv(csv_file_path, csv_columns, news_dicList)     # use rest of the time





#%% BBC CSV

url_list = google_BBC_links(query)
title_list, article_list, date_list = BBC_links_scrape(url_list)
#for idx in range(len(title_list)):
#    print(date_list[idx], "  :  ", title_list[idx], "  :  ", article_list[idx][0:15])
news_dicList = save_to_CSV.lists_to_dictList('BBC', title_list, date_list, article_list, url_list)
print(news_dicList)
csv_file_path = 'consensus_data.csv'
csv_columns = ['agency', 'title', 'date', 'article', 'link']
save_to_CSV.append_csv(csv_file_path, csv_columns, news_dicList)     # use rest of the time