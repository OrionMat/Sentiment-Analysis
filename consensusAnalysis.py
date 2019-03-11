from bs4 import BeautifulSoup
from googlesearch import search 
import requests

query = " Ethiopian Airlines plane crashed"

def get_article(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # get article title
    title = soup.find('h1')
    if title == None:
        print('no title')
        return "",""
    title_txt = title.get_text() 
    print(title_txt)

    # possibly get sub-titles (h2)

    # get all paragraphs in the article body
    articlebodies = soup.find_all('article')
    print(articlebodies, ' .. type: ', type(articlebodies))
    if articlebodies == None: # catches videos
        print('no article body')
        return "",""

    #for article in articlebodies
    #all_paragraphs = articlebody.find_all('p')
#
    ## get the text of all paragraphs and flatten them into an article
    #article = ''
    #for paragraph in all_paragraphs:
    #    text = paragraph.get_text()
    #    article = article + text
#
    #return title_txt, article


url = 'https://www.bbc.co.uk/news/world-africa-47521744'
get_article(url)





















#def article_scrape(query):
#
#    # gets 10 top links form google
#    article_links = search(query, tld="co.in", num=10, stop=10, pause=2)
#
#    title_list = []
#    article_list = []
#    for link in article_links:
#        title, article = get_article(link)
#        if title and article:
#            title_list = title_list + [title]
#            article_list = article_list + [article]


# AP
# Reuters
# Bloobburg
# NYT
# BBC

# observations (general google):
# text is written in p blocks