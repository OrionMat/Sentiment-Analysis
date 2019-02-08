import requests
from bs4 import BeautifulSoup

def get_article(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # get article title
    #title = soup.find('h1')
    #title_txt = title.get_text() 

    # get all paragraphs in the article body
    articlebody = soup.find(attrs={"name": "articleBody"})
    all_paragraphs = articlebody.find_all('p')

    # get the text of all paragraphs and flatten them into an article
    article = ''
    for paragraph in all_paragraphs:
        text = paragraph.get_text()
        article = article + text
    return(article)

def nyt_opinion():
    url = 'https://www.nytimes.com/section/opinion'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    article_links = soup.find_all('article') 
    #print(type(article_links))

    hyp_link_list = []
    for link in article_links:
        url = link.find('a')
        hyperlink = url.get('href')
        hyp_link_list = hyp_link_list + [hyperlink]

    article_list = []
    for hyp_link in hyp_link_list:
        url = 'https://www.nytimes.com/'+ hyp_link
        article = get_article(url)
        article_list = article_list + [article]

    return article_list

print(nyt_opinion()[3])