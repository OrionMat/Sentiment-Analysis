import nltk
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import json
import glob
import errno





# FUNCTIONS

# input: text (string) 
# output: average sentiment of the text and a list containing a tuple of each sentence with its corresponding sentiment
# sentiment is rating between -4 (VERY negative) and 4 (VERY positive)
def calcSentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentence_list = nltk.sent_tokenize(text)
    text_sentiment_avg, sentiment_paragraph_list = 0.0, 0.0

    if len(sentence_list) == 0:
        return text_sentiment_avg, sentiment_paragraph_list
        #raise ValueError('no sentences in the text given -> 0/0')

    sentiment_list = []
    sentence_sentiments = 0.0
    for sentence in sentence_list:
            vs = analyzer.polarity_scores(sentence)
            compound_sentiment = round(vs["compound"]*4, 4)
            sentiment_list = sentiment_list + [compound_sentiment]
            sentence_sentiments += compound_sentiment

    sentiment_paragraph_list = list(zip(sentence_list, sentiment_list))
    text_sentiment_avg = sentence_sentiments / len(sentence_list)

    return text_sentiment_avg, sentiment_paragraph_list

def get_json_text(json_file):
    json_string = json_file.read()
    json_dict = json.loads(json_string)
    #print(json_dict["text"])
    return json_dict["text"]

# string split into paragraphs
def split_paragraphs(article_text):
    paragraph_list = article_text.split("\n")
    paragraph_list = [paragraph.strip(' ') for paragraph in paragraph_list]
    paragraph_list = list(filter(None, paragraph_list))
    #print(paragraph_list)
    return paragraph_list

def paragraph_analysis(paragraph_list):
    para_sentiments_list = []
    modified_paragraph_list = []
    for paragraph in paragraph_list:
        para_sentiment, modified_paragraph = calcSentiment(paragraph)
        para_sentiments_list = para_sentiments_list + [para_sentiment]
        modified_paragraph_list = modified_paragraph_list + [modified_paragraph]
    return para_sentiments_list, modified_paragraph_list

def json_file_analysis(path):
    files = glob.glob(path)
    article_sentiment_tot_list = []
    sentence_sentiment_list = [] 
    for file_name in files:
        try:
            with open(file_name, 'r') as json_file:
                article_text = get_json_text(json_file)
                paragraph_list = split_paragraphs(article_text)
                para_sentiments_list, modified_paragraph_list = paragraph_analysis(paragraph_list)  # analysis of paragraph sentiment 
                article_sentiment_tot, modified_tot = calcSentiment(article_text)   # total article sentiment (from all sentences)
                
                
                sentence_list, sentiment_list = zip(*modified_tot)
                sentence_sentiment_list = sentence_sentiment_list + [sentiment_list]
                article_sentiment_tot_list = article_sentiment_tot_list + [article_sentiment_tot]
                
        except IOError as exc:
            if exc.errno != errno.EISDIR:
                raise

    return article_sentiment_tot_list, sentence_sentiment_list





# ANALYSIS

fake_news_path = 'C:\\Users\\orion\\Documents\\Python programming\\Sentiment Analysis\\FakeNewsNet-master\\Data\\BuzzFeed\\FakeNewsContent\\*.json'
fact_news_path = 'C:\\Users\\orion\\Documents\\Python programming\\Sentiment Analysis\\FakeNewsNet-master\\Data\\BuzzFeed\\RealNewsContent\\*.json'
fake_article_sentiment_tot_list, fake_sentence_sentiments = json_file_analysis(fake_news_path)
real_article_sentiment_tot_list, real_sentence_sentiments = json_file_analysis(fact_news_path)





# plot of sentiment for each article
plt.figure("Article sentiments")
plt.plot(np.arange(1, len(fake_article_sentiment_tot_list)+1), fake_article_sentiment_tot_list, '-ro')
plt.plot(np.arange(1, len(real_article_sentiment_tot_list)+1), real_article_sentiment_tot_list, '-gs')
plt.xlabel('Article index')
plt.ylabel('Average Sentiment Intensity')

plt.figure("Senctence sentiments")
fake_flat_list = [item for sublist in fake_sentence_sentiments for item in sublist]
real_flat_list = [item for sublist in real_sentence_sentiments for item in sublist]
plt.plot(np.arange(1, len(fake_flat_list)+1), fake_flat_list, '-r')
plt.figure()
plt.plot(np.arange(1, len(real_flat_list)+1), real_flat_list, '-g')
plt.xlabel('Sentence index')
plt.ylabel('Average Sentiment Intensity')





fake_article_sentiment_tot_list = np.asarray(fake_article_sentiment_tot_list)
real_article_sentiment_tot_list = np.asarray(real_article_sentiment_tot_list)

fake_avg = np.mean(fake_article_sentiment_tot_list)
fake_var = np.var(fake_article_sentiment_tot_list)
real_avg = np.mean(real_article_sentiment_tot_list)
real_var = np.var(real_article_sentiment_tot_list)

print("fake avg: " + str(fake_avg))
print("fake var: " + str(fake_var))
print("real_avg: " + str(real_avg))
print("real var: " + str(real_var))


plt.show()