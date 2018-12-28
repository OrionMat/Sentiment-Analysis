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

# input: paragraph (string) 
# output: average sentiment of the paragraph and a list containing a tuple of each sentence with its corresponding sentiment
# sentiment is rating between -4 (VERY negative) and 4 (VERY positive)
def calcSentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentence_list = nltk.sent_tokenize(text)
    para_sentiment_avg, sentiment_paragraph_list = 0.0, 0.0

    if len(sentence_list) == 0:
        return para_sentiment_avg, sentiment_paragraph_list
        #raise ValueError('no sentences in the text given -> 0/0')

    sentiment_list = []
    sentence_Sentiments = 0.0
    for sentence in sentence_list:
            vs = analyzer.polarity_scores(sentence)
            compound_sentiment = round(vs["compound"]*4, 4)
            sentiment_list = sentiment_list + [compound_sentiment]
            sentence_Sentiments += compound_sentiment

    sentiment_paragraph_list = list(zip(sentence_list, sentiment_list))
    para_sentiment_avg = sentence_Sentiments / len(sentence_list)

    return para_sentiment_avg, sentiment_paragraph_list





# ANALYSIS

# loops though all the fake articles
path = 'C:\\Users\\orion\\Documents\\Python programming\\Sentiment Analysis\\FakeNewsNet-master\\Data\\BuzzFeed\\FakeNewsContent\\*.json'
files = glob.glob(path)

article_sentiment_tot_list = []
for file_name in files:
    try:
        with open(file_name, 'r') as json_file:
            json_string = json_file.read()
            #print(json_string)
            json_dict = json.loads(json_string)
            article_text = json_dict["text"]
            #print(json_dict["text"])

            # string split into paragraphs
            paragraph_list = article_text.split("\n")
            paragraph_list = [paragraph.strip(' ') for paragraph in paragraph_list]
            paragraph_list = list(filter(None, paragraph_list))
            #print(paragraph_list)

            # paragraphs sentiment analysed
            
            para_sentiments_list = []
            modified_paragraph_list = []
            for paragraph in paragraph_list:
                para_sentiment, modified_paragraph = calcSentiment(paragraph)
                para_sentiments_list = para_sentiments_list + [para_sentiment]
                modified_paragraph_list = modified_paragraph_list + [modified_paragraph]

            # total sentiment (from all sentences)
            article_sentiment_tot, modified_tot = calcSentiment(article_text)
            article_sentiment_tot_list = article_sentiment_tot_list + [article_sentiment_tot]
            

    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise




'''
sentence_list, sentiment_list = zip(*modified_tot)
print(modified_tot)




# plot of sentiment for each sentence
plt.plot(np.arange(1, len(sentiment_list)+1), sentiment_list, '-s')
plt.xlabel('Sentence index')
plt.ylabel('Sentiment Intensity')

# plot of sentiment for each paragraph
plt.figure()
plt.plot(np.arange(1, len(para_sentiments_list)+1), para_sentiments_list, '-rs')
plt.xlabel('Paragraph index')
plt.ylabel('Average Sentiment Intensity')
plt.show()
'''

# plot of sentiment for each article
plt.figure()
plt.plot(np.arange(1, len(article_sentiment_tot_list)+1), article_sentiment_tot_list, '-rs')
plt.xlabel('Paragraph index')
plt.ylabel('Average Sentiment Intensity')
plt.show()

#print(article_sentiment_tot_list)