import nltk
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# input: paragraph (string) 
# output: average sentiment of the paragraph and a list containing a tuple of each sentence with its corresponding sentiment
# sentiment is rating between -4 (VERY negative) and 4 (VERY positive)
def calcSentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentence_list = nltk.sent_tokenize(text)
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

# article file to a string
article = open("article.txt", "r")
article_text = article.read()

# string split into paragraphs
paragraph_list = article_text.split("\n")
paragraph_list = list(filter(None, paragraph_list))


# paragraphs sentiment analysed
para_sentiments_list = []
modified_paragraph_list = []
for paragraph in paragraph_list:
    para_sentiment, modified_paragraph = calcSentiment(paragraph)
    para_sentiments_list = para_sentiments_list + [para_sentiment]
    modified_paragraph_list = modified_paragraph_list + [modified_paragraph]

# total sentiment (from all sentences)
sentiment_tot, modified_tot = calcSentiment(article_text)


sentence_list, sentiment_list = zip(*modified_tot)

plt.plot(np.arange(1, len(sentiment_list)+1), sentiment_list, '-s')
plt.xlabel('Sentence index')
plt.ylabel('Sentiment Intensity')

plt.figure()
plt.plot(np.arange(1, len(para_sentiments_list)+1), para_sentiments_list, '-rs')
plt.xlabel('Paragraph index')
plt.ylabel('Average Sentiment Intensity')
plt.show()