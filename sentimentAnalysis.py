import nltk
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# input: paragraph (string) 
# output: average sentiment of the paragraph and a list containing a tuple of each sentence with its corresponding sentiment
# sentiment is rating between -5 (VERY negative) and 5 (VERY positive)
def paragraphSentiment(paragraph):
    analyzer = SentimentIntensityAnalyzer()
    sentence_list = nltk.sent_tokenize(paragraph)
    sentiment_list = []
    sentence_Sentiments = 0.0
    for sentence in sentence_list:
            vs = analyzer.polarity_scores(sentence)
            compound_sentiment = round(vs["compound"]*4, 4)
            #print("{:-<100} {}".format(sentence, str(compound_sentiment)))
            sentiment_list = sentiment_list + [compound_sentiment]
            sentence_Sentiments += compound_sentiment

    sentiment_paragraph_list = list(zip(sentence_list, sentiment_list))
    para_sentiment_avg = sentence_Sentiments / len(sentence_list)

    return para_sentiment_avg, sentiment_paragraph_list

# article file to a string
article = open("article.txt", "r")
text = article.read()

# string split into paragraphs
paragraph_list = text.split("\n")
paragraph_list = list(filter(None, paragraph_list))


# paragraphs sentiment analysed
#para_sentiments_list = []
#for paragraph in paragraph_list:
    #para_sentiment, modified_paragraph = paragraphSentiment(paragraph)
    #para_sentiments_list = para_sentiments_list + para_sentiment
para_sentiment, modified_paragraph = paragraphSentiment(text)
print(text, "\n\n", modified_paragraph, "\n\n", "average sentiment: " + str(para_sentiment))

sentence_list, sentiment_list = zip(*modified_paragraph)

print("hi")
print("\n", sentiment_list) 

plt.plot(np.arange(1, len(sentiment_list)+1), sentiment_list, '-s')
plt.xlabel('Sentence index')
plt.ylabel('Sentiment Intensity')
plt.show()