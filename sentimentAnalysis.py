import nltk
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def paragraphSentiment(paragraph):
    analyzer = SentimentIntensityAnalyzer()
    sentence_list = nltk.sent_tokenize(paragraph)
    sentiment_sentence_list = sentence_list
    sentence_Sentiments = 0.0
    for idx, sentence in enumerate(sentence_list):
            vs = analyzer.polarity_scores(sentence)
            #print("{:-<100} {}".format(sentence, str(vs["compound"])))
            sentiment_sentence_list[idx] = sentence + "=> " + str(vs["compound"]) + "."
            sentence_Sentiments += vs["compound"]
    para_sentiment_avg = round(sentence_Sentiments / len(sentence_list), 4)
    sentiment_paragraph = " ".join(sentiment_sentence_list)

    return para_sentiment_avg, sentiment_paragraph

# article file into a string
article = open("article.txt", "r")
text = article.read()

para_sentiment, modified_text = paragraphSentiment(text)
print(text, "\n\n", modified_text, "\n\n", "average sentiment: " + str(para_sentiment*5))




#text = """It was one of the worst movies I've seen, despite good reviews. Unbelievably bad acting!! Poor direction. VERY poor production. The movie was bad. Very bad movie. VERY BAD movie!"""

