import nltk
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# article file into a string
article = open("article.txt", "r")
text = article.read()


analyzer = SentimentIntensityAnalyzer()

#text = """It was one of the worst movies I've seen, despite good reviews. Unbelievably bad acting!! Poor direction. VERY poor production. The movie was bad. Very bad movie. VERY BAD movie!"""
sentence_list = nltk.sent_tokenize(text)
print(sentence_list)
print('\n')

textSentiments = 0.0
for sentence in sentence_list:
        vs = analyzer.polarity_scores(sentence)
        print("{:-<120} {}".format(sentence, str(vs["compound"])))
        textSentiments += vs["compound"]
print("AVERAGE SENTIMENT FOR PARAGRAPH: \t" + str(round(textSentiments / len(sentence_list), 4)))
