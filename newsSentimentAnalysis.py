#%%
import nltk
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import json
import glob
import errno
import pandas
import random


#%%
# FUNCTIONS

# input: text (string) 
# output: average sentiment of the text and a list containing a tuple of each sentence with its corresponding sentiment
# sentiment is rating between -4 (EXTREAMLY negative) and 4 (EXTREAMLY positive)
def calcSentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentence_list = nltk.sent_tokenize(text)
    sentiment_list = [] 
    if not sentence_list:
        return sentence_list, sentiment_list
        #raise ValueError('no sentences in the text given -> 0/0')
    for sentence in sentence_list:
            vs = analyzer.polarity_scores(sentence)
            compound_sentiment = round(vs["compound"]*4, 4)
            sentiment_list = sentiment_list + [compound_sentiment]
    return sentence_list, sentiment_list

def get_json_text(json_file):
    json_string = json_file.read()
    json_dict = json.loads(json_string)
    return json_dict["text"]

def split_toParagraphs(article_text):
    paragraph_list = article_text.split("\n")
    paragraph_list = [paragraph.strip(' ') for paragraph in paragraph_list]
    paragraph_list = list(filter(None, paragraph_list))
    return paragraph_list

def paragraph_analysis(paragraph_list):
    all_para_sentiments = []
    all_para_sentences = []
    for paragraph in paragraph_list:
        para_sentences, para_sentiments = calcSentiment(paragraph)
        if para_sentiments:
            para_sentiment_avg = np.mean(np.asarray(para_sentiments))
            all_para_sentiments = all_para_sentiments + [para_sentiment_avg]
            all_para_sentences = all_para_sentences + [para_sentences]
    return all_para_sentences, all_para_sentiments

# preforms sentiment analysis on each article in a list of articles
def article_list_analysis(article_list):
    article_sentence_list = []
    sentence_sentiment_list = [] 
    avg_article_sentiment_list = []
    for article_text in article_list:
        if article_text:
            article_text = str(article_text)
            sentence_list, sentiment_list = calcSentiment(article_text)
            if sentiment_list:
                article_sentiment = np.mean(np.asarray(sentiment_list))
                article_sentence_list = article_sentence_list + [sentence_list]
                sentence_sentiment_list = sentence_sentiment_list + [sentiment_list]
                avg_article_sentiment_list = avg_article_sentiment_list + [article_sentiment]
    return article_sentence_list, sentence_sentiment_list, avg_article_sentiment_list

def politi_buzz_analysis(path):
    files = glob.glob(path)
    article_list = []
    for file_name in files:
        try:
            with open(file_name, 'r') as json_file:
                article_text = get_json_text(json_file)
                article_list = article_list + [article_text]   
        except IOError as exc:
            if exc.errno != errno.EISDIR:
                raise
    return article_list_analysis(article_list)

def BBC_analysis(path):
    files = glob.glob(path)
    article_list = []
    for file_name in files:
        try:
            with open(file_name, 'r') as txt_file:
                article_text = txt_file.read()
                article_list = article_list + [article_text]   
        except IOError as exc:
            if exc.errno != errno.EISDIR:
                raise
    return article_list_analysis(article_list)

def kaggel_Fact_Fake_analysis(path):
    df = pandas.read_csv(path)
    fake_df = df.loc[df['label'] == 1]
    real_df = df.loc[df['label'] == 0]
    article_list_fake = fake_df['text'].values.tolist()
    article_list_real = real_df['text'].values.tolist()
    article_list_fake = random.sample(article_list_fake, 100)  # needed as 13000 takes too long
    article_list_real = random.sample(article_list_real, 100)  # needed as 13000 takes too long
    fake_sentence_list, sentence_sentiment_fake_list, article_sentiment_fake_list = article_list_analysis(article_list_fake)
    real_sentence_list, sentence_sentiment_real_list, article_sentiment_real_list = article_list_analysis(article_list_real)
    return article_sentiment_fake_list, fake_sentence_list, sentence_sentiment_fake_list, article_sentiment_real_list, real_sentence_list, sentence_sentiment_real_list

def avg_var_calculation(sentiment_list, factOrFake, news_source):
    sentiment_array = np.asarray(sentiment_list)
    sentiment_avg = np.mean(sentiment_array)
    sentiment_var = np.var(sentiment_array)
    print(factOrFake + " article avg " + news_source + ": " + str(sentiment_avg))
    print(factOrFake + " article var " + news_source + ": " + str(sentiment_var))
    return sentiment_array, sentiment_avg, sentiment_var

def cal_article_abs_sentiment(list_list):
    mean_list = []
    for list in list_list:
        mean = np.mean( np.absolute( np.asarray(list)))
        mean_list = mean_list + [mean]
    return np.asarray(mean_list)

def kaggle_mult_news_analysis(path, publication):
    df = pandas.read_csv(path)

    df_publication = df.loc[df['publication'] == publication]

    article_list = df_publication['content'].values.tolist()

    article_list = random.sample(article_list, 100)  # needed as otherwise it takes too long

    sentence_list, sentence_sentiments, article_sentiments = article_list_analysis(article_list)

    return sentence_list, sentence_sentiments, article_sentiments

#def kaggel_Fake_analysis(path):
#    df = pandas.read_csv(path)
#    article_list = df['text'].values.tolist()
#    art_sub_list = random.sample(article_list, 100)  # needed as 13000 takes too long
#    sentence_list, sentence_sentiment_list, article_sentiment_tot_list = article_list_analysis(art_sub_list)
#    return article_sentiment_tot_list, sentence_list, sentence_sentiment_list