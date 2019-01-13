#%%
import nltk
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
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

#def kaggel_Fake_analysis(path):
#    df = pandas.read_csv(path)
#    article_list = df['text'].values.tolist()
#    art_sub_list = random.sample(article_list, 100)  # needed as 13000 takes too long
#    sentence_list, sentence_sentiment_list, article_sentiment_tot_list = article_list_analysis(art_sub_list)
#    return article_sentiment_tot_list, sentence_list, sentence_sentiment_list





#%%
# ANALYSIS

# FakeNewsNet data set (BizzFeed and PolitiFact)
buzz_fake_news_path = 'News_Data\\FakeNewsNet-master\\Data\\BuzzFeed\\FakeNewsContent\\*.json'
buzz_fact_news_path = 'News_Data\\FakeNewsNet-master\\Data\\BuzzFeed\\RealNewsContent\\*.json'
poli_fake_news_path = 'News_Data\\FakeNewsNet-master\\Data\\PolitiFact\\FakeNewsContent\\*.json'
poli_fact_news_path = 'News_Data\\FakeNewsNet-master\\Data\\PolitiFact\\RealNewsContent\\*.json'
buzz_fake_sentences, buzz_fake_sentence_sentiments, buzz_fake_article_sentiments = politi_buzz_analysis(buzz_fake_news_path)
buzz_real_sentences, buzz_real_sentence_sentiments, buzz_real_article_sentiments = politi_buzz_analysis(buzz_fact_news_path)
poli_fake_sentences, poli_fake_sentence_sentiments, poli_fake_article_sentiments = politi_buzz_analysis(poli_fake_news_path)
poli_real_sentences, poli_real_sentence_sentiments, poli_real_article_sentiments = politi_buzz_analysis(poli_fact_news_path)

# Kaggle real and fake data set
kagg_news_path = 'News_Data\\reliable-nonreliable-news-kaggle\\train.csv'
kagg_fake_article_sentiments, kagg_fake_sentences, kagg_fake_sentence_sentiments, kagg_real_article_sentiments, kagg_real_sentences, kagg_real_sentence_sentiments = kaggel_Fact_Fake_analysis(kagg_news_path)

# Kraggek fake data set
#kagg_fake_news_path = 'News_Data\\fake-news-kaggle\\fake.csv'
#kagg_fake_article_sentiments, kagg_fake_sentence_sentiments = kaggel_Fake_analysis(kagg_fake_news_path)

buzz_fake_flat_list = [item for sublist in buzz_fake_sentence_sentiments for item in sublist]
buzz_real_flat_list = [item for sublist in buzz_real_sentence_sentiments for item in sublist]
poli_fake_flat_list = [item for sublist in poli_fake_sentence_sentiments for item in sublist]
poli_real_flat_list = [item for sublist in poli_real_sentence_sentiments for item in sublist]
kagg_fake_flat_list = [item for sublist in kagg_fake_sentence_sentiments for item in sublist]
kagg_real_flat_list = [item for sublist in kagg_real_sentence_sentiments for item in sublist]



#%%
# mean and varience calculations
buzz_fake_article_sentiments, fake_art_avg_buzz, fake_art_var_buzz = avg_var_calculation(buzz_fake_article_sentiments, "fake", "(buzz)")
buzz_real_article_sentiments, real_art_avg_buzz, real_art_var_buzz = avg_var_calculation(buzz_real_article_sentiments, "real", "(buzz)")
poli_fake_article_sentiments, fake_art_avg_poli, fake_art_var_poli = avg_var_calculation(poli_fake_article_sentiments, "fake", "(poli)")
poli_real_article_sentiments, real_art_avg_poli, real_art_var_poli = avg_var_calculation(poli_real_article_sentiments, "real", "(poli)")
kagg_fake_article_sentiments, fake_art_avg_kagg, fake_art_var_kagg = avg_var_calculation(kagg_fake_article_sentiments, "fake", "(kagg)")
kagg_real_article_sentiments, real_art_avg_kagg, real_art_var_kagg = avg_var_calculation(kagg_real_article_sentiments, "real", "(kagg)")




#%%
# PLOTS

# scatter plots: (of article average sentiments)
plt.figure("BuzzFeed article sentiments")
plt.plot(np.arange(1, len(buzz_fake_article_sentiments)+1), buzz_fake_article_sentiments, 'r*')
plt.plot(np.arange(1, len(buzz_real_article_sentiments)+1), buzz_real_article_sentiments, 'g*')
plt.title('BuzzFeed article sentiments')
plt.xlabel('Article index')
plt.ylabel('Average Sentiment Intensity')

plt.figure("PolitiFact article sentiments")
plt.plot(np.arange(1, len(poli_fake_article_sentiments)+1), poli_fake_article_sentiments, 'rs')
plt.plot(np.arange(1, len(poli_real_article_sentiments)+1), poli_real_article_sentiments, 'gs')
plt.title('PolitiFact article sentiments')
plt.xlabel('Article index')
plt.ylabel('Average Sentiment Intensity')

plt.figure("Kaggel article sentiments")
plt.plot(np.arange(1, len(kagg_fake_article_sentiments)+1), kagg_fake_article_sentiments, 'r^')
plt.plot(np.arange(1, len(kagg_real_article_sentiments)+1), kagg_real_article_sentiments, 'g^')
plt.title('Kaggel article sentiments')
plt.xlabel('Article index')
plt.ylabel('Average Sentiment Intensity')

plt.figure("Article sentiments")
plt.plot(np.arange(1, len(buzz_fake_article_sentiments)+1), buzz_fake_article_sentiments, 'r*')
plt.plot(np.arange(1, len(buzz_real_article_sentiments)+1), buzz_real_article_sentiments, 'g*')
plt.plot(np.arange(1, len(poli_fake_article_sentiments)+1), poli_fake_article_sentiments, 'rs')
plt.plot(np.arange(1, len(poli_real_article_sentiments)+1), poli_real_article_sentiments, 'gs')
plt.plot(np.arange(1, len(kagg_fake_article_sentiments)+1), kagg_fake_article_sentiments, 'r^')
plt.plot(np.arange(1, len(kagg_real_article_sentiments)+1), kagg_real_article_sentiments, 'g^')
plt.title('Article sentiments')
plt.xlabel('Article index')
plt.ylabel('Average Sentiment Intensity')

#%%
# histogram plots: (of article average sentiments)

bin_num = 'auto' # np.linspace(-4, 4, 50)

plt.figure("BuzzFeed fact and fake articles")
plt.hist(buzz_real_article_sentiments, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Real')
plt.hist(buzz_fake_article_sentiments, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Fake')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intencity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('BuzzFeed fact and fake articles')

#plt.figure("BuzzFeed KDE: Fact vs. Fake (articles)")
df_news = pandas.DataFrame({'Fake': buzz_fake_article_sentiments, 'Real': buzz_real_article_sentiments})
df_news.plot.kde(title='BuzzFeed KDE (articles)')
plt.xlabel('Sentiment intencity')
plt.ylabel('Probability')

plt.figure("PloitiFact fact and fake articles")
plt.hist(poli_fake_article_sentiments, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Fake')
plt.hist(poli_real_article_sentiments, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Real')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intencity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('PloitiFact fact and fake articles')

#plt.figure("PolitiFact KDE: Fact vs. Fake")
df_news_real = pandas.DataFrame({'Real': poli_real_article_sentiments})
df_news_fake = pandas.DataFrame({'Fake': poli_fake_article_sentiments})
df_news = pandas.concat([df_news_real,df_news_fake], axis=1)
df_news.plot.kde(title='PolitiFact KDE (articles)')
plt.xlabel('Sentiment intencity')
plt.ylabel('Probability')

plt.figure("Kaggel fact and fake articles")
plt.hist(kagg_fake_article_sentiments, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Fake')
plt.hist(kagg_real_article_sentiments, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Real')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intencity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('Kaggel fact and fake articles')

#plt.figure("Kaggel KDE: Fact vs. Fake")
df_news_real = pandas.DataFrame({'Real': kagg_real_article_sentiments})
df_news_fake = pandas.DataFrame({'Fake': kagg_fake_article_sentiments})
df_news = pandas.concat([df_news_real,df_news_fake], axis=1)
df_news.plot.kde(title='Kaggel KDE (articles)')
plt.xlabel('Sentiment intencity')
plt.ylabel('Probability')

#%%

# histogram plots: (of sentence sentiments)

plt.figure("BuzzFeed senctence sentiments") # plot of sentiment for each sentence
plt.hist(buzz_fake_flat_list, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Fake')
plt.hist(buzz_real_flat_list, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Real')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intencity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('BuzzFeed senctence sentiments')

#plt.figure("BuzzFeed KDE: Fact vs. Fake (sentences)")
df_news_real = pandas.DataFrame({'Real': buzz_real_flat_list})
df_news_fake = pandas.DataFrame({'Fake': buzz_fake_flat_list})
df_news = pandas.concat([df_news_real,df_news_fake], axis=1)
df_news.plot.kde(title='BuzzFeed KDE (sentences)')
plt.xlabel('Sentiment intencity')
plt.ylabel('Probability')

plt.figure("PolitiFact senctence sentiments") # plot of sentiment for each sentence
plt.hist(poli_fake_flat_list, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Fake')
plt.hist(poli_real_flat_list, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Real')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intencity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('PolitiFact senctence sentiments')

#plt.figure("PolitiFact KDE: Fact vs. Fake (sentences)")
df_news_real = pandas.DataFrame({'Real': poli_real_flat_list})
df_news_fake = pandas.DataFrame({'Fake': poli_fake_flat_list})
df_news = pandas.concat([df_news_real,df_news_fake], axis=1)
df_news.plot.kde(title='PolitiFact KDE (sentences)')
plt.xlabel('Sentiment intencity')
plt.ylabel('Probability')

plt.figure("Kaggel senctence sentiments") # plot of sentiment for each sentence
plt.hist(kagg_fake_flat_list, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Fake')
plt.hist(kagg_real_flat_list, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Real')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intencity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('Kaggel senctence sentiments')

#plt.figure("Kaggel KDE: Fact vs. Fake (sentences)")
df_news_real = pandas.DataFrame({'Real': kagg_real_flat_list})
df_news_fake = pandas.DataFrame({'Fake': kagg_fake_flat_list})
df_news = pandas.concat([df_news_real,df_news_fake], axis=1)
df_news.plot.kde(title='Kaggel KDE (sentences)')
plt.xlabel('Sentiment intencity')
plt.ylabel('Probability')

plt.show()