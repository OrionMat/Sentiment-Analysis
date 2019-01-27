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

# BBC data set
BBC_busin_news_path = 'News_Data\\bbc-articles\\bbc\\business\\*.txt'
BBC_enter_news_path = 'News_Data\\bbc-articles\\bbc\\entertainment\\*.txt'
BBC_polit_news_path = 'News_Data\\bbc-articles\\bbc\\politics\\*.txt'
BBC_sport_news_path = 'News_Data\\bbc-articles\\bbc\\sport\\*.txt'
BBC_techn_news_path = 'News_Data\\bbc-articles\\bbc\\tech\\*.txt'
BBC_busin_sentences, BBC_busin_sentence_sentiments, BBC_busin_article_sentiments  = BBC_analysis(BBC_busin_news_path)
BBC_enter_sentences, BBC_enter_sentence_sentiments, BBC_enter_article_sentiments  = BBC_analysis(BBC_enter_news_path)
BBC_polit_sentences, BBC_polit_sentence_sentiments, BBC_polit_article_sentiments  = BBC_analysis(BBC_polit_news_path)
BBC_sport_sentences, BBC_sport_sentence_sentiments, BBC_sport_article_sentiments  = BBC_analysis(BBC_sport_news_path)
BBC_techn_sentences, BBC_techn_sentence_sentiments, BBC_techn_article_sentiments  = BBC_analysis(BBC_techn_news_path)

buzz_fake_flat_list = [item for sublist in buzz_fake_sentence_sentiments for item in sublist]
buzz_real_flat_list = [item for sublist in buzz_real_sentence_sentiments for item in sublist]
poli_fake_flat_list = [item for sublist in poli_fake_sentence_sentiments for item in sublist]
poli_real_flat_list = [item for sublist in poli_real_sentence_sentiments for item in sublist]
kagg_fake_flat_list = [item for sublist in kagg_fake_sentence_sentiments for item in sublist]
kagg_real_flat_list = [item for sublist in kagg_real_sentence_sentiments for item in sublist]
BBC_busin_flat_list = [item for sublist in BBC_busin_sentence_sentiments for item in sublist]
BBC_enter_flat_list = [item for sublist in BBC_enter_sentence_sentiments for item in sublist]
BBC_polit_flat_list = [item for sublist in BBC_polit_sentence_sentiments for item in sublist]
BBC_sport_flat_list = [item for sublist in BBC_sport_sentence_sentiments for item in sublist]
BBC_techn_flat_list = [item for sublist in BBC_techn_sentence_sentiments for item in sublist]

#%%
# mean and varience calculations
# sentiment lists to arrays
buzz_fake_article_sentiments, fake_art_avg_buzz, fake_art_var_buzz = avg_var_calculation(buzz_fake_article_sentiments, "fake", "(buzz)")
buzz_real_article_sentiments, real_art_avg_buzz, real_art_var_buzz = avg_var_calculation(buzz_real_article_sentiments, "real", "(buzz)")
poli_fake_article_sentiments, fake_art_avg_poli, fake_art_var_poli = avg_var_calculation(poli_fake_article_sentiments, "fake", "(poli)")
poli_real_article_sentiments, real_art_avg_poli, real_art_var_poli = avg_var_calculation(poli_real_article_sentiments, "real", "(poli)")
kagg_fake_article_sentiments, fake_art_avg_kagg, fake_art_var_kagg = avg_var_calculation(kagg_fake_article_sentiments, "fake", "(kagg)")
kagg_real_article_sentiments, real_art_avg_kagg, real_art_var_kagg = avg_var_calculation(kagg_real_article_sentiments, "real", "(kagg)")
BBC_busin_article_sentiments, busin_art_avg_BBC, busin_art_var_BBC = avg_var_calculation(BBC_busin_article_sentiments, "business", "(BBC)")
BBC_enter_article_sentiments, enter_art_avg_BBC, enter_art_var_BBC = avg_var_calculation(BBC_enter_article_sentiments, "entertainment", "(BBC)")
BBC_polit_article_sentiments, polit_art_avg_BBC, polit_art_var_BBC = avg_var_calculation(BBC_polit_article_sentiments, "politics", "(BBC)")
BBC_sport_article_sentiments, sport_art_avg_BBC, sport_art_var_BBC = avg_var_calculation(BBC_sport_article_sentiments, "sports", "(BBC)")
BBC_techn_article_sentiments, techn_art_avg_BBC, techn_art_var_BBC = avg_var_calculation(BBC_techn_article_sentiments, "technology", "(BBC)")

all_fake_sentiments = np.concatenate((buzz_fake_article_sentiments, poli_fake_article_sentiments, kagg_fake_article_sentiments), axis=None) # move this and do more with it
all_real_sentiments = np.concatenate((buzz_real_article_sentiments, poli_real_article_sentiments, kagg_real_article_sentiments), axis=None)



#%%
# PLOTS

# scatter plots: (of article average sentiments)
plt.figure("BuzzFeed article sentiments")
plt.plot(np.arange(1, len(buzz_fake_article_sentiments)+1), buzz_fake_article_sentiments, 'r*', label='Fake')
plt.plot(np.arange(1, len(buzz_real_article_sentiments)+1), buzz_real_article_sentiments, 'g*', label='Fact')
plt.axis([0, 100, -4, 4])
plt.legend(loc='upper right')
plt.title('BuzzFeed article sentiments')
plt.xlabel('Article index')
plt.ylabel('Average Sentiment Intensity')
plt.savefig("C:\\Users\\Orion\\Documents\\GitHub\\Sentiment-Analysis\\figures\\BuzzFeed_scatter.eps", dpi=1000, format='eps')

plt.figure("PolitiFact article sentiments")
plt.plot(np.arange(1, len(poli_fake_article_sentiments)+1), poli_fake_article_sentiments, 'rx', label='Fake')
plt.plot(np.arange(1, len(poli_real_article_sentiments)+1), poli_real_article_sentiments, 'gx', label='Fact')
plt.axis([0, 130, -4, 4])
plt.legend(loc='upper right')
plt.title('PolitiFact article sentiments')
plt.xlabel('Article index')
plt.ylabel('Average Sentiment Intensity')
plt.savefig("C:\\Users\\Orion\\Documents\\GitHub\\Sentiment-Analysis\\figures\\PolitiFact_scatter.eps", dpi=1000, format='eps')

plt.figure("Kaggel article sentiments")
plt.plot(np.arange(1, len(kagg_fake_article_sentiments)+1), kagg_fake_article_sentiments, 'r.', label='Fake')
plt.plot(np.arange(1, len(kagg_real_article_sentiments)+1), kagg_real_article_sentiments, 'g.', label='Fact')
plt.axis([0, 110, -4, 4])
plt.legend(loc='upper right')
plt.title('Kaggel article sentiments')
plt.xlabel('Article index')
plt.ylabel('Average Sentiment Intensity')
plt.savefig("C:\\Users\\Orion\\Documents\\GitHub\\Sentiment-Analysis\\figures\\Kaggel_scatter.eps", dpi=1000, format='eps')

plt.figure("Article sentiments")
plt.plot(np.arange(1, len(buzz_fake_article_sentiments)+1), buzz_fake_article_sentiments, 'r*', label='BuzzFeed Fake')
plt.plot(np.arange(1, len(buzz_real_article_sentiments)+1), buzz_real_article_sentiments, 'g*', label='BuzzFeed Fact')
plt.plot(np.arange(1, len(poli_fake_article_sentiments)+1), poli_fake_article_sentiments, 'rx', label='PolitiFact Fake')
plt.plot(np.arange(1, len(poli_real_article_sentiments)+1), poli_real_article_sentiments, 'gx', label='PolitiFact Fact')
plt.plot(np.arange(1, len(kagg_fake_article_sentiments)+1), kagg_fake_article_sentiments, 'r.', label='Kaggel Fake')
plt.plot(np.arange(1, len(kagg_real_article_sentiments)+1), kagg_real_article_sentiments, 'g.', label='Kaggel Fact')
plt.axis([0, 130, -4, 4])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Article sentiments')
plt.xlabel('Article index')
plt.ylabel('Average Sentiment Intensity')
plt.savefig("C:\\Users\\Orion\\Documents\\GitHub\\Sentiment-Analysis\\figures\\All_scatter.eps", dpi=1000, format='eps')

plt.figure("BBC article sentiments")
plt.plot(np.arange(1, len(BBC_busin_article_sentiments)+1), BBC_busin_article_sentiments, 'r.', label='Business')
plt.plot(np.arange(1, len(BBC_enter_article_sentiments)+1), BBC_enter_article_sentiments, 'g.', label='Entertainment')
plt.plot(np.arange(1, len(BBC_polit_article_sentiments)+1), BBC_polit_article_sentiments, 'b.', label='Politics')
plt.plot(np.arange(1, len(BBC_sport_article_sentiments)+1), BBC_sport_article_sentiments, 'y.', label='Sport')
plt.plot(np.arange(1, len(BBC_techn_article_sentiments)+1), BBC_techn_article_sentiments, 'm.', label='Technology')
plt.axis([0, 520, -4, 4])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('BBC article sentiments')
plt.xlabel('Article index')
plt.ylabel('Average Sentiment Intensity')
plt.savefig("C:\\Users\\Orion\\Documents\\GitHub\\Sentiment-Analysis\\figures\\BBC_scatter.eps", dpi=1000, format='eps')

#%%
# histogram plots: (of article average sentiments)

bin_num = 'auto' # np.linspace(-4, 4, 50)

plt.figure("BuzzFeed fact and fake articles")
plt.hist(buzz_real_article_sentiments, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Fact')
plt.hist(buzz_fake_article_sentiments, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Fake')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intencity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('BuzzFeed fact and fake articles')
plt.savefig("C:\\Users\\Orion\\Documents\\GitHub\\Sentiment-Analysis\\figures\\BuzzFeed_hist.eps", dpi=1000, format='eps')

#plt.figure("BuzzFeed KDE: Fact vs. Fake (articles)")
df_news_buzz = pandas.DataFrame({'Fake': buzz_fake_article_sentiments, 'Fact': buzz_real_article_sentiments})
df_news_buzz.plot.kde(title='BuzzFeed KDE (articles)')
plt.xlabel('Sentiment intencity')
plt.ylabel('Probability')
plt.savefig("C:\\Users\\Orion\\Documents\\GitHub\\Sentiment-Analysis\\figures\\BuzzFeed_KDE.eps", dpi=1000, format='eps')

plt.figure("PloitiFact fact and fake articles")
plt.hist(poli_fake_article_sentiments, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Fake')
plt.hist(poli_real_article_sentiments, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Fact')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intencity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('PloitiFact fact and fake articles')
plt.savefig("C:\\Users\\Orion\\Documents\\GitHub\\Sentiment-Analysis\\figures\\PolitiFact_hist.eps", dpi=1000, format='eps')

#plt.figure("PolitiFact KDE: Fact vs. Fake")
df_news_real = pandas.DataFrame({'Fact': poli_real_article_sentiments})
df_news_fake = pandas.DataFrame({'Fake': poli_fake_article_sentiments})
df_news_poli = pandas.concat([df_news_fake,df_news_real], axis=1)
df_news_poli.plot.kde(title='PolitiFact KDE (articles)')
plt.xlabel('Sentiment intencity')
plt.ylabel('Probability')
plt.savefig("C:\\Users\\Orion\\Documents\\GitHub\\Sentiment-Analysis\\figures\\PolitiFact_KDE.eps", dpi=1000, format='eps')

plt.figure("Kaggel fact and fake articles")
plt.hist(kagg_fake_article_sentiments, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Fake')
plt.hist(kagg_real_article_sentiments, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Fact')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intencity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('Kaggel fact and fake articles')
plt.savefig("C:\\Users\\Orion\\Documents\\GitHub\\Sentiment-Analysis\\figures\\Kaggel_hist.eps", dpi=1000, format='eps')

#plt.figure("Kaggel KDE: Fact vs. Fake")
df_news_real = pandas.DataFrame({'Fact': kagg_real_article_sentiments})
df_news_fake = pandas.DataFrame({'Fake': kagg_fake_article_sentiments})
df_news_kagg = pandas.concat([df_news_fake,df_news_real], axis=1)
df_news_kagg.plot.kde(title='Kaggel KDE (articles)')
plt.xlabel('Sentiment intencity')
plt.ylabel('Probability')
plt.savefig("C:\\Users\\Orion\\Documents\\GitHub\\Sentiment-Analysis\\figures\\Kaggel_KDE.eps", dpi=1000, format='eps')

plt.figure("BBC articles")
plt.hist(BBC_busin_article_sentiments, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Business')
plt.hist(BBC_enter_article_sentiments, bins=bin_num, color='g', alpha=0.7, rwidth=0.85, label='Entertainment')
plt.hist(BBC_polit_article_sentiments, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Ploitics')
plt.hist(BBC_sport_article_sentiments, bins=bin_num, color='y', alpha=0.7, rwidth=0.85, label='Sport')
plt.hist(BBC_techn_article_sentiments, bins=bin_num, color='m', alpha=0.7, rwidth=0.85, label='Technology')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intencity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('BBC article sentiment histogram')
plt.savefig("C:\\Users\\Orion\\Documents\\GitHub\\Sentiment-Analysis\\figures\\BBC_hist.eps", dpi=1000, format='eps')

#plt.figure("BBC KDE")
df_news_busin = pandas.DataFrame({'Business': BBC_busin_article_sentiments})
df_news_enter = pandas.DataFrame({'Entertainment': BBC_enter_article_sentiments})
df_news_polit = pandas.DataFrame({'Politics': BBC_polit_article_sentiments})
df_news_sport = pandas.DataFrame({'Sport': BBC_sport_article_sentiments})
df_news_techn = pandas.DataFrame({'Technology': BBC_techn_article_sentiments})
df_news = pandas.concat([df_news_busin, df_news_enter, df_news_polit, df_news_sport, df_news_techn], axis=1)
df_news.plot.kde(title='BBC articles kernel density estimation (KDE)')
plt.xlabel('Sentiment intencity')
plt.ylabel('Probability')
plt.savefig("C:\\Users\\Orion\\Documents\\GitHub\\Sentiment-Analysis\\figures\\BBC_KDE.eps", dpi=1000, format='eps')

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
plt.savefig("C:\\Users\\Orion\\Documents\\GitHub\\Sentiment-Analysis\\figures\\BuzzFeed_hist_sent.eps", dpi=1000, format='eps')

#plt.figure("BuzzFeed KDE: Fact vs. Fake (sentences)")
df_news_real = pandas.DataFrame({'Real': buzz_real_flat_list})
df_news_fake = pandas.DataFrame({'Fake': buzz_fake_flat_list})
df_news = pandas.concat([df_news_real,df_news_fake], axis=1)
df_news.plot.kde(title='BuzzFeed KDE (sentences)')
plt.xlabel('Sentiment intencity')
plt.ylabel('Probability')
plt.savefig("C:\\Users\\Orion\\Documents\\GitHub\\Sentiment-Analysis\\figures\\BuzzFeed_KDE_sent.eps", dpi=1000, format='eps')

plt.figure("PolitiFact senctence sentiments") # plot of sentiment for each sentence
plt.hist(poli_fake_flat_list, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Fake')
plt.hist(poli_real_flat_list, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Real')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intencity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('PolitiFact senctence sentiments')
plt.savefig("C:\\Users\\Orion\\Documents\\GitHub\\Sentiment-Analysis\\figures\\PolitiFact_hist_sent.eps", dpi=1000, format='eps')

#plt.figure("PolitiFact KDE: Fact vs. Fake (sentences)")
df_news_real = pandas.DataFrame({'Real': poli_real_flat_list})
df_news_fake = pandas.DataFrame({'Fake': poli_fake_flat_list})
df_news = pandas.concat([df_news_real,df_news_fake], axis=1)
df_news.plot.kde(title='PolitiFact KDE (sentences)')
plt.xlabel('Sentiment intencity')
plt.ylabel('Probability')
plt.savefig("C:\\Users\\Orion\\Documents\\GitHub\\Sentiment-Analysis\\figures\\PolitiFact_KDE_sent.eps", dpi=1000, format='eps')

plt.figure("Kaggel senctence sentiments") # plot of sentiment for each sentence
plt.hist(kagg_fake_flat_list, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Fake')
plt.hist(kagg_real_flat_list, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Real')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intencity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('Kaggel senctence sentiments')
plt.savefig("C:\\Users\\Orion\\Documents\\GitHub\\Sentiment-Analysis\\figures\\Kaggel_hist_sent.eps", dpi=1000, format='eps')

#plt.figure("Kaggel KDE: Fact vs. Fake (sentences)")
df_news_real = pandas.DataFrame({'Real': kagg_real_flat_list})
df_news_fake = pandas.DataFrame({'Fake': kagg_fake_flat_list})
df_news = pandas.concat([df_news_real,df_news_fake], axis=1)
df_news.plot.kde(title='Kaggel KDE (sentences)')
plt.xlabel('Sentiment intencity')
plt.ylabel('Probability')
plt.savefig("C:\\Users\\Orion\\Documents\\GitHub\\Sentiment-Analysis\\figures\\Kaggel_KDE_sent.eps", dpi=1000, format='eps')

plt.figure("BBC senctence sentiments") # plot of sentiment for each sentence
plt.hist(BBC_busin_flat_list, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Business')
plt.hist(BBC_enter_flat_list, bins=bin_num, color='g', alpha=0.7, rwidth=0.85, label='Entertainment')
plt.hist(BBC_polit_flat_list, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Politics')
plt.hist(BBC_sport_flat_list, bins=bin_num, color='y', alpha=0.7, rwidth=0.85, label='Sport')
plt.hist(BBC_techn_flat_list, bins=bin_num, color='m', alpha=0.7, rwidth=0.85, label='Technology')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intencity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('BBC senctence sentiments')
plt.savefig("C:\\Users\\Orion\\Documents\\GitHub\\Sentiment-Analysis\\figures\\BBC_hist_sent.eps", dpi=1000, format='eps')

#plt.figure("BBC KDE: Fact vs. polit (sentences)")
df_news_busin = pandas.DataFrame({'Business': BBC_busin_flat_list})
df_news_enter = pandas.DataFrame({'Entertainment': BBC_enter_flat_list})
df_news_polit = pandas.DataFrame({'Politics': BBC_polit_flat_list})
df_news_sport = pandas.DataFrame({'Sport': BBC_sport_flat_list})
df_news_techn = pandas.DataFrame({'Technology': BBC_techn_flat_list})
df_news = pandas.concat([df_news_busin, df_news_enter, df_news_polit, df_news_sport, df_news_techn], axis=1)
df_news.plot.kde(title='BBC KDE (sentences)')
plt.xlabel('Sentiment intencity')
plt.ylabel('Probability')
plt.savefig("C:\\Users\\Orion\\Documents\\GitHub\\Sentiment-Analysis\\figures\\BBC_KDE_sent.eps", dpi=1000, format='eps')


# SUB-PLOTS
#%%

# Scatter plots
plt.subplot(2, 2, 1)
plt.plot(np.arange(1, len(buzz_fake_article_sentiments)+1), buzz_fake_article_sentiments, 'r*', label='Fake')
plt.plot(np.arange(1, len(buzz_real_article_sentiments)+1), buzz_real_article_sentiments, 'g*', label='Fact')
plt.axis([0, 100, -4, 4])
plt.legend(loc='upper right')
plt.title('BuzzFeed article sentiments')
plt.xlabel('Article index')
plt.ylabel('Average Sentiment Intensity')

plt.subplot(2, 2, 2)
plt.plot(np.arange(1, len(poli_fake_article_sentiments)+1), poli_fake_article_sentiments, 'rx', label='Fake')
plt.plot(np.arange(1, len(poli_real_article_sentiments)+1), poli_real_article_sentiments, 'gx', label='Fact')
plt.axis([0, 130, -4, 4])
plt.legend(loc='upper right')
plt.title('PolitiFact article sentiments')
plt.xlabel('Article index')
plt.ylabel('Average Sentiment Intensity')

plt.subplot(2, 2, 3)
plt.plot(np.arange(1, len(kagg_fake_article_sentiments)+1), kagg_fake_article_sentiments, 'r.', label='Fake')
plt.plot(np.arange(1, len(kagg_real_article_sentiments)+1), kagg_real_article_sentiments, 'g.', label='Fact')
plt.axis([0, 110, -4, 4])
plt.legend(loc='upper right')
plt.title('Kaggel article sentiments')
plt.xlabel('Article index')
plt.ylabel('Average Sentiment Intensity')

plt.subplot(2, 2, 4)
plt.plot(np.arange(1, len(buzz_fake_article_sentiments)+1), buzz_fake_article_sentiments, 'r*', label='BuzzFeed Fake')
plt.plot(np.arange(1, len(buzz_real_article_sentiments)+1), buzz_real_article_sentiments, 'g*', label='BuzzFeed Fact')
plt.plot(np.arange(1, len(poli_fake_article_sentiments)+1), poli_fake_article_sentiments, 'rx', label='PolitiFact Fake')
plt.plot(np.arange(1, len(poli_real_article_sentiments)+1), poli_real_article_sentiments, 'gx', label='PolitiFact Fact')
plt.plot(np.arange(1, len(kagg_fake_article_sentiments)+1), kagg_fake_article_sentiments, 'r.', label='Kaggel Fake')
plt.plot(np.arange(1, len(kagg_real_article_sentiments)+1), kagg_real_article_sentiments, 'g.', label='Kaggel Fact')
plt.axis([0, 130, -4, 4])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('All article sentiments')
plt.xlabel('Article index')
plt.ylabel('Average Sentiment Intensity')

# Histograms
plt.figure()

plt.subplot(2, 2, 1)
plt.hist(buzz_fake_article_sentiments, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Fake')
plt.hist(buzz_real_article_sentiments, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Fact')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intencity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('BuzzFeed fact and fake articles')

plt.subplot(2, 2, 2)
plt.hist(poli_fake_article_sentiments, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Fake')
plt.hist(poli_real_article_sentiments, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Fact')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intencity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('PloitiFact fact and fake articles')

plt.subplot(2, 2, 3)
plt.hist(kagg_fake_article_sentiments, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Fake')
plt.hist(kagg_real_article_sentiments, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Fact')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intencity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('Kaggel fact and fake articles')

plt.subplot(2, 2, 4)
plt.hist(all_fake_sentiments, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Fake')
plt.hist(all_real_sentiments, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Fact')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intencity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('All fact and fake articles')

# KDEs
fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0,0].set_xlabel('Sentiment Intensity')
axes[0,1].set_xlabel('Sentiment Intensity')
axes[1,0].set_xlabel('Sentiment Intensity')
axes[1,1].set_xlabel('Sentiment Intensity')

df_news_buzz.plot.kde(title='BuzzFeed KDE (articles)', ax=axes[0,0], color='rb')

df_news_poli.plot.kde(title='PolitiFact KDE (articles)', ax=axes[0,1], color='rb')

df_news_kagg.plot.kde(title='Kaggel KDE (articles)', ax=axes[1,0], color='rb')

df_news_real = pandas.DataFrame({'Fact': all_real_sentiments})
df_news_fake = pandas.DataFrame({'Fake': all_fake_sentiments})
df_news_all = pandas.concat([df_news_fake,df_news_real], axis=1)
df_news_all.plot.kde(title='All KDE (articles)', ax=axes[1,1], color='rb')




plt.show()