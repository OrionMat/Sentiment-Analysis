# BBC and FakeNewsNet sentiment analysis and plots
#%%
import newsSentimentAnalysis as NSA
import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy import stats

#%%
# ANALYSIS

# FakeNewsNet data set (BizzFeed and PolitiFact)
buzz_fake_news_path = 'News_Data\\FakeNewsNet-master\\Data\\BuzzFeed\\FakeNewsContent\\*.json'
buzz_fact_news_path = 'News_Data\\FakeNewsNet-master\\Data\\BuzzFeed\\RealNewsContent\\*.json'
poli_fake_news_path = 'News_Data\\FakeNewsNet-master\\Data\\PolitiFact\\FakeNewsContent\\*.json'
poli_fact_news_path = 'News_Data\\FakeNewsNet-master\\Data\\PolitiFact\\RealNewsContent\\*.json'
buzz_fake_sentences, buzz_fake_sentence_sentiments, buzz_fake_article_sentiments = NSA.politi_buzz_analysis(buzz_fake_news_path)
buzz_real_sentences, buzz_real_sentence_sentiments, buzz_real_article_sentiments = NSA.politi_buzz_analysis(buzz_fact_news_path)
poli_fake_sentences, poli_fake_sentence_sentiments, poli_fake_article_sentiments = NSA.politi_buzz_analysis(poli_fake_news_path)
poli_real_sentences, poli_real_sentence_sentiments, poli_real_article_sentiments = NSA.politi_buzz_analysis(poli_fact_news_path)

# Kaggle real and fake data set
kagg_news_path = 'News_Data\\reliable-nonreliable-news-kaggle\\train.csv'
kagg_fake_article_sentiments, kagg_fake_sentences, kagg_fake_sentence_sentiments, kagg_real_article_sentiments, kagg_real_sentences, kagg_real_sentence_sentiments = NSA.kaggle_Fact_Fake_analysis(kagg_news_path)

# Kraggek fake data set
#kagg_fake_news_path = 'News_Data\\fake-news-kaggle\\fake.csv'
#kagg_fake_article_sentiments, kagg_fake_sentence_sentiments = Kaggle_Fake_analysis(kagg_fake_news_path)

# BBC data set
BBC_busin_news_path = 'News_Data\\bbc-articles\\bbc\\business\\*.txt'
BBC_enter_news_path = 'News_Data\\bbc-articles\\bbc\\entertainment\\*.txt'
BBC_polit_news_path = 'News_Data\\bbc-articles\\bbc\\politics\\*.txt'
BBC_sport_news_path = 'News_Data\\bbc-articles\\bbc\\sport\\*.txt'
BBC_techn_news_path = 'News_Data\\bbc-articles\\bbc\\tech\\*.txt'
BBC_busin_sentences, BBC_busin_sentence_sentiments, BBC_busin_article_sentiments  = NSA.BBC_analysis(BBC_busin_news_path)
BBC_enter_sentences, BBC_enter_sentence_sentiments, BBC_enter_article_sentiments  = NSA.BBC_analysis(BBC_enter_news_path)
BBC_polit_sentences, BBC_polit_sentence_sentiments, BBC_polit_article_sentiments  = NSA.BBC_analysis(BBC_polit_news_path)
BBC_sport_sentences, BBC_sport_sentence_sentiments, BBC_sport_article_sentiments  = NSA.BBC_analysis(BBC_sport_news_path)
BBC_techn_sentences, BBC_techn_sentence_sentiments, BBC_techn_article_sentiments  = NSA.BBC_analysis(BBC_techn_news_path)
#%%
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

# mean and varience calculations
# sentiment lists to arrays
buzz_fake_article_sentiments, fake_art_avg_buzz, fake_art_var_buzz = NSA.avg_var_calculation(buzz_fake_article_sentiments, "fake", "(buzz)")
buzz_real_article_sentiments, real_art_avg_buzz, real_art_var_buzz = NSA.avg_var_calculation(buzz_real_article_sentiments, "real", "(buzz)")
poli_fake_article_sentiments, fake_art_avg_poli, fake_art_var_poli = NSA.avg_var_calculation(poli_fake_article_sentiments, "fake", "(poli)")
poli_real_article_sentiments, real_art_avg_poli, real_art_var_poli = NSA.avg_var_calculation(poli_real_article_sentiments, "real", "(poli)")
kagg_fake_article_sentiments, fake_art_avg_kagg, fake_art_var_kagg = NSA.avg_var_calculation(kagg_fake_article_sentiments, "fake", "(kagg)")
kagg_real_article_sentiments, real_art_avg_kagg, real_art_var_kagg = NSA.avg_var_calculation(kagg_real_article_sentiments, "real", "(kagg)")
BBC_busin_article_sentiments, busin_art_avg_BBC, busin_art_var_BBC = NSA.avg_var_calculation(BBC_busin_article_sentiments, "business", "(BBC)")
BBC_enter_article_sentiments, enter_art_avg_BBC, enter_art_var_BBC = NSA.avg_var_calculation(BBC_enter_article_sentiments, "entertainment", "(BBC)")
BBC_polit_article_sentiments, polit_art_avg_BBC, polit_art_var_BBC = NSA.avg_var_calculation(BBC_polit_article_sentiments, "politics", "(BBC)")
BBC_sport_article_sentiments, sport_art_avg_BBC, sport_art_var_BBC = NSA.avg_var_calculation(BBC_sport_article_sentiments, "sports", "(BBC)")
BBC_techn_article_sentiments, techn_art_avg_BBC, techn_art_var_BBC = NSA.avg_var_calculation(BBC_techn_article_sentiments, "technology", "(BBC)")

all_fake_article_sentiments = np.concatenate((buzz_fake_article_sentiments, poli_fake_article_sentiments, kagg_fake_article_sentiments), axis=None)
all_real_article_sentiments = np.concatenate((buzz_real_article_sentiments, poli_real_article_sentiments, kagg_real_article_sentiments), axis=None)
all_fake_sentence_sentiments = np.concatenate((buzz_fake_flat_list, poli_fake_flat_list, kagg_fake_flat_list), axis=None)
all_real_sentence_sentiments = np.concatenate((buzz_real_flat_list, poli_real_flat_list, kagg_real_flat_list), axis=None)

buzz_fake_art_abs = NSA.cal_article_abs_sentiment(buzz_fake_sentence_sentiments)
buzz_real_art_abs = NSA.cal_article_abs_sentiment(buzz_real_sentence_sentiments)
poli_fake_art_abs = NSA.cal_article_abs_sentiment(poli_fake_sentence_sentiments)
poli_real_art_abs = NSA.cal_article_abs_sentiment(poli_real_sentence_sentiments)
kagg_fake_art_abs = NSA.cal_article_abs_sentiment(kagg_fake_sentence_sentiments)
kagg_real_art_abs = NSA.cal_article_abs_sentiment(kagg_real_sentence_sentiments)
BBC_busin_art_abs = NSA.cal_article_abs_sentiment(BBC_busin_sentence_sentiments)
BBC_enter_art_abs = NSA.cal_article_abs_sentiment(BBC_enter_sentence_sentiments)
BBC_polit_art_abs = NSA.cal_article_abs_sentiment(BBC_polit_sentence_sentiments)
BBC_sport_art_abs = NSA.cal_article_abs_sentiment(BBC_sport_sentence_sentiments)
BBC_techn_art_abs = NSA.cal_article_abs_sentiment(BBC_techn_sentence_sentiments)

all_fake_art_abs = np.concatenate((buzz_fake_art_abs, poli_fake_art_abs, kagg_fake_art_abs), axis=None)
all_real_art_abs = np.concatenate((buzz_real_art_abs, poli_real_art_abs, kagg_real_art_abs), axis=None)

KS_buzz_art = stats.ks_2samp(buzz_fake_art_abs, buzz_real_art_abs)
KS_poli_art = stats.ks_2samp(poli_fake_art_abs, poli_real_art_abs)
KS_kagg_art = stats.ks_2samp(kagg_fake_art_abs, kagg_real_art_abs)
KS_all_art = stats.ks_2samp(all_fake_art_abs, all_real_art_abs)


#KS_buzz_sent = stats.ks_2samp(buzz_fake_flat_list, buzz_real_flat_list)

# very strict test: find more suitable tests
# the lower the statistic value to more similar the distributions are
# the higher the p value the more similare the distributions are
print('\n')
print('buzz article abs K-S test:', KS_buzz_art, '\n')
print('poli article K-S test:', KS_poli_art, '\n')
print('kagg article K-S test:', KS_kagg_art, '\n')
print('All article K-S test:', KS_all_art, '\n')


# Article dataframes:
# BuzzFeed
df_news_buzz_article = pandas.DataFrame({'Fake': buzz_fake_article_sentiments, 'Fact': buzz_real_article_sentiments})
# PolitiFact
df_news_real = pandas.DataFrame({'Fact': poli_real_article_sentiments})
df_news_fake = pandas.DataFrame({'Fake': poli_fake_article_sentiments})
df_news_poli_article = pandas.concat([df_news_fake,df_news_real], axis=1)
# Kaggle
df_news_real = pandas.DataFrame({'Fact': kagg_real_article_sentiments})
df_news_fake = pandas.DataFrame({'Fake': kagg_fake_article_sentiments})
df_news_kagg_article = pandas.concat([df_news_fake,df_news_real], axis=1)
# BBC - avg
df_news_busin = pandas.DataFrame({'Business': BBC_busin_article_sentiments})
df_news_enter = pandas.DataFrame({'Entertainment': BBC_enter_article_sentiments})
df_news_polit = pandas.DataFrame({'Politics': BBC_polit_article_sentiments})
df_news_sport = pandas.DataFrame({'Sport': BBC_sport_article_sentiments})
df_news_techn = pandas.DataFrame({'Technology': BBC_techn_article_sentiments})
df_news_BBC_article = pandas.concat([df_news_busin, df_news_enter, df_news_polit, df_news_sport, df_news_techn], axis=1)
# BBC - abs
df_news_busin_abs = pandas.DataFrame({'Business': BBC_busin_art_abs})
df_news_enter_abs = pandas.DataFrame({'Entertainment': BBC_enter_art_abs})
df_news_polit_abs = pandas.DataFrame({'Politics': BBC_polit_art_abs})
df_news_sport_abs = pandas.DataFrame({'Sport': BBC_sport_art_abs})
df_news_techn_abs = pandas.DataFrame({'Technology': BBC_techn_art_abs})
df_news_BBC_article_abs = pandas.concat([df_news_busin_abs, df_news_enter_abs, df_news_polit_abs, df_news_sport_abs, df_news_techn_abs], axis=1)
# All
df_news_real = pandas.DataFrame({'Fact': all_real_article_sentiments})
df_news_fake = pandas.DataFrame({'Fake': all_fake_article_sentiments})
df_news_all_articles = pandas.concat([df_news_fake,df_news_real], axis=1)
# abs
df_news_fake = pandas.DataFrame({'Fake': buzz_fake_art_abs})
df_news_real = pandas.DataFrame({'real': buzz_real_art_abs})
df_news_buzz_art_abs = pandas.concat([df_news_fake,df_news_real], axis=1)
df_news_fake = pandas.DataFrame({'Fake': poli_fake_art_abs})
df_news_real = pandas.DataFrame({'real': poli_real_art_abs})
df_news_poli_art_abs = pandas.concat([df_news_fake,df_news_real], axis=1)
df_news_fake = pandas.DataFrame({'Fake': kagg_fake_art_abs})
df_news_real = pandas.DataFrame({'real': kagg_real_art_abs})
df_news_kagg_art_abs = pandas.concat([df_news_fake,df_news_real], axis=1)
df_news_fake = pandas.DataFrame({'Fake': all_fake_art_abs})
df_news_real = pandas.DataFrame({'real': all_real_art_abs})
df_news_all_art_abs = pandas.concat([df_news_fake, df_news_real], axis=1)


# Sentence dataframes:
# BuzzFeed
df_news_real = pandas.DataFrame({'Fact': buzz_real_flat_list})
df_news_fake = pandas.DataFrame({'Fake': buzz_fake_flat_list})
df_news_buzz_sentence = pandas.concat([df_news_fake,df_news_real], axis=1)
# PolitiFact
df_news_real = pandas.DataFrame({'Fact': poli_real_flat_list})
df_news_fake = pandas.DataFrame({'Fake': poli_fake_flat_list})
df_news_poli_sentence = pandas.concat([df_news_fake,df_news_real], axis=1)
# Kaggle
df_news_real = pandas.DataFrame({'Fact': kagg_real_flat_list})
df_news_fake = pandas.DataFrame({'Fake': kagg_fake_flat_list})
df_news_kagg_sentence = pandas.concat([df_news_fake,df_news_real], axis=1)
# BBC
df_news_busin = pandas.DataFrame({'Business': BBC_busin_flat_list})
df_news_enter = pandas.DataFrame({'Entertainment': BBC_enter_flat_list})
df_news_polit = pandas.DataFrame({'Politics': BBC_polit_flat_list})
df_news_sport = pandas.DataFrame({'Sport': BBC_sport_flat_list})
df_news_techn = pandas.DataFrame({'Technology': BBC_techn_flat_list})
df_news_BBC_sentence = pandas.concat([df_news_busin, df_news_enter, df_news_polit, df_news_sport, df_news_techn], axis=1)
# All
df_news_real = pandas.DataFrame({'Fact': all_real_sentence_sentiments})
df_news_fake = pandas.DataFrame({'Fake': all_fake_sentence_sentiments})
df_news_all_sentence = pandas.concat([df_news_fake,df_news_real], axis=1)

bin_num = 'auto' # np.linspace(-4, 4, 50)



# PLOTS


#%% Article:
# # scatter plots:
# BuzzFeed

plt.figure("BuzzFeed article sentiments")
plt.plot(np.arange(1, len(buzz_fake_article_sentiments)+1), buzz_fake_article_sentiments, 'r*', label='Fake')
plt.plot(np.arange(1, len(buzz_real_article_sentiments)+1), buzz_real_article_sentiments, 'g*', label='Fact')
plt.axis([0, 100, -4, 4])
plt.legend(loc='upper right')
plt.title('BuzzFeed article sentiments')
plt.xlabel('Article index')
plt.ylabel('Average Sentiment Intensity')
# PolitiFact
plt.figure("PolitiFact article sentiments")
plt.plot(np.arange(1, len(poli_fake_article_sentiments)+1), poli_fake_article_sentiments, 'rx', label='Fake')
plt.plot(np.arange(1, len(poli_real_article_sentiments)+1), poli_real_article_sentiments, 'gx', label='Fact')
plt.axis([0, 130, -4, 4])
plt.legend(loc='upper right')
plt.title('PolitiFact article sentiments')
plt.xlabel('Article index')
plt.ylabel('Average Sentiment Intensity')
# Kaggle
plt.figure("Kaggle article sentiments")
plt.plot(np.arange(1, len(kagg_fake_article_sentiments)+1), kagg_fake_article_sentiments, 'r.', label='Fake')
plt.plot(np.arange(1, len(kagg_real_article_sentiments)+1), kagg_real_article_sentiments, 'g.', label='Fact')
plt.axis([0, 110, -4, 4])
plt.legend(loc='upper right')
plt.title('Kaggle article sentiments')
plt.xlabel('Article index')
plt.ylabel('Average Sentiment Intensity')
# All
plt.figure("Article sentiments")
plt.plot(np.arange(1, len(buzz_fake_article_sentiments)+1), buzz_fake_article_sentiments, 'r*', label='BuzzFeed Fake')
plt.plot(np.arange(1, len(buzz_real_article_sentiments)+1), buzz_real_article_sentiments, 'g*', label='BuzzFeed Fact')
plt.plot(np.arange(1, len(poli_fake_article_sentiments)+1), poli_fake_article_sentiments, 'rx', label='PolitiFact Fake')
plt.plot(np.arange(1, len(poli_real_article_sentiments)+1), poli_real_article_sentiments, 'gx', label='PolitiFact Fact')
plt.plot(np.arange(1, len(kagg_fake_article_sentiments)+1), kagg_fake_article_sentiments, 'r.', label='Kaggle Fake')
plt.plot(np.arange(1, len(kagg_real_article_sentiments)+1), kagg_real_article_sentiments, 'g.', label='Kaggle Fact')
plt.axis([0, 130, -4, 4])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Article sentiments')
plt.xlabel('Article index')
plt.ylabel('Average Sentiment Intensity')
# BBC
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

# Histograms and KDE
# BuzzFeed hist
plt.figure("BuzzFeed fact and fake articles")
plt.hist(buzz_real_article_sentiments, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Fact')
plt.hist(buzz_fake_article_sentiments, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Fake')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intensity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('BuzzFeed fact and fake articles')
# BuzzFeed KDE
df_news_buzz_article.plot.kde(title='BuzzFeed KDE (articles)')
plt.xlabel('Sentiment intensity')
plt.ylabel('Density')
# PolitiFact hist
plt.figure("PloitiFact fact and fake articles")
plt.hist(poli_fake_article_sentiments, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Fake')
plt.hist(poli_real_article_sentiments, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Fact')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intensity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('PloitiFact fact and fake articles')
# PolitiFact KDE
df_news_poli_article.plot.kde(title='PolitiFact KDE (articles)')
plt.xlabel('Sentiment intensity')
plt.ylabel('Density')
# Kaggle hist
plt.figure("Kaggle fact and fake articles")
plt.hist(kagg_fake_article_sentiments, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Fake')
plt.hist(kagg_real_article_sentiments, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Fact')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intensity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('Kaggle fact and fake articles')
# Kaggle KDE
df_news_kagg_article.plot.kde(title='Kaggle KDE (articles)')
plt.xlabel('Sentiment intensity')
plt.ylabel('Density')
# BBC hist
plt.figure("BBC articles")
plt.hist(BBC_busin_article_sentiments, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Business')
plt.hist(BBC_enter_article_sentiments, bins=bin_num, color='g', alpha=0.7, rwidth=0.85, label='Entertainment')
plt.hist(BBC_polit_article_sentiments, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Ploitics')
plt.hist(BBC_sport_article_sentiments, bins=bin_num, color='y', alpha=0.7, rwidth=0.85, label='Sport')
plt.hist(BBC_techn_article_sentiments, bins=bin_num, color='m', alpha=0.7, rwidth=0.85, label='Technology')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intensity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('BBC article sentiment histogram')
# BBC KDE
df_news_BBC_article.plot.kde(title='BBC articles kernel density estimation (KDE)')
plt.xlabel('Sentiment intensity')
plt.ylabel('Density')
# BBC absolute article sentiment KDE
df_news_BBC_article_abs.plot.kde(title='BBC articles absolute sentiment intensity KDE')
plt.legend(loc='upper right')
plt.xlabel('Article absolute sentiment intensity')
plt.ylabel('Density')


#%% Sentences:

# Histograms and KDE
# BuzzFeed hist
plt.figure("BuzzFeed senctence sentiments")
plt.hist(buzz_fake_flat_list, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Fake')
plt.hist(buzz_real_flat_list, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Real')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intensity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('BuzzFeed senctence sentiments')
# BuzzFeed KDE
df_news_buzz_sentence.plot.kde(title='BuzzFeed KDE (sentences)')
plt.xlabel('Sentiment intensity')
plt.ylabel('Density')
# PolitiFact hist
plt.figure("PolitiFact senctence sentiments")
plt.hist(poli_fake_flat_list, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Fake')
plt.hist(poli_real_flat_list, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Real')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intensity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('PolitiFact senctence sentiments')
# PloitiFact KDE
df_news_poli_sentence.plot.kde(title='PolitiFact KDE (sentences)')
plt.xlabel('Sentiment intensity')
plt.ylabel('Density')
# Kaggle hist
plt.figure("Kaggle senctence sentiments")
plt.hist(kagg_fake_flat_list, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Fake')
plt.hist(kagg_real_flat_list, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Real')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intensity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('Kaggle senctence sentiments')
# Kaggle KDE
df_news_kagg_sentence.plot.kde(title='Kaggle KDE (sentences)')
plt.xlabel('Sentiment intensity')
plt.ylabel('Density')
# BBC hist
plt.figure("BBC senctence sentiments")
plt.hist(BBC_busin_flat_list, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Business')
plt.hist(BBC_enter_flat_list, bins=bin_num, color='g', alpha=0.7, rwidth=0.85, label='Entertainment')
plt.hist(BBC_polit_flat_list, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Politics')
plt.hist(BBC_sport_flat_list, bins=bin_num, color='y', alpha=0.7, rwidth=0.85, label='Sport')
plt.hist(BBC_techn_flat_list, bins=bin_num, color='m', alpha=0.7, rwidth=0.85, label='Technology')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intensity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('BBC senctence sentiments')
# BBC KDE
df_news_BBC_sentence.plot.kde(title='BBC KDE (sentences)')
plt.xlabel('Sentiment intensity')
plt.ylabel('Density')



# SUB-PLOTS

#%% Articles:

# Scatter plots:
# BuzzFeed
plt.subplot(2, 2, 1)
plt.plot(np.arange(1, len(buzz_fake_article_sentiments)+1), buzz_fake_article_sentiments, 'r*', label='Fake')
plt.plot(np.arange(1, len(buzz_real_article_sentiments)+1), buzz_real_article_sentiments, 'g*', label='Fact')
plt.axis([0, 100, -4, 4])
plt.legend(loc='upper right')
plt.title('BuzzFeed article sentiments')
plt.xlabel('Article index')
plt.ylabel('Average Sentiment Intensity')
# PolitiFact
plt.subplot(2, 2, 2)
plt.plot(np.arange(1, len(poli_fake_article_sentiments)+1), poli_fake_article_sentiments, 'rx', label='Fake')
plt.plot(np.arange(1, len(poli_real_article_sentiments)+1), poli_real_article_sentiments, 'gx', label='Fact')
plt.axis([0, 130, -4, 4])
plt.legend(loc='upper right')
plt.title('PolitiFact article sentiments')
plt.xlabel('Article index')
plt.ylabel('Average Sentiment Intensity')
# Kaggle
plt.subplot(2, 2, 3)
plt.plot(np.arange(1, len(kagg_fake_article_sentiments)+1), kagg_fake_article_sentiments, 'r.', label='Fake')
plt.plot(np.arange(1, len(kagg_real_article_sentiments)+1), kagg_real_article_sentiments, 'g.', label='Fact')
plt.axis([0, 110, -4, 4])
plt.legend(loc='upper right')
plt.title('Kaggle article sentiments')
plt.xlabel('Article index')
plt.ylabel('Average Sentiment Intensity')
# All
plt.subplot(2, 2, 4)
plt.plot(np.arange(1, len(buzz_fake_article_sentiments)+1), buzz_fake_article_sentiments, 'r*', label='BuzzFeed Fake')
plt.plot(np.arange(1, len(buzz_real_article_sentiments)+1), buzz_real_article_sentiments, 'g*', label='BuzzFeed Fact')
plt.plot(np.arange(1, len(poli_fake_article_sentiments)+1), poli_fake_article_sentiments, 'rx', label='PolitiFact Fake')
plt.plot(np.arange(1, len(poli_real_article_sentiments)+1), poli_real_article_sentiments, 'gx', label='PolitiFact Fact')
plt.plot(np.arange(1, len(kagg_fake_article_sentiments)+1), kagg_fake_article_sentiments, 'r.', label='Kaggle Fake')
plt.plot(np.arange(1, len(kagg_real_article_sentiments)+1), kagg_real_article_sentiments, 'g.', label='Kaggle Fact')
plt.axis([0, 130, -4, 4])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('All article sentiments')
plt.xlabel('Article index')
plt.ylabel('Average Sentiment Intensity')

# Ordered plots
plt.figure()
# BuzzFeed
plt.subplot(2, 2, 1)
plt.plot(np.arange(1, len(buzz_fake_article_sentiments)+1), np.sort(buzz_fake_article_sentiments), 'r*', label='Fake')
plt.plot(np.arange(1, len(buzz_real_article_sentiments)+1), np.sort(buzz_real_article_sentiments), 'g*', label='Fact')
plt.axis([0, 100, -4, 4])
plt.legend(loc='upper left')
plt.title('BuzzFeed ordered article sentiments')
plt.xlabel('Article index')
plt.ylabel('Average Sentiment Intensity')
# PolitiFact
plt.subplot(2, 2, 2)
plt.plot(np.arange(1, len(poli_fake_article_sentiments)+1), np.sort(poli_fake_article_sentiments), 'rx', label='Fake')
plt.plot(np.arange(1, len(poli_real_article_sentiments)+1), np.sort(poli_real_article_sentiments), 'gx', label='Fact')
plt.axis([0, 130, -4, 4])
plt.legend(loc='upper left')
plt.title('PolitiFact ordered article sentiments')
plt.xlabel('Article index')
plt.ylabel('Average Sentiment Intensity')
# Kaggle
plt.subplot(2, 2, 3)
plt.plot(np.arange(1, len(kagg_fake_article_sentiments)+1), np.sort(kagg_fake_article_sentiments), 'r.', label='Fake')
plt.plot(np.arange(1, len(kagg_real_article_sentiments)+1), np.sort(kagg_real_article_sentiments), 'g.', label='Fact')
plt.axis([0, 110, -4, 4])
plt.legend(loc='upper left')
plt.title('Kaggle article sentiments')
plt.xlabel('Article index')
plt.ylabel('Average Sentiment Intensity')
# All
plt.subplot(2, 2, 4)
plt.plot(np.arange(1, len(all_fake_article_sentiments)+1), np.sort(all_fake_article_sentiments), 'r.', label='Fake')
plt.plot(np.arange(1, len(all_real_article_sentiments)+1), np.sort(all_real_article_sentiments), 'g.', label='Fact')
plt.axis([0, 320, -4, 4])
plt.legend(loc='upper left')
plt.title('All article sentiments')
plt.xlabel('Article index')
plt.ylabel('Average Sentiment Intensity')

# Histograms:
plt.figure()
# BuzzFeed
plt.subplot(2, 2, 1)
plt.hist(buzz_fake_article_sentiments, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Fake')
plt.hist(buzz_real_article_sentiments, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Fact')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intensity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('BuzzFeed fact and fake articles')
# PolitiFact
plt.subplot(2, 2, 2)
plt.hist(poli_fake_article_sentiments, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Fake')
plt.hist(poli_real_article_sentiments, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Fact')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intensity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('PloitiFact fact and fake articles')
# Kaggle
plt.subplot(2, 2, 3)
plt.hist(kagg_fake_article_sentiments, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Fake')
plt.hist(kagg_real_article_sentiments, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Fact')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intensity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('Kaggle fact and fake articles')
# All
plt.subplot(2, 2, 4)
plt.hist(all_fake_article_sentiments, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Fake')
plt.hist(all_real_article_sentiments, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Fact')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intensity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('All fact and fake articles')

# KDEs:
fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0,0].set_xlabel('Sentiment Intensity')
axes[0,1].set_xlabel('Sentiment Intensity')
axes[1,0].set_xlabel('Sentiment Intensity')
axes[1,1].set_xlabel('Sentiment Intensity')
df_news_buzz_article.plot.kde(title='BuzzFeed KDE (articles)', ax=axes[0,0], color='rb')
df_news_poli_article.plot.kde(title='PolitiFact KDE (articles)', ax=axes[0,1], color='rb')
df_news_kagg_article.plot.kde(title='Kaggle KDE (articles)', ax=axes[1,0], color='rb')
df_news_all_articles.plot.kde(title='All KDE (articles)', ax=axes[1,1], color='rb')

#%% Sentences:

# Scatter plots:
# BuzzFeed
plt.figure()
plt.subplot(2, 2, 1)
plt.plot(np.arange(1, len(buzz_fake_flat_list)+1), buzz_fake_flat_list, 'r*', label='Fake')
plt.plot(np.arange(1, len(buzz_real_flat_list)+1), buzz_real_flat_list, 'g*', label='Fact')
#plt.axis([0, 100, -4, 4])
plt.legend(loc='upper right')
plt.title('BuzzFeed sentence sentiments')
plt.xlabel('Sentence index')
plt.ylabel('Average Sentiment Intensity')
# PolitiFact
plt.subplot(2, 2, 2)
plt.plot(np.arange(1, len(poli_fake_flat_list)+1), poli_fake_flat_list, 'rx', label='Fake')
plt.plot(np.arange(1, len(poli_real_flat_list)+1), poli_real_flat_list, 'gx', label='Fact')
#plt.axis([0, 130, -4, 4])
plt.legend(loc='upper right')
plt.title('PolitiFact sentence sentiments')
plt.xlabel('Sentence index')
plt.ylabel('Average Sentiment Intensity')
# Kaggle
plt.subplot(2, 2, 3)
plt.plot(np.arange(1, len(kagg_fake_flat_list)+1), kagg_fake_flat_list, 'r.', label='Fake')
plt.plot(np.arange(1, len(kagg_real_flat_list)+1), kagg_real_flat_list, 'g.', label='Fact')
#plt.axis([0, 110, -4, 4])
plt.legend(loc='upper right')
plt.title('Kaggle sentence sentiments')
plt.xlabel('Sentence index')
plt.ylabel('Average Sentiment Intensity')
# All
plt.subplot(2, 2, 4)
plt.plot(np.arange(1, len(buzz_fake_flat_list)+1), buzz_fake_flat_list, 'r*', label='BuzzFeed Fake')
plt.plot(np.arange(1, len(buzz_real_flat_list)+1), buzz_real_flat_list, 'g*', label='BuzzFeed Fact')
plt.plot(np.arange(1, len(poli_fake_flat_list)+1), poli_fake_flat_list, 'rx', label='PolitiFact Fake')
plt.plot(np.arange(1, len(poli_real_flat_list)+1), poli_real_flat_list, 'gx', label='PolitiFact Fact')
plt.plot(np.arange(1, len(kagg_fake_flat_list)+1), kagg_fake_flat_list, 'r.', label='Kaggle Fake')
plt.plot(np.arange(1, len(kagg_real_flat_list)+1), kagg_real_flat_list, 'g.', label='Kaggle Fact')
#plt.axis([0, 130, -4, 4])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('All sentence sentiments')
plt.xlabel('Sentence index')
plt.ylabel('Average Sentiment Intensity')

# Histograms:
plt.figure()
# BuzzFeed
plt.subplot(2, 2, 1)
plt.hist(buzz_fake_flat_list, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Fake')
plt.hist(buzz_real_flat_list, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Fact')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intensity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('BuzzFeed fact and fake sentences')
# PolitiFact
plt.subplot(2, 2, 2)
plt.hist(poli_fake_flat_list, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Fake')
plt.hist(poli_real_flat_list, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Fact')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intensity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('PloitiFact fact and fake sentences')
# Kaggle
plt.subplot(2, 2, 3)
plt.hist(kagg_fake_flat_list, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Fake')
plt.hist(kagg_real_flat_list, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Fact')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intensity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('Kaggle fact and fake sentences')
# All
plt.subplot(2, 2, 4)
plt.hist(all_fake_sentence_sentiments, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Fake')
plt.hist(all_real_sentence_sentiments, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Fact')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intensity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('All fact and fake sentences')

# KDEs:
fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0,0].set_xlabel('Sentiment Intensity')
axes[0,1].set_xlabel('Sentiment Intensity')
axes[1,0].set_xlabel('Sentiment Intensity')
axes[1,1].set_xlabel('Sentiment Intensity')
df_news_buzz_sentence.plot.kde(title='BuzzFeed KDE (sentences)', ax=axes[0,0], color='rb')
df_news_poli_sentence.plot.kde(title='PolitiFact KDE (sentences)', ax=axes[0,1], color='rb')
df_news_kagg_sentence.plot.kde(title='Kaggle KDE (sentences)', ax=axes[1,0], color='rb')
df_news_all_sentence.plot.kde(title='All KDE (sentences)', ax=axes[1,1], color='rb')

# Absolute sentiment:

#%% Articles

# scatter
fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0,0].set_xlabel('Article index')
axes[0,1].set_xlabel('Article index')
axes[1,0].set_xlabel('Article index')
axes[1,1].set_xlabel('Article index')
axes[0,0].set_ylabel('Sentiment Intensity')
axes[0,1].set_ylabel('Sentiment Intensity')
axes[1,0].set_ylabel('Sentiment Intensity')
axes[1,1].set_ylabel('Sentiment Intensity')
axes[0,0].plot(np.arange(1, len(buzz_fake_art_abs)+1), buzz_fake_art_abs, 'r*', label='Fake')
axes[0,0].plot(np.arange(1, len(buzz_real_art_abs)+1), buzz_real_art_abs, 'g*', label='Fact')
axes[0,1].plot(np.arange(1, len(poli_fake_art_abs)+1), poli_fake_art_abs, 'rx', label='Fake')
axes[0,1].plot(np.arange(1, len(poli_real_art_abs)+1), poli_real_art_abs, 'gx', label='Fact')
axes[1,0].plot(np.arange(1, len(kagg_fake_art_abs)+1), kagg_fake_art_abs, 'r.', label='Fake')
axes[1,0].plot(np.arange(1, len(kagg_real_art_abs)+1), kagg_real_art_abs, 'g.', label='Fact')
axes[1,1].plot(np.arange(1, len(buzz_fake_art_abs)+1), buzz_fake_art_abs, 'r*', label='BuzzFeed Fake')
axes[1,1].plot(np.arange(1, len(buzz_real_art_abs)+1), buzz_real_art_abs, 'g*', label='BuzzFeed Fact')
axes[1,1].plot(np.arange(1, len(poli_fake_art_abs)+1), poli_fake_art_abs, 'rx', label='PolitiFact Fake')
axes[1,1].plot(np.arange(1, len(poli_real_art_abs)+1), poli_real_art_abs, 'gx', label='PolitiFact Fact')
axes[1,1].plot(np.arange(1, len(kagg_fake_art_abs)+1), kagg_fake_art_abs, 'r.', label='Kaggle Fake')
axes[1,1].plot(np.arange(1, len(kagg_real_art_abs)+1), kagg_real_art_abs, 'g.', label='Kaggle Fact')
axes[0,0].axis([0, 100, -1, 4])
axes[0,1].axis([0, 130, -1, 4])
axes[1,0].axis([0, 110, -1, 4])
axes[1,1].axis([0, 130, -1, 4])
axes[0,0].legend(loc='upper right')
axes[0,1].legend(loc='upper right')
axes[1,0].legend(loc='upper right')
axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
axes[0,0].set_title('BuzzFeed absolute article sentiments')
axes[0,1].set_title('PolitiFact absolute article sentiments')
axes[1,0].set_title('Kaggle absolute article sentiments')
axes[1,1].set_title('All absolute article sentiments')


# ordered
fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0,0].set_xlabel('Article index')
axes[0,1].set_xlabel('Article index')
axes[1,0].set_xlabel('Article index')
axes[1,1].set_xlabel('Article index')
axes[0,0].set_ylabel('Sentiment Intensity')
axes[0,1].set_ylabel('Sentiment Intensity')
axes[1,0].set_ylabel('Sentiment Intensity')
axes[1,1].set_ylabel('Sentiment Intensity')
axes[0,0].plot(np.arange(1, len(buzz_fake_art_abs)+1), np.sort(buzz_fake_art_abs), 'r*', label='Fake')
axes[0,0].plot(np.arange(1, len(buzz_real_art_abs)+1), np.sort(buzz_real_art_abs), 'g*', label='Fact')
axes[0,1].plot(np.arange(1, len(poli_fake_art_abs)+1), np.sort(poli_fake_art_abs), 'rx', label='Fake')
axes[0,1].plot(np.arange(1, len(poli_real_art_abs)+1), np.sort(poli_real_art_abs), 'gx', label='Fact')
axes[1,0].plot(np.arange(1, len(kagg_fake_art_abs)+1), np.sort(kagg_fake_art_abs), 'r.', label='Fake')
axes[1,0].plot(np.arange(1, len(kagg_real_art_abs)+1), np.sort(kagg_real_art_abs), 'g.', label='Fact')
axes[1,1].plot(np.arange(1, len(all_fake_art_abs)+1), np.sort(all_fake_art_abs), 'r.', label='Fake')
axes[1,1].plot(np.arange(1, len(all_real_art_abs)+1), np.sort(all_real_art_abs), 'g.', label='Fact')
axes[0,0].axis([0, 100, -1, 4])
axes[0,1].axis([0, 130, -1, 4])
axes[1,0].axis([0, 110, -1, 4])
axes[1,1].axis([0, 340, -1, 4])
axes[0,0].legend(loc='upper left')
axes[0,1].legend(loc='upper left')
axes[1,0].legend(loc='upper left')
axes[1,1].legend(loc='upper left')
axes[0,0].set_title('BuzzFeed absolute article sentiments')
axes[0,1].set_title('PolitiFact absolute article sentiments')
axes[1,0].set_title('Kaggle absolute article sentiments')
axes[1,1].set_title('All absolute article sentiments')

# KDE abs
fig, axes = plt.subplots(nrows=2, ncols=2)
df_news_buzz_art_abs.plot.kde(title='BuzzFeed absolute KDE (articles)', ax=axes[0,0], color='rb')
df_news_poli_art_abs.plot.kde(title='PolitiFact absolute KDE (articles)', ax=axes[0,1], color='rb')
df_news_kagg_art_abs.plot.kde(title='Kaggle absolute KDE (articles)', ax=axes[1,0], color='rb')
df_news_all_art_abs.plot.kde(title='All absolute KDE (articles)', ax=axes[1,1], color='rb')
axes[0,0].set_xlabel('Article absolute sentiment intensity')
axes[0,0].set_ylabel('Density')
axes[0,0].legend(loc='upper right')
axes[0,1].set_xlabel('Article absolute sentiment intensity')
axes[0,1].set_ylabel('Density')
axes[0,1].legend(loc='upper right')
axes[1,0].set_xlabel('Article absolute sentiment intensity')
axes[1,0].set_ylabel('Density')
axes[1,0].legend(loc='upper right')
axes[1,1].set_xlabel('Article absolute sentiment intensity')
axes[1,1].set_ylabel('Density')
axes[1,1].legend(loc='upper right')

#
#df_news_real = pandas.DataFrame({'Fact': np.absolute(np.asarray(buzz_real_flat_list))})
#df_news_fake = pandas.DataFrame({'Fake': np.absolute(np.asarray(buzz_fake_flat_list))})
#df_news_buzz_sentence_abs = pandas.concat([df_news_fake,df_news_real], axis=1)
#df_news_buzz_sentence_abs.plot.kde(title='BuzzFeed KDE (sentences)', color='rb')

plt.show()