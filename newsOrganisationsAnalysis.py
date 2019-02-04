#%%
import newsSentimentAnalysis as NSA
import matplotlib.pyplot as plt
import numpy as np
import pandas

#%% ANALYSIS

path = 'News_Data\\multi_News_kaggle\\articles1.csv'
NYT_sentence_list, NYT_sentence_sentiments, NYT_article_sentiments = NSA.kaggle_mult_news_analysis(path, 'New York Times')
#reuters_sentence_list, reuters_sentence_sentiments, reuters_article_sentiments = NSA.kaggle_mult_news_analysis(path, 'Reuters')

NYT_flat_list = [item for sublist in NYT_sentence_sentiments for item in sublist]
#reuters_flat_list = [item for sublist in reuters_sentence_sentiments for item in sublist]

NYT_art_abs = NSA.cal_article_abs_sentiment(NYT_sentence_sentiments)
#reuters_art_abs = NSA.cal_article_abs_sentiment(reuters_sentence_sentiments)

# mean and varience calculations
# sentiment lists to arrays
NYT_article_sentiments, art_avg_NYT, art_var_NYT = NSA.avg_var_calculation(NYT_article_sentiments, "", "(NYT)")
#reuters_article_sentiments, art_avg_reuters, art_var_reuters = NSA.avg_var_calculation(reuters_article_sentiments, "", "(reuters)")


#all_article_sentiments = np.concatenate((NYT_article_sentiments, poli_article_sentiments, kagg_article_sentiments), axis=None) # move this and do more with it
#all_sentence_sentiments = np.concatenate((NYT_flat_list, poli_flat_list, kagg_flat_list), axis=None)

# Article dataframes:
# NYT
df_news_NYT_article_avg = pandas.DataFrame({'NYT': NYT_article_sentiments})
#df_news_reuters_article_avg = pandas.DataFrame({'reuters': reuters_article_sentiments})

df_news_NYT_article_abs = pandas.DataFrame({'NYT': NYT_art_abs})
#df_news_reuters_article_abs = pandas.DataFrame({'reuters': reuters_art_abs})

# All
#df_news_all_articles = pandas.concat([df_news_fake,df_news], axis=1)

# Sentence dataframes:
# NYT
df_news_NYT_sentence_avg = pandas.DataFrame({'NYT': NYT_flat_list})
#df_news_reuters_sentence_avg = pandas.DataFrame({'reuters': reuters_flat_list})

df_news_NYT_sentence_abs = pandas.DataFrame({'NYT': np.absolute(np.asarray(NYT_flat_list))})
#df_news_reuters_sentence_abs = pandas.DataFrame({'reuters': np.absolute(np.asarray(reuters_flat_list))})
#df_news_ALL_sentence = pandas.concat([df_news_fake,df_news], axis=1)

bin_num = 'auto' # np.linspace(-4, 4, 50)



# SUB-PLOTS

#%% Articles:

# Scatter plots:
# New York Times
plt.subplot(2, 2, 1)
plt.plot(np.arange(1, len(NYT_article_sentiments)+1), NYT_article_sentiments, 'b*', label='NYT')
#plt.plot(np.arange(1, len(reuters_article_sentiments)+1), reuters_article_sentiments, 'b*', label='reuters')
plt.axis([0, 110, -4, 4])
#plt.legend(loc='upper right')
plt.title('New York Times article sentiments')
plt.xlabel('Article index')
plt.ylabel('Average Sentiment Intensity')

# All
#plt.subplot(2, 2, 4)
#plt.plot(np.arange(1, len(buzz_fake_article_sentiments)+1), buzz_fake_article_sentiments, 'r*', label='NYT Fake')
#plt.plot(np.arange(1, len(buzz_article_sentiments)+1), buzz_article_sentiments, 'g*', label='NYT Fact')
#plt.plot(np.arange(1, len(poli_fake_article_sentiments)+1), poli_fake_article_sentiments, 'rx', label='PolitiFact Fake')
#plt.plot(np.arange(1, len(poli_article_sentiments)+1), poli_article_sentiments, 'gx', label='PolitiFact Fact')
#plt.plot(np.arange(1, len(kagg_fake_article_sentiments)+1), kagg_fake_article_sentiments, 'r.', label='Kaggel Fake')
#plt.plot(np.arange(1, len(kagg_article_sentiments)+1), kagg_article_sentiments, 'g.', label='Kaggel Fact')
#plt.axis([0, 130, -4, 4])
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.title('All article sentiments')
#plt.xlabel('Article index')
#plt.ylabel('Average Sentiment Intensity')

# Histograms:
plt.figure()
# Article histograms
plt.subplot(2, 2, 1)
plt.hist(NYT_article_sentiments, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='NYT')
#plt.hist(reuters_article_sentiments, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='reuters')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intensity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('Article histograms articles')

# All
#plt.subplot(2, 2, 4)
#plt.hist(all_fake_article_sentiments, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Fake')
#plt.hist(all_article_sentiments, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Fact')
#plt.grid(axis='y', alpha=0.75)
#plt.xlabel('Sentiment intensity')
#plt.ylabel('Frequency')
#plt.legend(loc='upper right')
#plt.title('All fact and fake articles')

# KDEs avg:
fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0,0].set_xlabel('Sentiment Intensity')
axes[0,1].set_xlabel('Sentiment Intensity')
axes[1,0].set_xlabel('Sentiment Intensity')
axes[1,1].set_xlabel('Sentiment Intensity')
df_news_NYT_article_avg.plot.kde(title='New York Times average KDE (articles)', ax=axes[0,0], color='b')
#df_news_reuters_article_avg.plot.kde(title='Reuters average KDE (articles)', ax=axes[0,0], color='b')
#df_news_poli_article.plot.kde(title='PolitiFact KDE (articles)', ax=axes[0,1], color='b')
#df_news_kagg_article.plot.kde(title='Kaggel KDE (articles)', ax=axes[1,0], color='b')
#df_news_all_articles.plot.kde(title='All KDE (articles)', ax=axes[1,1], color='b')

# KDEs abs:
fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0,0].set_xlabel('Sentiment Intensity')
axes[0,1].set_xlabel('Sentiment Intensity')
axes[1,0].set_xlabel('Sentiment Intensity')
axes[1,1].set_xlabel('Sentiment Intensity')
df_news_NYT_article_abs.plot.kde(title='New York Times absolute KDE (articles)', ax=axes[0,0], color='b')
#df_news_reuters_article_abs.plot.kde(title='Reuters absolute KDE (articles)', ax=axes[0,0], color='b')
#df_news_poli_article.plot.kde(title='PolitiFact KDE (articles)', ax=axes[0,1], color='b')
#df_news_kagg_article.plot.kde(title='Kaggel KDE (articles)', ax=axes[1,0], color='b')
#df_news_all_articles.plot.kde(title='All KDE (articles)', ax=axes[1,1], color='b')

#%% Sentences:

# Scatter plots:
# 
plt.figure()
plt.subplot(2, 2, 1)
plt.plot(np.arange(1, len(NYT_flat_list)+1), NYT_flat_list, 'b*', label='NYT')
#plt.plot(np.arange(1, len(reuters_flat_list)+1), reuters_flat_list, 'b*', label='Reuters')
#plt.axis([0, 100, -4, 4])
plt.legend(loc='upper right')
plt.title('Sentence sentiments')
plt.xlabel('Sentence index')
plt.ylabel('Average Sentiment Intensity')

# All
#plt.subplot(2, 2, 4)
#plt.plot(np.arange(1, len(buzz_fake_flat_list)+1), buzz_fake_flat_list, 'r*', label='NYT Fake')
#plt.plot(np.arange(1, len(buzz_flat_list)+1), buzz_flat_list, 'g*', label='NYT Fact')
#plt.plot(np.arange(1, len(poli_fake_flat_list)+1), poli_fake_flat_list, 'rx', label='PolitiFact Fake')
#plt.plot(np.arange(1, len(poli_flat_list)+1), poli_flat_list, 'gx', label='PolitiFact Fact')
#plt.plot(np.arange(1, len(kagg_fake_flat_list)+1), kagg_fake_flat_list, 'r.', label='Kaggel Fake')
#plt.plot(np.arange(1, len(kagg_flat_list)+1), kagg_flat_list, 'g.', label='Kaggel Fact')
##plt.axis([0, 130, -4, 4])
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.title('All sentence sentiments')
#plt.xlabel('Sentence index')
#plt.ylabel('Average Sentiment Intensity')

# Histograms:
plt.figure()
# New York Times
plt.subplot(2, 2, 1)
plt.hist(NYT_flat_list, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='NYT')
#plt.hist(reuters_flat_list, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='reuters')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Sentiment intensity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('New York Times sentences')
# All
#plt.subplot(2, 2, 4)
#plt.hist(all_fake_sentence_sentiments, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Fake')
#plt.hist(all_sentence_sentiments, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='Fact')
#plt.grid(axis='y', alpha=0.75)
#plt.xlabel('Sentiment intensity')
#plt.ylabel('Frequency')
#plt.legend(loc='upper right')
#plt.title('All fact and fake sentences')

# KDEs:
fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0,0].set_xlabel('Sentiment Intensity')
axes[0,1].set_xlabel('Sentiment Intensity')
axes[1,0].set_xlabel('Sentiment Intensity')
axes[1,1].set_xlabel('Sentiment Intensity')
df_news_NYT_sentence_avg.plot.kde(title='New York Times KDE (sentences)', ax=axes[0,0], color='b')
#df_news_reuters_sentence_avg.plot.kde(title='New York Times KDE (sentences)', ax=axes[0,0], color='b')
#df_news_poli_sentence.plot.kde(title='PolitiFact KDE (sentences)', ax=axes[0,1], color='b')
#df_news_kagg_sentence.plot.kde(title='Kaggel KDE (sentences)', ax=axes[1,0], color='b')
#df_news_all_sentence.plot.kde(title='All KDE (sentences)', ax=axes[1,1], color='b')

# KDEs:
fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0,0].set_xlabel('Sentiment Intensity')
axes[0,1].set_xlabel('Sentiment Intensity')
axes[1,0].set_xlabel('Sentiment Intensity')
axes[1,1].set_xlabel('Sentiment Intensity')
df_news_NYT_sentence_abs.plot.kde(title='New York Times KDE (sentences)', ax=axes[0,0], color='b')
#df_news_reuters_sentence_abs.plot.kde(title='New York Times KDE (sentences)', ax=axes[0,0], color='b')
#df_news_poli_sentence.plot.kde(title='PolitiFact KDE (sentences)', ax=axes[0,1], color='b')
#df_news_kagg_sentence.plot.kde(title='Kaggel KDE (sentences)', ax=axes[1,0], color='b')
#df_news_all_sentence.plot.kde(title='All KDE (sentences)', ax=axes[1,1], color='b')

plt.show()