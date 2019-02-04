#%%
import newsSentimentAnalysis as NSA
import matplotlib.pyplot as plt
import numpy as np
import pandas

#%% ANALYSIS

data1_path = 'News_Data\\multi_News_kaggle\\articles1.csv'
data2_path = 'News_Data\\multi_News_kaggle\\articles2.csv'
data3_path = 'News_Data\\multi_News_kaggle\\articles3.csv'
NYT_sentence_list, NYT_sentence_sentiments, NYT_article_sentiments = NSA.kaggle_mult_news_analysis(data1_path, 'New York Times')
reuters_sentence_list, reuters_sentence_sentiments, reuters_article_sentiments = NSA.kaggle_mult_news_analysis(data3_path, 'Reuters')
fox_sentence_list, fox_sentence_sentiments, fox_article_sentiments = NSA.kaggle_mult_news_analysis(data2_path, 'Fox News')
CNN_sentence_list, CNN_sentence_sentiments, CNN_article_sentiments = NSA.kaggle_mult_news_analysis(data1_path, 'CNN')
BB_sentence_list, BB_sentence_sentiments, BB_article_sentiments = NSA.kaggle_mult_news_analysis(data1_path, 'Breitbart')
GUA_sentence_list1, GUA_sentence_sentiments1, GUA_article_sentiments1 = NSA.kaggle_mult_news_analysis(data2_path, 'Guardian')
GUA_sentence_list2, GUA_sentence_sentiments2, GUA_article_sentiments2 = NSA.kaggle_mult_news_analysis(data3_path, 'Guardian')

GUA_sentence_list = GUA_sentence_list1 + GUA_sentence_list2
GUA_sentence_sentiments = GUA_sentence_sentiments1 + GUA_sentence_sentiments2
GUA_article_sentiments = GUA_article_sentiments1 + GUA_article_sentiments2

NYT_flat_list = [item for sublist in NYT_sentence_sentiments for item in sublist]
reuters_flat_list = [item for sublist in reuters_sentence_sentiments for item in sublist]
fox_flat_list = [item for sublist in fox_sentence_sentiments for item in sublist]
CNN_flat_list = [item for sublist in CNN_sentence_sentiments for item in sublist]
BB_flat_list = [item for sublist in BB_sentence_sentiments for item in sublist]
GUA_flat_list = [item for sublist in GUA_sentence_sentiments for item in sublist]

NYT_art_abs = NSA.cal_article_abs_sentiment(NYT_sentence_sentiments)
reuters_art_abs = NSA.cal_article_abs_sentiment(reuters_sentence_sentiments)
fox_art_abs = NSA.cal_article_abs_sentiment(fox_sentence_sentiments)
CNN_art_abs = NSA.cal_article_abs_sentiment(CNN_sentence_sentiments)
BB_art_abs = NSA.cal_article_abs_sentiment(BB_sentence_sentiments)
GUA_art_abs = NSA.cal_article_abs_sentiment(GUA_sentence_sentiments)

# mean and varience calculations
# sentiment lists to arrays
NYT_article_sentiments, art_avg_NYT, art_var_NYT = NSA.avg_var_calculation(NYT_article_sentiments, "", "(NYT)")
reuters_article_sentiments, art_avg_reuters, art_var_reuters = NSA.avg_var_calculation(reuters_article_sentiments, "", "(reuters)")
fox_article_sentiments, art_avg_fox, art_var_fox = NSA.avg_var_calculation(fox_article_sentiments, "", "(fox)")
CNN_article_sentiments, art_avg_CNN, art_var_CNN = NSA.avg_var_calculation(CNN_article_sentiments, "", "(CNN)")
BB_article_sentiments, art_avg_BB, art_var_BB = NSA.avg_var_calculation(BB_article_sentiments, "", "(BB)")
GUA_article_sentiments, art_avg_GUA, art_var_GUA = NSA.avg_var_calculation(GUA_article_sentiments, "", "(GUA)")

#all_article_sentiments = np.concatenate((NYT_article_sentiments, poli_article_sentiments, kagg_article_sentiments), axis=None) # move this and do more with it
#all_sentence_sentiments = np.concatenate((NYT_flat_list, poli_flat_list, kagg_flat_list), axis=None)

# Article dataframes:
# NYT
df_news_NYT_article_avg = pandas.DataFrame({'NYT': NYT_article_sentiments})
df_news_NYT_article_abs = pandas.DataFrame({'NYT': NYT_art_abs})

df_news_reuters_article_avg = pandas.DataFrame({'Reuters': reuters_article_sentiments})
df_news_reuters_article_abs = pandas.DataFrame({'Reuters': reuters_art_abs})

df_news_fox_article_avg = pandas.DataFrame({'Fox': fox_article_sentiments})
df_news_fox_article_abs = pandas.DataFrame({'Fox': fox_art_abs})

df_news_CNN_article_avg = pandas.DataFrame({'CNN': CNN_article_sentiments})
df_news_CNN_article_abs = pandas.DataFrame({'CNN': CNN_art_abs})

df_news_BB_article_avg = pandas.DataFrame({'Breitbart': BB_article_sentiments})
df_news_BB_article_abs = pandas.DataFrame({'Breitbart': BB_art_abs})

df_news_GUA_article_avg = pandas.DataFrame({'Guardian': GUA_article_sentiments})
df_news_GUA_article_abs = pandas.DataFrame({'Guardian': GUA_art_abs})

# All
#df_news_all_articles = pandas.concat([df_news_fake,df_news], axis=1)

# Sentence dataframes:
# NYT
df_news_NYT_sentence_avg = pandas.DataFrame({'NYT': NYT_flat_list})
df_news_NYT_sentence_abs = pandas.DataFrame({'NYT': np.absolute(np.asarray(NYT_flat_list))})

df_news_reuters_sentence_avg = pandas.DataFrame({'Reuters': reuters_flat_list})
df_news_reuters_sentence_abs = pandas.DataFrame({'Reuters': np.absolute(np.asarray(reuters_flat_list))})

df_news_fox_sentence_avg = pandas.DataFrame({'Fox': fox_flat_list})
df_news_fox_sentence_abs = pandas.DataFrame({'Fox': np.absolute(np.asarray(fox_flat_list))})

df_news_CNN_sentence_avg = pandas.DataFrame({'CNN': CNN_flat_list})
df_news_CNN_sentence_abs = pandas.DataFrame({'CNN': np.absolute(np.asarray(CNN_flat_list))})

df_news_BB_sentence_avg = pandas.DataFrame({'Breitbart': BB_flat_list})
df_news_BB_sentence_abs = pandas.DataFrame({'Breitbart': np.absolute(np.asarray(BB_flat_list))})

df_news_GUA_sentence_avg = pandas.DataFrame({'Guardian': GUA_flat_list})
df_news_GUA_sentence_abs = pandas.DataFrame({'Guardian': np.absolute(np.asarray(GUA_flat_list))})

#df_news_ALL_sentence = pandas.concat([df_news_fake,df_news], axis=1)

bin_num = 'auto' # np.linspace(-4, 4, 50)



# SUB-PLOTS

#%% Articles:

# Scatter plots:
# All
plt.subplot(2, 2, 1)
plt.plot(np.arange(1, len(NYT_article_sentiments)+1), NYT_article_sentiments, 'b+', label='NYT')
plt.plot(np.arange(1, len(reuters_article_sentiments)+1), reuters_article_sentiments, 'g+', label='Reuters')
plt.plot(np.arange(1, len(fox_article_sentiments)+1), fox_article_sentiments, 'm+', label='Fox')
plt.plot(np.arange(1, len(CNN_article_sentiments)+1), CNN_article_sentiments, 'y+', label='CNN')
plt.plot(np.arange(1, len(BB_article_sentiments)+1), BB_article_sentiments, 'r+', label='Breitbart')
plt.plot(np.arange(1, len(GUA_article_sentiments)+1), GUA_article_sentiments, 'c+', label='Guardian')
plt.axis([0, 101, -4, 4])
plt.legend(loc='upper right')
plt.title('New article sentiments')
plt.xlabel('Article index')
plt.ylabel('Average Sentiment Intensity')
# most neutral
plt.subplot(2, 2, 1)
# least neutral
plt.subplot(2, 2, 1)
# Comp
plt.subplot(2, 2, 1)

# All
#plt.subplot(2, 2, 4)
#plt.plot(np.arange(1, len(buzz_fake_article_sentiments)+1), buzz_fake_article_sentiments, 'r+', label='NYT Fake')
#plt.plot(np.arange(1, len(buzz_article_sentiments)+1), buzz_article_sentiments, 'g+', label='NYT Fact')
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
# All
plt.subplot(2, 2, 1)
plt.hist(NYT_article_sentiments, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='NYT')
plt.hist(reuters_article_sentiments, bins=bin_num, color='g', alpha=0.7, rwidth=0.85, label='Reuters')
plt.hist(fox_article_sentiments, bins=bin_num, color='m', alpha=0.7, rwidth=0.85, label='Fox')
plt.hist(CNN_article_sentiments, bins=bin_num, color='y', alpha=0.7, rwidth=0.85, label='CNN')
plt.hist(BB_article_sentiments, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Breitbart')
plt.hist(GUA_article_sentiments, bins=bin_num, color='c', alpha=0.7, rwidth=0.85, label='Guardian')
plt.grid(axis='y', alpha=0.75)
plt.axis([-4, 4, 0, 50])
plt.xlabel('Sentiment intensity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('Average article sentiment')
# Reuters
plt.subplot(2, 2, 1)
# Fox
plt.subplot(2, 2, 1)
# CNN
plt.subplot(2, 2, 1)
# BB
plt.subplot(2, 2, 1)


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
axes[0,0].set_xlim([-4,4])
axes[0,0].set_ylim([0,1])
axes[0,1].set_xlim([-4,4])
axes[0,1].set_ylim([0,1])
axes[1,0].set_xlim([-4,4])
axes[1,0].set_ylim([0,1])
axes[1,1].set_xlim([-4,4])
axes[1,1].set_ylim([0,1])
df_news_NYT_article_avg.plot.kde(ax=axes[0,0], color='b')
df_news_reuters_article_avg.plot.kde(ax=axes[0,0], color='g')
df_news_fox_article_avg.plot.kde(ax=axes[0,0], color='m')
df_news_CNN_article_avg.plot.kde(ax=axes[0,0], color='y')
df_news_BB_article_avg.plot.kde(ax=axes[0,0], color='r')
df_news_GUA_article_avg.plot.kde(ax=axes[0,0], color='c')
#df_news_poli_article.plot.kde(title='PolitiFact KDE (articles)', ax=axes[0,1], color='b')
#df_news_kagg_article.plot.kde(title='Kaggel KDE (articles)', ax=axes[1,0], color='b')
#df_news_all_articles.plot.kde(title='All KDE (articles)', ax=axes[1,1], color='b')

# KDEs abs:
fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0,0].set_xlabel('Sentiment Intensity')
axes[0,1].set_xlabel('Sentiment Intensity')
axes[1,0].set_xlabel('Sentiment Intensity')
axes[1,1].set_xlabel('Sentiment Intensity')
axes[0,0].set_xlim([-1,4])
axes[0,0].set_ylim([0,2])
axes[0,1].set_xlim([-1,4])
axes[0,1].set_ylim([0,2])
axes[1,0].set_xlim([-1,4])
axes[1,0].set_ylim([0,2])
axes[1,1].set_xlim([-1,4])
axes[1,1].set_ylim([0,2])
df_news_NYT_article_abs.plot.kde(ax=axes[0,0], color='b')
df_news_reuters_article_abs.plot.kde(ax=axes[0,0], color='g')
df_news_fox_article_abs.plot.kde(ax=axes[0,0], color='m')
df_news_CNN_article_abs.plot.kde(ax=axes[0,0], color='y')
df_news_BB_article_abs.plot.kde(ax=axes[0,0], color='r')
df_news_GUA_article_abs.plot.kde(ax=axes[0,0], color='c')
#df_news_poli_article.plot.kde(title='PolitiFact KDE (articles)', ax=axes[0,1], color='b')
#df_news_kagg_article.plot.kde(title='Kaggel KDE (articles)', ax=axes[1,0], color='b')
#df_news_all_articles.plot.kde(title='All KDE (articles)', ax=axes[1,1], color='b')

#%% Sentences:

# Scatter plots:
# NYT
plt.figure()
plt.subplot(2, 2, 1)
plt.plot(np.arange(1, len(NYT_flat_list)+1), NYT_flat_list, 'b+', label='NYT')
plt.plot(np.arange(1, len(reuters_flat_list)+1), reuters_flat_list, 'g+', label='Reuters')
plt.plot(np.arange(1, len(fox_flat_list)+1), fox_flat_list, 'm+', label='Fox')
plt.plot(np.arange(1, len(CNN_flat_list)+1), CNN_flat_list, 'y+', label='CNN')
plt.plot(np.arange(1, len(BB_flat_list)+1), BB_flat_list, 'r+', label='Breitbart')
plt.plot(np.arange(1, len(GUA_flat_list)+1), GUA_flat_list, 'c+', label='Guardian')
plt.axis([0, 7000, -4, 4])
plt.legend(loc='upper right')
plt.title('Sentence sentiments')
plt.xlabel('Sentence index')
plt.ylabel('Average Sentiment Intensity')
# Reuters
plt.subplot(2, 2, 1)
# Fox
plt.subplot(2, 2, 1)
# CNN
plt.subplot(2, 2, 1)
# BB
plt.subplot(2, 2, 1)

# All
#plt.subplot(2, 2, 4)
#plt.plot(np.arange(1, len(buzz_fake_flat_list)+1), buzz_fake_flat_list, 'r+', label='NYT Fake')
#plt.plot(np.arange(1, len(buzz_flat_list)+1), buzz_flat_list, 'g+', label='NYT Fact')
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
# All
plt.subplot(2, 2, 1)
plt.hist(NYT_flat_list, bins=bin_num, color='b', alpha=0.7, rwidth=0.85, label='NYT')
plt.hist(reuters_flat_list, bins=bin_num, color='g', alpha=0.7, rwidth=0.85, label='Reuters')
plt.hist(fox_flat_list, bins=bin_num, color='m', alpha=0.7, rwidth=0.85, label='Fox')
plt.hist(CNN_flat_list, bins=bin_num, color='y', alpha=0.7, rwidth=0.85, label='CNN')
plt.hist(BB_flat_list, bins=bin_num, color='r', alpha=0.7, rwidth=0.85, label='Breitbart')
plt.hist(GUA_flat_list, bins=bin_num, color='c', alpha=0.7, rwidth=0.85, label='Guardian')
plt.grid(axis='y', alpha=0.75)
plt.axis([-4, 4, 0, 3000])
plt.xlabel('Sentiment intensity')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('Sentence sentiments')
# 
plt.subplot(2, 2, 1)
#
plt.subplot(2, 2, 1)
# CNN
plt.subplot(2, 2, 1)
# BB
plt.subplot(2, 2, 1)

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
axes[0,0].set_xlim([-4,4])
axes[0,0].set_ylim([0,1])
axes[0,1].set_xlim([-4,4])
axes[0,1].set_ylim([0,1])
axes[1,0].set_xlim([-4,4])
axes[1,0].set_ylim([0,1])
axes[1,1].set_xlim([-4,4])
axes[1,1].set_ylim([0,1])
df_news_NYT_sentence_avg.plot.kde(ax=axes[0,0], color='b')
df_news_reuters_sentence_avg.plot.kde(ax=axes[0,0], color='g')
df_news_fox_sentence_avg.plot.kde(ax=axes[0,0], color='m')
df_news_CNN_sentence_avg.plot.kde(ax=axes[0,0], color='y')
df_news_BB_sentence_avg.plot.kde(ax=axes[0,0], color='r')
df_news_GUA_sentence_avg.plot.kde(ax=axes[0,0], color='c')
#df_news_poli_sentence.plot.kde(title='PolitiFact KDE (sentences)', ax=axes[0,1], color='b')
#df_news_kagg_sentence.plot.kde(title='Kaggel KDE (sentences)', ax=axes[1,0], color='b')
#df_news_all_sentence.plot.kde(title='All KDE (sentences)', ax=axes[1,1], color='b')

# KDEs:
fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0,0].set_xlabel('Sentiment Intensity')
axes[0,1].set_xlabel('Sentiment Intensity')
axes[1,0].set_xlabel('Sentiment Intensity')
axes[1,1].set_xlabel('Sentiment Intensity')
axes[0,0].set_xlim([-2,5])
axes[0,0].set_ylim([0,1])
axes[0,1].set_xlim([-2,5])
axes[0,1].set_ylim([0,1])
axes[1,0].set_xlim([-2,5])
axes[1,0].set_ylim([0,1])
axes[1,1].set_xlim([-2,5])
axes[1,1].set_ylim([0,1])
df_news_NYT_sentence_abs.plot.kde(ax=axes[0,0], color='b')
df_news_reuters_sentence_abs.plot.kde(ax=axes[0,0], color='g')
df_news_fox_sentence_abs.plot.kde(ax=axes[0,0], color='m')
df_news_CNN_sentence_abs.plot.kde(ax=axes[0,0], color='y')
df_news_BB_sentence_abs.plot.kde(ax=axes[0,0], color='r')
df_news_GUA_sentence_abs.plot.kde(ax=axes[0,0], color='c')
#df_news_poli_sentence.plot.kde(title='PolitiFact KDE (sentences)', ax=axes[0,1], color='b')
#df_news_kagg_sentence.plot.kde(title='Kaggel KDE (sentences)', ax=axes[1,0], color='b')
#df_news_all_sentence.plot.kde(title='All KDE (sentences)', ax=axes[1,1], color='b')

plt.show()