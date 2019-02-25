#%%
import newsSentimentAnalysis as NSA
import matplotlib.pyplot as plt
import numpy as np
import pandas

nyt_opinion_url = 'https://www.nytimes.com/section/opinion'
nyt_world_url = 'https://www.nytimes.com/section/world'

NYTscrap_opinion_title_list, NYTscrap_opinion_sentence_list, NYTscrap_opinion_sentence_sentiments, NYTscrap_opinion_article_sentiments = NSA.nyt_scrape_analysis(nyt_opinion_url)
NYTscrap_world_title_list, NYTscrap_world_sentence_list, NYTscrap_world_sentence_sentiments, NYTscrap_world_article_sentiments = NSA.nyt_scrape_analysis(nyt_world_url)

#print(NYTscrap_opinion_sentence_list[0], '\n\n')
#print(NYTscrap_opinion_title_list[0], '\n\n')
#data = {'Title':title, 
#        'Author':authorname, 
#        'PageLink':pagelinks, 
#        'Article':myarticle, 
#        'Date':datetime.now()}
#data = {'Article':myarticle}

#print(len(NYTscrap_opinion_sentence_list), "  ", len(NYTscrap_opinion_title_list))

dictionary = dict(zip(NYTscrap_opinion_title_list, NYTscrap_opinion_sentence_list))
print(dictionary)
# %%

NYTscrap_opinion_flat_list = [item for sublist in NYTscrap_opinion_sentence_sentiments for item in sublist]
NYTscrap_world_flat_list = [item for sublist in NYTscrap_world_sentence_sentiments for item in sublist]

NYTscrap_opinion_art_abs = NSA.cal_article_abs_sentiment(NYTscrap_opinion_sentence_sentiments)
NYTscrap_world_art_abs = NSA.cal_article_abs_sentiment(NYTscrap_world_sentence_sentiments)

NYTscrap_opinion_article_sentiments, art_avg_NYTscrap_opinion, art_var_NYTscrap_opinion = NSA.avg_var_calculation(NYTscrap_opinion_article_sentiments, "", "(NYTscrap_opinion)")
NYTscrap_world_article_sentiments, art_avg_NYTscrap_world, art_var_NYTscrap_world = NSA.avg_var_calculation(NYTscrap_world_article_sentiments, "", "(NYTscrap_world)")

df_news_NYTscrap_opinion_article_avg = pandas.DataFrame({'NYT opinion articles': NYTscrap_opinion_article_sentiments})
df_news_NYTscrap_world_article_avg = pandas.DataFrame({'NYT world news articles': NYTscrap_world_article_sentiments})
df_news_NYTscrap_article_avg = pandas.concat([df_news_NYTscrap_opinion_article_avg, df_news_NYTscrap_world_article_avg], axis=1)


df_news_NYTscrap_opinion_article_abs = pandas.DataFrame({'NYT opinion articles': NYTscrap_opinion_art_abs})
df_news_NYTscrap_world_article_abs = pandas.DataFrame({'NYT world news articles': NYTscrap_world_art_abs})
df_news_NYTscrap_article_abs = pandas.concat([df_news_NYTscrap_opinion_article_abs, df_news_NYTscrap_world_article_abs], axis=1)







df_news_NYTscrap_opinion_sentence_avg = pandas.DataFrame({'NYTscrap_opinion': NYTscrap_opinion_flat_list})
df_news_NYTscrap_opinion_sentence_abs = pandas.DataFrame({'NYTscrap_opinion': np.absolute(np.asarray(NYTscrap_opinion_flat_list))})

df_news_NYTscrap_world_sentence_avg = pandas.DataFrame({'NYTscrap_world': NYTscrap_world_flat_list})
df_news_NYTscrap_world_sentence_abs = pandas.DataFrame({'NYTscrap_world': np.absolute(np.asarray(NYTscrap_world_flat_list))})

bin_num = 'auto' # np.linspace(-4, 4, 50)





#%% PLOTS
# average (NYT)
plt.subplot(2, 2, 1)
plt.plot(np.arange(1, len(NYTscrap_opinion_article_sentiments)+1), NYTscrap_opinion_article_sentiments, 'g+', label='NYT opinion articles')
plt.plot(np.arange(1, len(NYTscrap_world_article_sentiments)+1), NYTscrap_world_article_sentiments, 'm+', label='NYT world news articles')
plt.axis([0, 101, -4, 4])
plt.legend(loc='upper right')
plt.title('NYT average article sentiments')
plt.xlabel('Article index')
plt.ylabel('Average Sentiment Intensity')
# absolute average (NYT)
plt.subplot(2, 2, 2)
plt.plot(np.arange(1, len(NYTscrap_opinion_art_abs)+1), NYTscrap_opinion_art_abs, 'g+', label='NYT opinion articles')
plt.plot(np.arange(1, len(NYTscrap_world_art_abs)+1), NYTscrap_world_art_abs, 'm+', label='NYT world news articles')
plt.axis([0, 101, -4, 4])
plt.legend(loc='upper right')
plt.title('NYT absolute article sentiments')
plt.xlabel('Article index')
plt.ylabel('Average Sentiment Intensity')

ax = plt.subplot(2, 2, 3)
df_news_NYTscrap_article_avg.plot.kde(title='NYT average article sentiments KDE', ax=ax, color='gm')
ax.legend(loc='upper right')

ax = plt.subplot(2, 2, 4)
df_news_NYTscrap_article_abs.plot.kde(title='NYT average absolute article sentiments KDE', ax=ax, color='gm')
ax.legend(loc='upper right')

plt.subplot(2, 2, 1)

plt.show()


## control: have this else where

#control_art_list_op = ['Love him, hes the best. every one of his policies are GREAT!! Good man.', 'Love him, hes the best. every one of his policies are GREAT!! Good man.']
#control_art_list_nu = ['hate him, hes a bad bloke. Hes violent, annoying and smelly. worst person ever.', 'I dislike him. Hes ugly, rude and bitter.']
#op_sentence_list, op_sentence_sentiments, op_article_sentiments = NSA.article_list_analysis(control_art_list_op)
#nu_sentence_list, nu_sentence_sentiments, nu_article_sentiments = NSA.article_list_analysis(control_art_list_nu)
#
#op_art_abs = NSA.cal_article_abs_sentiment(op_sentence_sentiments)
#nu_art_abs = NSA.cal_article_abs_sentiment(nu_sentence_sentiments)
#
#plt.subplot(2, 2, 3)
#plt.plot(np.arange(1, len(op_article_sentiments)+1), op_article_sentiments, 'g+', label='control positive')
#plt.plot(np.arange(1, len(nu_article_sentiments)+1), nu_article_sentiments, 'm+', label='control negative')
#plt.axis([0, 101, -4, 4])
#plt.legend(loc='upper right')
#plt.title('Control experiment')
#plt.xlabel('Article index')
#plt.ylabel('Average Sentiment Intensity')