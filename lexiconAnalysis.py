import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def lexToDataFrame(VADER_lexicon):
    lex_dict = {}
    sentiment_list = []
    words_list = []
    for line in VADER_lexicon:
        (word, sentiment) = line.strip().split('\t')[0:2]
        sentiment_list = sentiment_list + [float(sentiment)]
        words_list = words_list + [word]
    lex_df = pd.DataFrame({'lex' : words_list, 'sent' : sentiment_list})
    return lex_df

VADER_lexicon = open("VADER_lexicon.txt", "r")
lex_df = lexToDataFrame(VADER_lexicon)
#print(lex_df['sent'])

fig, axes = plt.subplots()
lex_df['sent'].plot.kde(title='VADER lexicon KDE', ax = axes)

good_line = plt.axvline(x=lex_df.iloc[3327]['sent'], color = 'g', label= "\"" + lex_df.iloc[3327]['lex'] + "\"")
#bad_line = plt.axvline(x=lex_df.iloc[943]['sent'], color = 'r', label=lex_df.iloc[943]['lex'])
careless_line = plt.axvline(x=lex_df.iloc[1272]['sent'], color = 'r', label= "\"" + lex_df.iloc[1272]['lex'] + "\"")

neut_bound = plt.axvline(x=-0.2, color = 'y', linestyle = '--', label="neutral bound")
plt.axvline(x=+0.2, color = 'y', linestyle = '--')

plt.legend(handles=[good_line, neut_bound, careless_line], loc='upper right')
plt.xlabel('Sentiment Intensity')
plt.show()