# AIM:
'''
input:
- spread-sheet of positive and negative words
- article

output:
- 'total sentiment score'
- article with the highlighted positive\negative words

problems it should handel:
- negation i.e not good
- subject location
'''


# LIBRARIES
import numpy as np
import pandas as pd


# USER INPUT
filename = 'words.xlsx' # name of file containing positive/negative words


# DATA

# read spread-sheet into a dictionary 
# dictionary containes (positive) words (key) and their scores (value)
file = pd.ExcelFile(filename)
df = file.parse(0)
posWords_list = df['positive words:'].tolist()
posScores_list = df['positive scores:'].tolist()
negWords_list = df['negative words:'].tolist()
negScores_list = df['negative scores:'].tolist()
posWordScore_list = list(zip(posWords_list, posScores_list))
negWordScore_list = list(zip(negWords_list, negScores_list))
WordScore_list = posWordScore_list + negWordScore_list
posNegWords_dict = dict(WordScore_list)   

print("data-frame: \n")
print(df)   # prints dataframe
print('\n')
print(posNegWords_dict)

# read file into a string

# text to be analysed
text = "Einstein was a fantastic scientist and a great person. \
Newton was also a fantastic scientist but apparently he was not a good person. \
in other words, Newton was potentially a horrible person to some people \
though you could not call him frail or slight"


# FUNCTIONS

# takes a list of words and a dictionary
# returns a tuple of the total score of the list and a modified list 
def scoreWords(textWords_list, words_dict):
    total = 0

    textWordsCopy_list = textWords_list
    for idx, word in enumerate(textWords_list):
        if word in words_dict:
            total = total + words_dict[word]
            textWordsCopy_list[idx] = textWords_list[idx] + "~(" + str(words_dict[word]) + ")"

    return total, textWordsCopy_list


# CODE

# split string into list of words
textWords = text.split()

# find the positive words
# give an overall score
# highlight the words in the article
posTot, posList = scoreWords(textWords, posNegWords_dict)
posText = " ".join(posList)

print(posText)

# find way of matching?
# delete all repititions, all tenses and alternations of each word (people, ),
# remove capitals, delete words like 'and' etc