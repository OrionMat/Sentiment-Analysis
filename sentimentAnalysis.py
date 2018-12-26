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
article = open("article.txt", "r")
text = article.read()

# text to be analysed
'''text = "Einstein was a fantastic scientist and a great person. \
Newton was also a fantastic scientist but apparently he was not a good person. \
in other words, Newton was potentially a horrible person to some people \
though you could not call him frail or slight"'''


# FUNCTIONS

# takes a list of words and a dictionary where the words are (potential) keys and word scores are values
# returns a tuple of the total score of the list and a modified list 
def scoreWords(textWords_list, words_dict):
    totalScore = 0
    totalPos = 0
    totalNeg = 0

    textWordsCopy_list = textWords_list
    for idx, word in enumerate(textWords_list):
        if word in words_dict:
            totalScore = totalScore + words_dict[word]
            textWordsCopy_list[idx] = textWords_list[idx] + "~(" + str(words_dict[word]) + ")"
            if words_dict[word] > 0:
                totalPos = totalPos + 1
            elif words_dict[word] < 0:
                totalNeg = totalNeg + 1
            else:
                print("word has a score of 0 - should never occure")

    return totalScore, totalPos, totalNeg, textWordsCopy_list


# CODE

# split string into list of words
textWords = text.split()

# finds the positive\negative words
# give an overall score
# highlight the words in the article
total, posTotal, negTotal, modifiedList = scoreWords(textWords, posNegWords_dict)
posText = " ".join(modifiedList)

print("\n" + posText + "\n")
print("total = " + str(total) + ", num_posWords = " + str(posTotal), ", num_negWords = " + str(negTotal))

# find way of matching?
# delete all repititions, all tenses and alternations of each word (people, ),
# remove capitals, delete words like 'and' etc