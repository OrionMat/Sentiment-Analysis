# AIM:
'''
input:
- spread-sheet of positive and negative words
- article

output:
- 'total sentiment score'
- article with the highlighted words

problems it should handel:
- negation i.e not good
- subject location
'''

# DATA

# read spread-sheet into a dictionary 

# dictionary of positive words and their values
posWords = {
    "good": 1,
    "great": 2,
    "fantastic": 3
}

# read file into a string

# text to be analysed
text = "Einstein was a fantastic scientist and a great person. \
Newton was also a fantastic scientist but apparently he was not a good person."


# FUNCTIONS

# takes a list of (positive) words *** can change so it takes general list of words and argument saying if words are positive or negative
# returns a tuple of the total positive score of the list and a modified list 
def findPosWords(wordsList):
    total = 0

    wordsListPos = wordsList
    for idx, word in enumerate(wordsList):
        if word in posWords:
            total = total + posWords[word]
            wordsListPos[idx] = wordsList[idx] + "~(+" + str(posWords[word]) + ")"

    return total, wordsListPos


# CODE

# split string into list of words
textWords = text.split()

# find the positive words
# give an overall score
# highlight the words in the article
posTot, posList = findPosWords(textWords)
posText = " ".join(posList)

print(posText)



# find way of matching?
# delete all repititions, all tenses and alternations of each word (people, ),
# remove capitals, delete words like 'and' etc
