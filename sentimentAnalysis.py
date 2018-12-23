# DATA

# dictionary of positive words and their values
posWords = {
    "good": 1,
    "great": 2,
    "fantastic": 3
}

# text to be analysed
text = "Einstein was a fantastic scientist and a great person. \
Newton was also a fantastic scientist but apparently he was not a good person."


# FUNCTIONS

# takes a list of words
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

textWords = text.split()
posTot, posList = findPosWords(textWords)
posText = " ".join(posList)

print(posText)




# delete all repititions, all tenses and alternations of each word (people, ), remove capitals, delete words like 'and' etc
