pos_words_file = open("positive_words.txt")

# delete all repititions, all tenses and alternations of each word (people, ), remove capitals, delete words like 'and' etc
pos_words_string = pos_words_file.read()
print(pos_words_string)
print("hello")
thisdict =	{
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
print(thisdict)
print(thisdict["model"])