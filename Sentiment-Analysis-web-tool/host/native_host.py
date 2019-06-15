import sys
import json
import struct
import csv
import random
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import numpy as np

# Function to send a message to Chrome.
def send_message(MSG_DICT):
    msg_json = json.dumps(MSG_DICT, separators=(",", ":"))          # Converts dictionary into string containing JSON format.
    msg_json_utf8 = msg_json.encode("utf-8")                        # Encodes string with UTF-8.
    sys.stdout.buffer.write(struct.pack("i", len(msg_json_utf8)))   # Writes the message size. (Writing to buffer because writing bytes object.)
    sys.stdout.buffer.write(msg_json_utf8)                          # Writes the message itself. (Writing to buffer because writing bytes object.)

# Function to read a message from Chrome.
def read_message():
    text_length_bytes = sys.stdin.buffer.read(4)                        # Reads the first 4 bytes of the message (which designates message length).
    text_length = struct.unpack("i", text_length_bytes)[0]              # Unpacks the first 4 bytes that are the message length. [0] required because unpack returns tuple with required data at index 0.
    text_decoded = sys.stdin.buffer.read(text_length).decode("utf-8")   # Reads and decodes the text (which is JSON) of the message. [...] Then use the data.
    text_dict = json.loads(text_decoded)
    return text_dict

# input: text (string) 
# output: list of sentences in text and list of corresponding corresponding sentiment scores
# sentiment is rating between -4 (EXTREAMLY negative) and 4 (EXTREAMLY positive)
def calcSentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentence_list = nltk.sent_tokenize(text)
    sentiment_list = [] 
    if not sentence_list:
        return sentence_list, sentiment_list
        #raise ValueError('no sentences in the text given -> 0/0')
    for sentence in sentence_list:
            vs = analyzer.polarity_scores(sentence)
            compound_sentiment = round(vs["compound"]*4, 4)
            sentiment_list = sentiment_list + [compound_sentiment]
    return sentence_list, sentiment_list

def results_calculation(sentiment_list):
    sentiment_array = np.round(np.asarray(sentiment_list),  decimals=3, out=None)
    sentiment_avg = np.round(np.mean(sentiment_array),  decimals=3, out=None)
    sentiment_avg_abs = np.round(np.mean(np.absolute(sentiment_array)),  decimals=3, out=None)
    sentiment_var = np.round(np.var(sentiment_array),  decimals=3, out=None)
    return sentiment_avg, sentiment_avg_abs, sentiment_var

send_message({"name" : "test", "text" : "testing messaging"})

query_dict = read_message()
query = query_dict['text']
# query = "Can Trump declare a national emergency to build a wall?. Two London teens killed within 15 minutes of each other. I am sickened to hear that two young lives have been ended within minutes of each. Tooting MP Rosena Allin-Khan said the killing was heartbreaking and absolutely tragic. Today is the most WONDERFUL day. LOVE, LOVE LOVE."

sentence_list, sentiment_list = calcSentiment(query)
# for idx in range(len(sentence_list)):
#     print(str(sentence_list[idx]) + " : " + str(sentiment_list[idx]) + "\n")

sentiment_avg, sentiment_avg_abs, sentiment_var = results_calculation(sentiment_list)
# print(sentiment_avg, sentiment_avg_abs, sentiment_var)

result_list = [sentiment_avg] + [sentiment_avg_abs]
results = json.dumps(result_list)
send_message({"name" : "articleResults", "text" : "sending results of articles", "results" : results})


# print(results)