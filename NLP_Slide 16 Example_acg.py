# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 09:43:55 2021

@author: Travis Martin
https://www.nltk.org/api/nltk.tokenize.html
"""


import requests
from bs4 import BeautifulSoup

import nltk
from nltk.tokenize import TweetTokenizer

import gensim


from nltk.corpus import stopwords

webpage= "https://www.trianglemtb.com/"
#couple of different pages
Req = requests.get(webpage)
#Req = requests.get("https://formula1.com/")

#print(Req.text[0:100])
#print(Req.text)


SoupText = BeautifulSoup(Req.text, features="lxml")
#print(SoupText)


PageText = SoupText.get_text()
#print(PageText[2200:2500])
#print(PageText)


#tokenization
# nltk.download('punkt')
sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
Punctuation = sentence_detector.tokenize(PageText.strip())
#print(Punctuation)


Sentences = nltk.tokenize.sent_tokenize(PageText)
# print(Sentences)


Words = nltk.tokenize.word_tokenize(PageText)
# print(Words)


Twitter = TweetTokenizer()
# print(Twitter.tokenize(PageText))


# nltk.download('stopwords')

# print("\n")
# for word in stopwords.words('english'):
#    print(word)

stopchars= [",","'","''","``",":","}","{","(",")","[","]","#",".","url",";","The","-","?","'s","...","--","http","!","//","%"]

dataset = []
for word in Words:
    if word not in stopwords.words('english') and word not in stopchars:
        dataset.append([word])

# for word in Words:
#     if word not in stopchars:
#         dataset.append([word])

print(dataset)

Model = gensim.models.Word2Vec(dataset, min_count=2)
print(f"\n\nThe Key to Word2Vec Index Model is: {Model.wv.key_to_index}")


comp_word= "route"
#vector = Model.wv['Lewis']
vector = Model.wv[comp_word]
similar_words = Model.wv.most_similar(comp_word, topn=10)
# print(f"\n \n The similar word to vector)
print(f"\n\nThe vector model for the word '{comp_word}' from the wepage [{webpage}] is: \n{vector}")
print(f"\n\nThe following words from [{webpage}] have the highest similarity to the word '{comp_word}': \n{similar_words}")







