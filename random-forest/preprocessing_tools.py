import re
import nltk
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

# convert input document to a sequence of words, remove stop words (optionally)
def review_to_wordlist( review, remove_stopwords=True ):
    # remove html
    review_text = BeautifulSoup(review).get_text()
    # remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    # words to lower and split
    words = review_text.lower().split()
    # reremove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return(words)
    
# split a input into parsed sentences, return list of sentenses 
# (wsentence is a list of words)
def review_to_sentences( review, tokenizer, remove_stopwords=True ):
    # split the paragraph into sentences (use nltk tokenizer)
    raw_sentences = tokenizer.tokenize(review.decode('utf8').strip())
    # loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # skip it if sentence is empty
        if len(raw_sentence) > 0:
            # else, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, remove_stopwords ))
    return sentences