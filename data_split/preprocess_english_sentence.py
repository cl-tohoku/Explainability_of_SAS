from nltk import word_tokenize
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
import argparse
import logzero
from logzero import logger
import logging
from os import path
from typing import List

stopword_set = set(stopwords.words())
stemmer = PorterStemmer()
def preprocess(sentence, remove_stopwords=False, is_steming=False, is_lower=False):        
    sentence = word_tokenize(sentence)  # word tokenize
    if remove_stopwords:
        sentence = [word for word in sentence if word not in stopword_set]  # delete stopword
    if is_steming:
        sentence = [stemmer.stem(word) for word in sentence]  # stemming
    if is_lower:
        sentence = [word.lower() for word in sentence]  # lower-casing
    return sentence