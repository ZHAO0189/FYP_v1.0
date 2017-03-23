import json
import os
import nltk.data
from nltk.corpus import stopwords
import re

dirname = '/Users/zhaozinian/Documents/UNIVERSITY/FYP/NEWS/Labeled Corpus/train/'
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def my_sentences(dirname):
    corpus = []

    for file in os.listdir(dirname):
        entity = file.split('_')[0]
        if file.find('.json') > 0:
            with open(dirname + file) as json_data:
                data = json.load(json_data)
                li = data[entity]
                for item in li:
                    content = item['content']
                    corpus+=article_to_sentences(content,tokenizer)

    return corpus

def sentence_to_wordlist(sentence,remove_stopwords=False):
    #Function to convert a sentence into a list of words.
    sentence = clean_str(sentence)
    words = sentence.split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    return words

def article_to_sentences(article, tokenizer, remove_stopwords=False):
    #Function to convert an article into a list of sentences, each sentence is a list of words.
    raw_sentences = tokenizer.tokenize(article)
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(sentence_to_wordlist(raw_sentence,remove_stopwords))

    return sentences

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    #string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\(", " ", string)
    #string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    #string = re.sub(r"\d", "NUMNUMNUM", string)
    string = re.sub(r"\d", " ", string)
    return string.strip() if TREC else string.strip().lower()



sentences = my_sentences(dirname)

# Import the built-in logging module and configure it so that Word2Vec
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality
min_word_count = 10   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 15          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print ("Training model...")
model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()

#model_name = "300features_10minwords_10context"
#model.save(model_name)

print(model.doesnt_match("facebook google twitter speech".split()))