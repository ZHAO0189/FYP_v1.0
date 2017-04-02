import nltk
from nltk.corpus import stopwords
import re
import cPickle
from nltk.tokenize import word_tokenize
import heapq
import json


def article_to_sentences(article):
    #Function to convert an article into a list of sentences, each sentence is a list of words.
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(article)
    return raw_sentences


def calculate_sentence_score(sentences_list):
    pkl_file = open("/Users/zhaozinian/Desktop/trydjango18/mysite/testapp/sentiWordNetDict.pkl", 'rb')
    dictionary = cPickle.load(pkl_file)
    pkl_file.close()
    sentence_score = {}

    for sentence in sentences_list:
        score = 0.0
        text = word_tokenize(sentence)
        sentence_tags = nltk.pos_tag(text)
        for item in sentence_tags:
            word = item[0]
            tag = item[1]
            if tag == 'JJ' or tag == 'JJR' or tag == 'JJS':
                word = word.lower() + "#" + "a"
            elif tag == 'NN' or tag == 'NNS' or tag == 'NNP' or tag == 'NNPS':
                word = word.lower() + "#" + "n"
            elif tag == 'RB' or tag == 'RBR' or tag == 'RBS' or tag == 'RP':
                word = word.lower() + "#" + "r"
            elif  tag == 'VB' or tag == 'VBD' or tag == 'VBG' or tag == 'VBN' or tag == 'VBP' or tag == 'VBZ':
                word = word.lower() + "#" + "v"
            else:
                continue

            if word in dictionary:
                score += dictionary[word]

        sentence_score[sentence] = score

    return sentence_score


def sentence_to_wordlist(sentence,remove_stopwords=False):
    #Function to convert a sentence into a list of words.
    sentence = clean_str(sentence)
    words = sentence.split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]

    return words


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


def find_pos_sentences(sentence_score):
    maxs = heapq.nlargest(3, sentence_score.itervalues())
    list = []
    for key in sentence_score:
        if sentence_score[key] in maxs and sentence_score[key]>0.5:
            key = key.encode('ascii', 'ignore')
            list.append(
                re.sub("\n"," ",key)
            )

    print list
    print len(list)
    return list


def find_neg_sentences(sentence_score):
    mins = heapq.nsmallest(3, sentence_score.itervalues())
    list = []
    for key in sentence_score:
        if sentence_score[key] in mins and sentence_score[key]<-0.5:
            key = key.encode('ascii', 'ignore')
            list.append(
                re.sub("\n", " ", key)
            )

    print list
    print len(list)
    return list


def sentence_extraction(corpus, predictions):
    results = {}

    for i in range(len(predictions)):

        if predictions[i] == "Negative":
            sentences = article_to_sentences(corpus[i])
            sentence_score = calculate_sentence_score(sentences)
            neg_sentences = find_neg_sentences(sentence_score)
            results[i] = neg_sentences

        if predictions[i] == "Positive":
            sentences = article_to_sentences(corpus[i])
            sentence_score = calculate_sentence_score(sentences)
            pos_sentences = find_pos_sentences(sentence_score)
            results[i] = pos_sentences

        if predictions[i] == "Neutral":
            results[i] = "null"

    return results

"""
with open('/Users/zhaozinian/Documents/UNIVERSITY/FYP/NEWS/Bloomberg-Google-2017-03-24--1w.json') as json_data:
    corpus = []
    data = json.load(json_data)
    news = data['Google']
    for item in news:
        corpus.append(item['content'])
    predictions = ["Positive","Positive","Positive","Negative","Positive","Positive","Positive","Positive"]
    results = sentence_extraction(corpus,predictions)
    print results
"""