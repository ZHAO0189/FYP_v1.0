import cPickle
import re


def load_pkl_vec(fname, vocab):
    """
    Load 300x1 vecs from self generated w2v
    :param fname: address and filename
    :param vocab: vocab is the features
    :return: vecs for all features
    """
    word_vecs = {}
    num_found = 0
    print ("Inside load_pkl_vec function.")
    #"E:\\A1113\\FYP\\BloombergNews\\BloombergNews\w2v.pkl"
    pkl_file = open(fname, 'rb')
    w2v = cPickle.load(pkl_file)
    pkl_file.close()

    for word in w2v:
        if word in vocab:
            num_found += 1
            word_vecs[word] = w2v[word]
            print "Found word vectors: " + word

    print "words found: %d in total words: %d" % (num_found, len(vocab))
    return word_vecs


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
    string = re.sub(r",", "  ", string)
    # string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    # string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\(", " ", string)
    # string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    # string = re.sub(r"\d", "NUMNUMNUM", string)
    string = re.sub(r"\d", " ", string)
    return string.strip() if TREC else string.strip().lower()