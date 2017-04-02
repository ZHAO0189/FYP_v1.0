import json
import numpy as np
import os
import re
import cPickle
import matplotlib.pyplot as plt
from io import open
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# mean for whole vocab, doc word vec ++, divided by len(doc), repeated word also ++
# word embeddings: save vectors to .p file using cpickle
directory = '/Users/zhaozinian/Documents/UNIVERSITY/FYP/NEWS/Labeled Corpus/All/'


def read_documents(directory):
    # Read file from directory. Return a list of articles.
    corpus = []
    label = []
    for file in os.listdir(directory):
        entity = file.split('.')[0]
        print file

        if file.find('.json') >=0:
            with open(directory + file) as json_data:
                data = json.load(json_data)
                li = data[entity]

                for item in li:
                    corpus.append(clean_str(item['content']))
                    label.append(item['label'])

    print "All files have been read and loaded."
    return corpus, label


def rand_num_array(length):
    cv_idx = []

    for i in range(length):
        cv_idx.append(np.random.randint(0, 10))

    return cv_idx


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


def make_idx_data_cv(data, label, cv_index, cv):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    train_label, test_label = [], []

    #Change Positive, Negative, Neutral to 1, -1, 0.
    idx_neg = [idx for idx in range(len(label)) if label[idx]==u'Negative']
    labels_num = np.ones(len(label))
    labels_num[idx_neg] = -1
    idx_neu = [idx for idx in range(len(label)) if label[idx]==u'Neutral']
    labels_num[idx_neu] = 0
    print labels_num

    #rev_old is a list, but in numpy array, it is a 2-dimension array. If not flattened, the datasets will be 3-dimension.
    for index, rev in enumerate(data):

        if cv_index[index] == cv:
            test.append(rev)
            test_label.append(labels_num[index])
        else:
            train.append(rev)
            train_label.append(labels_num[index])
    train = np.array(train, dtype="float")
    test = np.array(test, dtype="float",ndmin=2)
    return [train, test], [train_label, test_label]


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    num_found = 0
    print ("Inside load_bin_vec function.")
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)

                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)

            word = word.decode('utf-8', 'ignore')

            if word in vocab:
                num_found = num_found + 1
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
                print "Found word vector: " + word
            else:
                f.read(binary_len)

    print "words found: %d in total words: %d" % (num_found, len(vocab))
    return word_vecs


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


def add_unknown_words(word_vecs, vocab, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size, k))
    W[0] = np.zeros(k)
    i = 0
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def average_embedding_w2v(word_vecs, vocabulary, k=300):
    # Get new w2v which has been subtracted by the mean
    w2v = dict()
    new_embed = np.zeros((1, k))
    for word in vocabulary:
        new_embed = new_embed + word_vecs[word]
    mean = new_embed / len(vocabulary)

    for word in vocabulary:
        w2v[word] = word_vecs[word] - mean

    print "Word Vectors have been subtracted by its mean vector. "
    return w2v


def average_embedding_news(corpus, word_vecs, vocabulary, k=300):
    # Get list of vecs for each news using average_embedding_w2v
    arr = np.empty((0, k))
    for news in corpus:
        print "Calculating News Vector...."
        new_embed = np.zeros((1, k))
        h = 0
        news = news.split(' ')
        for word in news:
            if word in word_vecs.keys():
                new_embed = new_embed + word_vecs[word]
                h += 1
        new_embed = new_embed / h
        arr = np.vstack([arr, new_embed])
    print "News Vectors have been calculated. Size:"
    print len(arr)

    return arr


def svm_we_training(news_vec, label, cv_index, c):
    # Start 10-fold Cross Validation:
    score_array = []

    for j in range(10):
        datasets, labelsets = make_idx_data_cv(news_vec, label, cv_index, j)

        """
        #SVM Classification
        """
        print "====================Start SVM Classifier Training " + str(j + 1) + "=============================="
        #clf = svm.SVC(kernel='rbf', C=c)
        clf = svm.LinearSVC(C=c)
        print "Training Dataset Shape: "
        print datasets[0].shape

        clf.fit(datasets[0], labelsets[0])
        print "Training Complete."

        # Start predicting testing corpus.
        print "Start Predicting..."
        score = clf.score(datasets[1], labelsets[1])
        score_array.append(score)
        # if score > max_score:
        # max_score = score
        # Save classifier using Joblib. this method can only pickle a model to the disk
        # To load back the model, use "clf = joblib.load("filename.pkl")."
        # joblib.dump(clf, 'E:/A1113/FYP/BloombergNews/BloombergNews/SVM+BOW.pkl')
        print "Testing data accuracy :"
        print score

    print "====================Cross Validation for C = " + str(c) +"Complete=============================="
    # print "Highest accuracy is " + str(max_score)
    print "Average accuracy of SVM is " + str(np.mean(score_array))
    # print "Model with Highest Accuracy Saved at E:/A1113/FYP/BloombergNews/BloombergNews/SVM+BOW.pkl"

    return score_array


def rf_we_training(news_vec, label, cv_index):
    # Start 10-fold Cross Validation:
    score_array = []
    max_score = 0.0

    clf = RandomForestClassifier(n_estimators=350, min_samples_split=2, bootstrap=True, random_state=None,
                                 max_depth=15, class_weight=None)
    for j in range(10):
        datasets, labelsets = make_idx_data_cv(news_vec, label, cv_index, j)


        """
        #Random Forests Classification
        """

        print "==================Start Random Forests Classifier Training " + str(j + 1) + "===================="
        print "Training Dataset Shape: "
        print datasets[0].shape

        clf = clf.fit(datasets[0], labelsets[0])
        clf.fit(datasets[0], labelsets[0])

        print "Training Complete."
        clf.score(datasets[0],labelsets[0])
        # Start predicting testing corpus.
        print "Start Predicting..."
        score = clf.score(datasets[1], labelsets[1])
        score_array.append(score)
        # if score > max_score:
        # max_score = score
        # Save classifier using Joblib. this method can only pickle a model to the disk
        # To load back the model, use "clf = joblib.load("filename.pkl")."
        #joblib.dump(clf, '/Users/zhaozinian/Desktop/trydjango18/mysite/WE_RF_350_15.pkl')
        print "Testing data accuracy :"
        print score

    print "====================Cross Validation Complete=============================="
    # print "Highest accuracy is " + str(max_score)
    print "Average accuracy of Random Forests is " + str(np.mean(score_array))
    # print "Model with Highest Accuracy Saved at E:/A1113/FYP/BloombergNews/BloombergNews/RF+BOW.pkl"
    joblib.dump(clf, '/Users/zhaozinian/Desktop/trydjango18/mysite/WE_RF_350_15.pkl')
    return score_array


def nn_we_training(news_vec, label, cv_index):
    # Start 10-fold Cross Validation:
    score_array = []
    max_score = 0.0

    for j in range(10):
        datasets, labelsets = make_idx_data_cv(news_vec, label, cv_index, j)

        """
        #Neural Networks Classification
        """
        print "====================Start Neural Networks Classifier Training " + str(
            j + 1) + "=============================="
        clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(450,675), random_state=1, activation='tanh',
                            batch_size=25)
        print "Training Dataset Shape: "
        print datasets[0].shape

        clf.fit(datasets[0], labelsets[0])
        print "Training Complete."

        # Start predicting testing corpus.
        print "Start Predicting..."
        score = clf.score(datasets[1], labelsets[1])
        score_array.append(score)
        # if score > max_score:
        # max_score = score
        # Save classifier using Joblib. this method can only pickle a model to the disk
        # To load back the model, use "clf = joblib.load("filename.pkl")."
        # joblib.dump(clf, 'E:/A1113/FYP/BloombergNews/BloombergNews/SVM+BOW.pkl')
        print "Testing data accuracy :"
        print score

    print "====================Cross Validation Complete=============================="
    # print "Highest accuracy is " + str(max_score)
    print "Average accuracy of Neural Networks is " + str(np.mean(score_array))
    # print "Model with Highest Accuracy Saved at E:/A1113/FYP/BloombergNews/BloombergNews/SVM+BOW.pkl"

    return score_array


def we_predictions(corpus):
    clf = joblib.load("/Users/zhaozinian/Desktop/trydjango18/mysite/WE_RF_350_15.pkl")
    positive, negative, neutral = [0.0 for _ in range(3)]


    vectorizer = CountVectorizer(min_df=1, strip_accents='ascii')
    sparse_matrix = vectorizer.fit_transform(corpus)
    vocab = vectorizer.get_feature_names()

    w2v = load_pkl_vec('/Users/zhaozinian/Desktop/trydjango18/mysite/testapp/w2v_non_mean.pkl', vocab)
    print "Word Embeddings Loaded."
    news_vec = average_embedding_news(corpus, w2v, vocab)
    predictions = clf.predict(news_vec)
    print predictions
    predictions = ["Positive" if x==1.0 else x for x in predictions]
    predictions = ["Negative" if x == -1.0 else x for x in predictions]
    predictions = ["Neutral" if x == 0.0 else x for x in predictions]

    confidence =  clf.predict_proba(news_vec)
    for each in confidence:
        negative += each[0]
        neutral += each[1]
        positive += each[2]

    positive /= len(confidence)
    negative /= len(confidence)
    neutral /= len(confidence)
    print confidence
    print clf.classes_
    print positive, negative, neutral
    build_plot(predictions)
    return predictions, positive, negative, neutral


def build_plot(predictions):
    idx_neg = [idx for idx in range(len(predictions)) if predictions[idx] == 'Negative']
    neg = len(idx_neg)

    idx_pos = [idx for idx in range(len(predictions)) if predictions[idx] == 'Positive']
    pos = len(idx_pos)

    idx_neu = [idx for idx in range(len(predictions)) if predictions[idx] == 'Neutral']
    neu = len(idx_neu)

    labels = ["Positive", "Negative", "Neutral"]
    data = [pos, neg, neu]
    colors = ['#F1A94E', '#E45641', '#5D4C46']

    xlocations = np.arange(len(data))
    width = 0.5

    fig, ax = plt.subplots()
    rects = ax.bar(xlocations, data, width, color=colors)
    ax.set_ylabel("Number")
    ax.set_xlabel("Categories")
    ax.set_xticks(xlocations)
    ax.set_xticklabels(labels)
    ax.set_title('Classification Distribution')
    lim = max(data)
    ax.set_ylim([0, lim + 1])
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                '%d' % int(height),
                ha='center', va='bottom')

    plt.savefig('/Users/zhaozinian/Desktop/trydjango18/mysite/static/our_static/img/foo.png')


"""
corpus, label = read_documents(directory)
print "Corpus Size:"
print len(corpus)
print "Label Size:"
print len(label)


vectorizer = CountVectorizer(min_df=1, strip_accents='ascii')
sparse_matrix = vectorizer.fit_transform(corpus)
vocab = vectorizer.get_feature_names()
print 'Vocabulary Loaded. Vocabulary Size:'
print len(vocab)

w2v = load_pkl_vec('/Users/zhaozinian/Desktop/trydjango18/mysite/testapp/w2v_non_mean.pkl', vocab)
print "Word Embeddings Loaded."
#add_unknown_words(w2v, vocab, k=300)
#W, word_idx_map = get_W(w2v, k=300)

#avg_w2v = average_embedding_w2v(w2v, vocab)
news_vec = average_embedding_news(corpus, w2v, vocab)

cv_index = rand_num_array(len(corpus))


c = [0.1,1,10,100]
svm_score = {}
n_estimators = [250,300,350,400]
max = [10,15,20]
rf_score = {}


#for num in c:
 #   svm_score[num] = np.mean(svm_we_training(news_vec, label, cv_index,num))


rf_score = np.mean(rf_we_training(news_vec, label, cv_index))
#nn_score = nn_we_training(news_vec, label, cv_index)

print " "
#for num in c:
 #   print "SVM Score for c= "+ str(num)+" : " + str(svm_score[num])


print "Random Forests Score: " + str(rf_score)
#print "Neural Networks Score: " + str(np.mean(nn_score))
"""

# Global Vectors => another word embeddings method
