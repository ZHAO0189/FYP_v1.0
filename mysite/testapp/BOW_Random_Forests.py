import os
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.externals import joblib

#Three classifier: SVM, random forest and neural network.

directory = 'E:\\A1113\\FYP\\NEWS\\Labeled Corpus\\All\\'

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

def read_documents(directory):
    corpus = []
    label = []
    for file in os.listdir(directory):
        print "Loading File: " + file
        entity = file.split(".")[0]

        if file.find('.json') > 0:
            with open(directory+file) as json_data:
                data = json.load(json_data)
                li = data[entity]

                for item in li:
                    corpus.append(clean_str(item['content']))
                    label.append(item['label'])

    print "All files have been read and loaded."

    return corpus, label

def bow_random_forests(directory):
    corpus, label = read_documents(directory)
    print "Corpus Size: " + str(len(corpus))
    print "Label Size: " + str(len(label))

    """
    vectorize the corpus and convert it to a matrix. in the matrix, a row is a document, a column is a token(word).
    """
    vectorizer = CountVectorizer(min_df=1)
    sparse_matrix = vectorizer.fit_transform(corpus)
    print("Original BOW Matrix Shape: ")
    print(sparse_matrix.toarray().shape)
#print(vectorizer.vocabulary_.get('amazon'))
#print(sparse_matrix.toarray()[0][519])

    """
    tf-idf weighting
    """
    transformer = TfidfTransformer(smooth_idf=True)
    tfidf = transformer.fit_transform(sparse_matrix)
    tfidf = tfidf.todense()
    tfidf = np.array(tfidf)

    """
    SVD for LSA
    """
    svd = TruncatedSVD(300)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    reduced_matrix = lsa.fit_transform(tfidf)
    print(reduced_matrix.shape)

    print("Reduced Matrix Shape: ")
    print reduced_matrix.shape
    #print("Tf-idf Matrix Shape: ")
    #print tfidf.shape

    cv_index = []
    #Generate a list of random numbers for cross validation use.
    for i in range(len(corpus)):
        cv_index.append(np.random.randint(0, 10))

    #Start 10-fold Cross Validation:
    score_array = []
    max_score = 0.0

    for j in range (10):
        #datasets, labelsets = make_idx_data_cv(tfidf, label, cv_index, j)
        datasets, labelsets = make_idx_data_cv(reduced_matrix, label, cv_index, j)

        """
        #Random Forests Classification
        """
        print "==================Start Random Forests Classifier Training " + str(j + 1) + "===================="
        clf = RandomForestClassifier(n_estimators=10, min_samples_split=2, bootstrap=True, random_state=None, class_weight=None)

        print "Training Dataset Shape: "
        print datasets[0].shape

        clf = clf.fit(datasets[0], labelsets[0])
        clf.fit(datasets[0], labelsets[0])

        print "Training Complete."

        # Start predicting testing corpus.
        print "Start Predicting..."
        score = clf.score(datasets[1], labelsets[1])
        score_array.append(score)
        #if score > max_score:
            #max_score = score
            # Save classifier using Joblib. this method can only pickle a model to the disk
            # To load back the model, use "clf = joblib.load("filename.pkl")."
            #joblib.dump(clf, 'E:/A1113/FYP/BloombergNews/BloombergNews/RF+BOW.pkl')
        print "Testing data accuracy :"
        print score

    print "====================Cross Validation Complete=============================="
    #print "Highest accuracy is " + str(max_score)
    print "Average accuracy is " + str(np.mean(score_array))
    #print "Model with Highest Accuracy Saved at E:/A1113/FYP/BloombergNews/BloombergNews/RF+BOW.pkl"

bow_random_forests(directory)