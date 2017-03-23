import os
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import json
from sklearn import svm
from sklearn.externals import joblib

#Three classifier: SVM, random forest and neural network.
#Current work: 1) WordEmb: matrix row word, sum(vector),mean, mean_vector, word_vecto-mean
#              2) try param, SVM:c, RF:all.
#              3) only average accounts. max for models.
#              4) f1 score, precision score


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


def bow(corpus):
    """
    vectorize the corpus and convert it to a matrix. in the matrix, a row is a document, a column is a token(word).
    """
    vectorizer = CountVectorizer(min_df=1)
    sparse_matrix = vectorizer.fit_transform(corpus)
    print("Original BOW Matrix Shape: ")
    print(sparse_matrix.toarray().shape)
    # print(vectorizer.vocabulary_.get('amazon'))
    # print(sparse_matrix.toarray()[0][519])

    """
    tf-idf weighting
    """
    transformer = TfidfTransformer(smooth_idf=True)
    tfidf = transformer.fit_transform(sparse_matrix)
    tfidf = tfidf.todense()
    tfidf = np.array(tfidf)
    print("Tf-idf Matrix Shape: ")
    print tfidf.shape

    return tfidf


def rand_num_array(length):
    cv_idx = []

    for i in range(length):
        cv_idx.append(np.random.randint(0, 10))

    return cv_idx


def svm_training(label, cv_index, matrix, c):

    #Start 10-fold Cross Validation:
    score_array = []
    clf = svm.LinearSVC(C=c)

    for j in range(10):

        datasets, labelsets = make_idx_data_cv(matrix, label, cv_index, j)

        """
        #SVM Classification
        """
        print "====================Start SVM Classifier Training " + str(j+1) + "=============================="
        #clf = svm.SVC(kernel='rbf')

        print "Training Dataset Shape: "
        print datasets[0].shape

        clf.fit(datasets[0], labelsets[0])
        print "Training Complete."
        print clf.score(datasets[0],labelsets[0])
        #Start predicting testing corpus.
        print "Start Predicting..."
        score = clf.score(datasets[1], labelsets[1])
        score_array.append(score)
        #if score > max_score:
            #max_score = score
            # Save classifier using Joblib. this method can only pickle a model to the disk
            # To load back the model, use "clf = joblib.load("filename.pkl")."
            #joblib.dump(clf, 'E:/A1113/FYP/BloombergNews/BloombergNews/SVM+BOW.pkl')
        print "Testing data accuracy :"
        print score

    print "====================Cross Validation of SVM for C = "+ str(c) +"Complete=============================="
    # print "Highest accuracy is " + str(max_score)
    print "Average accuracy of SVM is " + str(np.mean(score_array))
    # print "Model with Highest Accuracy Saved at E:/A1113/FYP/BloombergNews/BloombergNews/SVM+BOW.pkl"

    return score_array


def random_forests_training(label, cv_index, matrix, max):
    # Start 10-fold Cross Validation:
    score_array = []
    max_score = 0.0

    for j in range(10):

        datasets, labelsets = make_idx_data_cv(matrix, label, cv_index, j)
        clf = RandomForestClassifier(n_estimators=350, min_samples_split=2, bootstrap=True, random_state=None,
                                     class_weight=None, max_depth=max)
        """
        #Random Forests Classification
        """
        print "==================Start Random Forests Classifier Training " + str(j + 1) + "===================="


        print "Training Dataset Shape: "
        print datasets[0].shape

        clf = clf.fit(datasets[0], labelsets[0])
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
        # joblib.dump(clf, 'E:/A1113/FYP/BloombergNews/BloombergNews/RF+BOW.pkl')
        print "Testing data accuracy :"
        print score

    print "====================Cross Validation of Random Forests Complete=============================="
    # print "Highest accuracy is " + str(max_score)
    print "Average accuracy of Random Forests is " + str(np.mean(score_array))
    # print "Model with Highest Accuracy Saved at E:/A1113/FYP/BloombergNews/BloombergNews/RF+BOW.pkl"

    return score_array


def neural_networks_training(label, cv_index, matrix):
    # Start 10-fold Cross Validation:
    score_array = []
    max_score = 0.0

    for j in range (10):

        datasets, labelsets = make_idx_data_cv(matrix, label, cv_index, j)

        """
        #Neural Networks Classification
        """
        print "====================Start Neural Networks Classifier Training " + str(j+1) + "=============================="
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(500, ), random_state=1)
        print "Training Dataset Shape: "
        print datasets[0].shape

        clf.fit(datasets[0], labelsets[0])
        print "Training Complete."

        #Start predicting testing corpus.
        print "Start Predicting..."
        score = clf.score(datasets[1], labelsets[1])
        score_array.append(score)
        #if score > max_score:
            #max_score = score
            # Save classifier using Joblib. this method can only pickle a model to the disk
            # To load back the model, use "clf = joblib.load("filename.pkl")."
            #joblib.dump(clf, 'E:/A1113/FYP/BloombergNews/BloombergNews/SVM+BOW.pkl')
        print "Testing data accuracy :"
        print score

    print "====================Cross Validation of Neural Networks Complete=============================="
    #print "Highest accuracy is " + str(max_score)
    print "Average accuracy of Neural Networks is " + str(np.mean(score_array))
    #print "Model with Highest Accuracy Saved at E:/A1113/FYP/BloombergNews/BloombergNews/SVM+BOW.pkl"

    return score_array


# Program Start From HERE.
directory = '/Users/zhaozinian/Documents/UNIVERSITY/FYP/NEWS/Labeled Corpus/All/'
corpus, label = read_documents(directory)
print "Corpus Size: " + str(len(corpus))
print "Label Size: " + str(len(label))

cv_index = rand_num_array(len(corpus))

tfidf = bow(corpus)
n_estimators = [250,300,350,400]
max = [10,15,20]
svm_score = {}
rf_score = {}

#SVM Training + BOW
#for num in c:
#    svm_score[num] = np.mean(svm_training(label, cv_index, tfidf, num))

#Random Forests Training + BOW
for num in max:
    rf_score[num] = np.mean(random_forests_training(label, cv_index, tfidf, num))

#Neural Networks Training + BOW
#nn_score = neural_networks_training(label, cv_index, tfidf)

print ""
#for num in c:
  #  print "SVM Score for C = "+ str(num) +": " + str(svm_score[num])
for num in max:
    print "Random Forests Score for n = " + str(num)+ ": "+str(rf_score[num])
#print "Neural Networks Score: " + str(np.mean(nn_score))


"""
#tsne method

model = TSNE(n_components=2, random_state=np.random)
array = model.fit_transform(reduced_matrix)

color_collect ={'Positive': 'red', 'Negative': 'blue', 'Neutral': 'black'}
color = [color_collect[i] for i in lab]
fig = plt.figure()

plt.scatter(array[:,0], array[:,1], c=color)
plt.title("TSNE Plot")
plt.show()
"""