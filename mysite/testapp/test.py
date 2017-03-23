from All import get_bloomberg_news, get_guardian_news, get_reuters_news


news = get_bloomberg_news("google","-1w")
print len(news)
print type(news)

#request = urllib.request.Request("https://www.bloomberg.com/news/articles/2016-10-06/facebook-testing-vr-headset-that-doesn-t-need-pc-connection")
#res = urllib.request.urlopen(request)
#soup = BeautifulSoup(res, 'html.parser')

#tag = soup.find('div', {"class": "article-body_content"})
#if tag == None:
 #   tag = soup.find('div', {"class": "body-copy"})
#list = tag.find_all('p')
#print(list)

#def extract_links(txt):
 #   # regex = r'http://www\.reuters\.com/article/[\w]+'
  #  regex = r'/article/[-\w\d]+'
   # match = re.findall(regex, txt)
    #return match

#txt = 'id: "USL1N1CV211", headline: "UPDATE 1-Highlights from Reuters\' exclusive interview with Donald<\/b> Trump<\/b>", ' \
 #     'date: "October 25, 2016 04:55pm EDT", href: "/article/usa-election-trump-highlights-update-1-g-idUSL1N1CV211", ' \
  #    'blurb: "... with Republican presidential nominee\nDonald<\/b> Trump<\/b>.\n\nSYRIA\n\"(Hillary Clinton) has..' \
   #   '.(Adds quotes from Trump<\/b>)\nBy Steve Holland\nOct 25 (Reuters) - Below...", mainPicUrl: "" '

#mat = extract_links(txt)


#headline = "UPDATE 1-Highlights from Reuters\' exclusive interview with Apple<\/b>"
#print(headline.find(headline_name))


"""
import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
para = 'Amazon.com Inc (AMZN.O) has filed for a patent to use airships to store products and serve as a base for delivery-drones.' \
       '\nThe patent application was filed two years ago but was spotted only on Wednesday by Zoe Leavitt, an analyst at technology data and research firm CB Insights.' \
       '\nAccording to the patent filing, drones launched from the so-called \"airborne fulfillment centers\" (AFCs) would use far less power than those launched from the ground.' \
       '\nThe AFCs would hover at about 45,000 feet (13,700 meters) and be restocked and resupplied by \"shuttles or smaller airships.\" bit.ly/2ihP1AU' \
       '\nAmazon, which was not immediately available for comment, has laid out plans to start using drones for deliveries next year.' \
       '\n\n (Reporting by Laharee Chatterjee in Bengaluru; Editing by Ted Kerr)\n'
raw_sentences = tokenizer.tokenize(para)
print(raw_sentences)
"""


# def clean_str(string, TREC=False):
#     """
#     Tokenization/string cleaning for all datasets except for SST.
#     Every dataset is lower cased except for TREC
#     """
#     string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
#     string = re.sub(r"\'s", " \'s", string)
#     string = re.sub(r"\'ve", " \'ve", string)
#     string = re.sub(r"n\'t", " n\'t", string)
#     string = re.sub(r"\'re", " \'re", string)
#     string = re.sub(r"\'d", " \'d", string)
#     string = re.sub(r"\'ll", " \'ll", string)
#     string = re.sub(r",", " , ", string)
#     string = re.sub(r"!", " ! ", string)
#     # string = re.sub(r"\(", " \( ", string)
#     string = re.sub(r"\(", " ", string)
#     # string = re.sub(r"\)", " \) ", string)
#     string = re.sub(r"\)", " ", string)
#     string = re.sub(r"\?", " \? ", string)
#     string = re.sub(r"\s{2,}", " ", string)
#     string = re.sub(r"\s{2,}", " ", string)
#     # string = re.sub(r"\d", "NUMNUMNUM", string)
#     string = re.sub(r"\d", " ", string)
#     return string.strip() if TREC else string.strip().lower()
#
#
# directory1 = 'E:\\A1113\\FYP\\NEWS\\Labeled Corpus\\All\\'
# directory2 = 'E:\\A1113\\FYP\\NEWS\\Unlabeled Corpus\\'
# corpus = []
#
# for file in os.listdir(directory1):
#     print "Loading File: " + file
#     entity = file.split(".")[0]
#
#     if file.find('.json') > 0:
#         with open(directory1 + file) as json_data:
#             data = json.load(json_data)
#             li = data[entity]
#
#             for item in li:
#                 corpus.append(clean_str(item['content']))
#
#
# print "All files from directory 1 have been read and loaded."
#
# for file in os.listdir(directory2):
#     print "Loading File: " + file
#     entity = file.split("-")[1]
#
#     if file.find('.json') > 0:
#         with open(directory2 + file) as json_data:
#             data = json.load(json_data)
#             li = data[entity]
#
#             for item in li:
#                 corpus.append(clean_str(item['content']))
#
# print len(corpus)
# print "All files from directory 2 have been read and loaded."
#
# vectorizer = CountVectorizer(min_df=1, strip_accents='ascii')
# sparse_matrix = vectorizer.fit_transform(corpus)
# vocab = vectorizer.get_feature_names()
# print 'Vocabulary Loaded. Vocabulary Size:'
# print len(vocab)
#
# w2v = load_bin_vec('E:/A1113/FYP/wv_google.bin', vocab)
# print "Word Embeddings Loaded."
# add_unknown_words(w2v, vocab, k=300)
# W, word_idx_map = get_W(w2v, k=300)
# avg_w2v = average_embedding_w2v(w2v, vocab)
# print type(w2v)
# print len(w2v)
#
# output = open("E:\\A1113\\FYP\\BloombergNews\\BloombergNews\\w2v.pkl", 'wb')
# cPickle.dump(w2v,output)
# output.close()
#
#
# pkl_file = open("E:\\A1113\\FYP\\BloombergNews\\BloombergNews\w2v.pkl", 'rb')
# w2v = cPickle.load(pkl_file)
# pkl_file.close()
# print type(w2v)
# print len(w2v)
#





