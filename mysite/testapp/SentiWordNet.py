import csv
import cPickle

with open('E:\\A1113\\FYP\\SentiWordNet_3.0.0_20130122.txt', 'rb') as csvfile:
    lineNumber = 0
    reader = csv.reader(csvfile, dialect='excel')
    tempDict = {}
    dictionary = {}
    for row in reader:
        lineNumber += 1
        line = row[0]
        # if it's a comment, skip this line
        if not line.startswith("#"):
            data = line.split("\t")
            wordTypeMarker = data[0]
            print line
            if len(data) != 6:
                continue
            print len(data)
            # Calculate synset score as score = PosS - Neg.
            synsetScore = float(data[2]) - float(data[3])
            print synsetScore
            synTermsSplit = data[4].split(" ")

            # Go through all terms of current synset.
            for synTermSplit in synTermsSplit:
                synTermAndRank = synTermSplit.split("#")
                synTerm = synTermAndRank[0] + "#" + wordTypeMarker
                synTermRank = int(synTermAndRank[1])

                if synTerm not in tempDict:
                    tempDict[synTerm] = {}

                tempDict[synTerm][synTermRank] = synsetScore
                print tempDict[synTerm]


    print tempDict['scintillate#v']
    for word, scoreMap in tempDict.iteritems():
        # Calculate weighted average. Weigh the synsets according to their rank
        # Score= 1/2*first + 1/3*second + 1/4*third ..... etc.
        # Sum = 1/1 + 1/2 + 1/3 ...
        score = 0.0
        sum = 0.0
        for rank, scr in scoreMap.iteritems():
            score += scr/float(rank)
            sum += 1.0/float(rank)
        score = score /sum

        dictionary[word] = score

    print len(dictionary)
    print type(dictionary)

    output = open("E:\\A1113\\FYP\\BloombergNews\\BloombergNews\\sentiWordNetDict.pkl", 'wb')
    cPickle.dump(dictionary, output)
    output.close()

    pkl_file = open("E:\\A1113\\FYP\\BloombergNews\\BloombergNews\\sentiWordNetDict.pkl", 'rb')
    dictionary = cPickle.load(pkl_file)
    pkl_file.close()
    print dictionary['able#a']
