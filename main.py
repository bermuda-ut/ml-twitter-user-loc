'''
# =============================================================================
#      FileName: main.py
#        Author: Max Lee
#         Email: hoso1312@gmail.com
#      HomePage: imnotbermuda.com
#       Version: 0.0.1
#    LastChange: 2016-10-07 07:11:35
# =============================================================================
# ################################
#             WARNING
# Badly coded in hackathon style!
# ################################
'''

import sys
import pickle
from helper import *
from features import *

STEP = 10000
DATA_DIR = "./data/"
BUILD_DIR = "./build/"
FEATURE_LIST = []
TYPE = sys.argv[1]
FLAGS = sys.argv[2:]


def main():
    if('build' in FLAGS):
        if(len(FLAGS) < 2):
            print("python3 main.py [train/dev/test] build [-1/0/1/2 vectorType] (build loc to tweet)")
            return

        from sklearn.preprocessing import RobustScaler
        from collections import defaultdict
        from filters import filterTweet
        from sklearn.feature_extraction.text import TfidfTransformer as tt

        fileDir = DATA_DIR + TYPE + "-tweets.txt"
        print("Building from", fileDir)
        fp = open(fileDir, 'r')

        userTweets = defaultdict(list) # userId -> list of tweetID
        tweetContent = dict() # tweetId -> tweet content
        tweetVector = dict() # tweetId -> tweet vector
        tweetLoc = dict() # tweetid -> tweet Location
        locationToTweets = defaultdict(list)

        print("--")

        setId = int(FLAGS[1])
        if(setId == -3):
            print("sim")
        if(setId == -2):
            print("finalized features 3")
        if(setId == -1):
            print("count x 2")
        if(setId >= 0):
            print("structure score")
        if(setId >= 1):
            print("user defined")
        if(setId >= 2 and setId != 3):
            print("bunch of stuff")

        print("--")

        X = []
        Y = []

        for line in fp:
            line = line.strip('\n')
            sep = line.split('\t')
            filtered = filterTweet(sep[2])

            if(len(filtered) == 0):
                continue

            userTweets[sep[0]].append(sep[1])
            tweetContent[sep[1]] = filtered
            tweetLoc[sep[1]] = sep[3]
            tweetVector[sep[1]] = buildVector(tweetContent[sep[1]], setId)
            X.append(tweetVector[sep[1]])
            Y.append(tweetLoc[sep[1]])

            if(len(FLAGS) == 3):
                locationToTweets[sep[3]].append(filtered)
            if(len(tweetContent) % STEP == 0):
                print("Read {} tweets".format(len(tweetContent)))
                #break

        fp.close()

        data = [userTweets, tweetContent, tweetVector, tweetLoc ]

        fname = BUILD_DIR + TYPE + "-obj-" + str(setId)
        print("Saving tweets to", fname)
        f = open(fname, "wb")
        s = pickle.dump(data, f)
        f.close()

        if(TYPE == "train"):
            if(setId == -3):
                rs = tt()
            else:
                rs = RobustScaler()
            rs.fit(X, y=Y)
            fname = BUILD_DIR + TYPE + "-scaler-" + str(setId)
            print("Saving scaler to", fname)
            f = open(fname, "wb")
            s = pickle.dump(rs, f)
            f.close()

        if(len(FLAGS) == 3):
            fname = BUILD_DIR + TYPE + "-locToTweet"
            print("Saving separated tweets to", fname)
            f = open(fname, "wb")
            s = pickle.dump(locationToTweets, f)
            f.close()

        print("Done")

    if('learn' in FLAGS):
        if(len(FLAGS) < 3):
            print("python3 main.py [train/develope/test] learn [-1/0/1/2 vectorType] [0/1/2/3 MLType] [# for kNN]")
            return

        mlread = int(FLAGS[1])
        mltype = int(FLAGS[2])

        from sklearn import tree, svm
        from sklearn.neural_network import MLPClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.dummy import DummyClassifier
        from sklearn.preprocessing import RobustScaler
        from sklearn.ensemble import RandomForestClassifier
        import numpy

        fname = BUILD_DIR + TYPE + "-obj-" + str(mlread)
        print("Learning from", fname)
        f = open(fname, "rb")
        data = pickle.load(f)
        userTweets, tweetContent, tweetVector, tweetLoc = data

        X = []
        Y = []
        for (k, v) in tweetVector.items():
            X.append(v)
            Y.append(tweetLoc[k])


        """

        fname = BUILD_DIR + TYPE + "-scaler-" + str(mlread)
        print("Using scaler from", fname)
        f = open(fname, "rb")
        rs = pickle.load(f)
        f.close()

        X = rs.transform(X)
        """

        #--

        if(mltype == 4 and len(FLAGS) >= 4):
            nnearest = int(FLAGS[3])
        else:
            nnearest = 5

        name = mlTypeToName(mltype, nnearest)

        # 0 1 5 6 7

        if(mltype == 0):
            clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=10)

        elif(mltype == 1):
            clf = GaussianNB()

        elif(mltype == 2):
            clf = MLPClassifier(activation="logistic")

        elif(mltype == 3):
            clf = svm.NuSVC(kernel='rbf', max_iter=50000)

        elif(mltype == 4):
            clf = KNeighborsClassifier(n_neighbors=nnearest)

        elif(mltype == 5):
            clf = DummyClassifier(strategy="most_frequent")

        elif(mltype == 6):
            clf = tree.DecisionTreeClassifier(max_depth=1)

        elif(mltype == 7):
            clf = DummyClassifier(strategy="uniform")

        elif(mltype == 8):
            clf = RandomForestClassifier(n_estimators=50, max_depth=5, max_leaf_nodes=20)

        else:
            print("Invalid classifier id")
            return

        print("Fitting", name)
        clf = clf.fit(X, y=Y)

        print("Score:", clf.score(X,Y))

        fname = BUILD_DIR + "clf-" + str(mlread) + "-" + name
        print("Saving classifier as", fname)
        f = open(fname, "wb")
        s = pickle.dump(clf, f)
        f.close()
        print("Done")

    elif("score" in FLAGS):
        if(len(FLAGS) < 3):
            print("python3 main.py [train/dev/test] score [-1/0/1/2 vectorType] [0/1/2/3 MLType] [# for kNN]")
            return

        from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score
        from sklearn.preprocessing import RobustScaler

        mlread = int(FLAGS[1])
        mltype = int(FLAGS[2])

        if(mltype == 4 and len(FLAGS) >= 4):
            nnearest = int(FLAGS[3])
        else:
            nnearest = 5

        name = mlTypeToName(mltype, nnearest)

        fname = BUILD_DIR + TYPE + "-obj-" + str(mlread)
        print("Opening and loading tweet vector", fname)
        f = open(fname, "rb")
        data = pickle.load(f)
        userTweets, tweetContent, tweetVector, tweetLoc = data
        f.close()

        X = []
        Y = []
        for (k, v) in tweetVector.items():
            X.append(v)
            Y.append(tweetLoc[k])

        
        """
        fname = BUILD_DIR + "train-scaler-" + str(mlread)
        print("Opening scaler", fname)
        f = open(fname, "rb")
        rs = pickle.load(f)
        f.close()

        X = rs.transform(X)
        """

        fname = BUILD_DIR + "clf-" + str(mlread) + "-" + name
        print("Opening classifier", fname)
        d = open(fname, "rb")
        clf = pickle.load(d)
        d.close()

        y_pred = clf.predict(X)
        print("---")
        print("Accuracy:", accuracy_score(Y, y_pred))
        print("")
        print("Precision ", end="")
        print("weighted:\t", precision_score(Y, y_pred, average='weighted'))
        print("\t  macro:\t", precision_score(Y, y_pred, average='macro'))
        print("\t  micro:\t", precision_score(Y, y_pred, average='micro'))
        print("")
        print("Recall ", end="")
        print("weighted:\t", recall_score(Y, y_pred, average='weighted'))
        print("\tmacro:  \t", recall_score(Y, y_pred, average='macro'))
        print("\tmicro:  \t", recall_score(Y, y_pred, average='micro'))
        print("")
        print("FBeta(0.5) ", end="")
        print("weighted:\t", fbeta_score(Y, y_pred, average='weighted', beta=0.5))
        print("\t   macro:\t", fbeta_score(Y, y_pred, average='macro', beta=0.5))
        print("\t   micro:\t", fbeta_score(Y, y_pred, average='micro', beta=0.5))

    elif("feng" in FLAGS):
        if(len(FLAGS) < 3):
            print("python3 main.py [train/develope/test] feng [-1/0/1/2 vectorType] [validationMethod]")
            return;

        from sklearn import tree
        from sklearn.feature_selection import mutual_info_classif as mi, chi2

        mlread = int(FLAGS[1])
        fname = BUILD_DIR + TYPE + "-obj-" + str(mlread)
        print("Learning from", fname)
        f = open(fname, "rb")
        data = pickle.load(f)
        userTweets, tweetContent, tweetVector, tweetLoc = data

        X = []
        Y = []
        for (k, v) in tweetVector.items():
            X.append(v)
            Y.append(tweetLoc[k])

        fname = BUILD_DIR + "train-scaler-" + str(mlread)
        print("Opening scaler", fname)
        f = open(fname, "rb")
        rs = pickle.load(f)
        f.close()
        X = rs.transform(X)

        if(FLAGS[-1] == '0'):
            clf = tree.DecisionTreeClassifier()
            clf.fit(X, Y)
            scores = clf.feature_importances_
            topics = getFeatures(mlread)
            num = len(scores)

            for i in range(len(scores)):
                print("{:10}\t|\t{:.3f}".format(topics[i], scores[i] * num))
        elif(FLAGS[-1] == '1'):
            print(mi(X,y=Y))
        elif(FLAGS[-1] == '2'):
            print(chi2(X,y=Y))

    elif("mine" in FLAGS):
        from collections import defaultdict
        fileDir = DATA_DIR + TYPE + "-tweets.txt"
        print("Mining from", fileDir)
        fp = open(fileDir, 'r')

        dd = defaultdict(set)

        for line in fp:
            line = line.strip('\n')
            sep = line.split('\t')

            words = sep[2].split()
            for w in words:
                if(w[0] != "@" and 'www' not in w and 'http' not in w):
                    z = ""
                    for l in w:
                        if l.isalpha():
                            z += l
                    dd[sep[3]].add(z.lower())

        ck= FLAGS[1]
        s = dd[ck]
        for k, v in dd.items():
            if(ck != k):
                s = s - dd[k]

        fname = BUILD_DIR + ck + "-words"
        print("Saving words as", fname)
        f = open(fname, "wb")
        s = pickle.dump(s, f)
        f.close()
        print("Done")




main();
print("--------------------")
