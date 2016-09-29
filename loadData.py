import sys
import numpy as np
from collections import defaultdict
from filters import filterTweet
from features import buildVector
from sklearn import tree
from sklearn.cross_validation import cross_val_score
#from sklearn import datasets as ds
#from sklearn import svm

STEP = 5000
DATA_DIR = "./data/"
FEATURE_LIST = []
TYPE = sys.argv[1]

fileDir = DATA_DIR + TYPE + "-tweets.txt"

print("Opening " + fileDir)
fp = open(fileDir, 'r')

userTweets = defaultdict(list) # userId -> list of tweetID
tweetContent = dict() # tweetId -> tweet content
tweetVector = dict() # tweetId -> tweet vector
tweetLoc = dict() # tweetid -> tweet Location

print("Loading tweets")
for line in fp:
    line = line.strip('\n')
    sep = line.split('\t')
    filtered = filterTweet(sep[2])

    if(len(filtered) == 0):
        continue

    """
    for uid, tids in userTweets.items():
        for tid in tids:
            if(tweetContent[tid] == filtered):
                continue
    """
    userTweets[sep[0]].append(sep[1])
    tweetContent[sep[1]] = filtered
    tweetLoc[sep[1]] = sep[3]
    tweetVector[sep[1]] = buildVector(tweetContent[sep[1]])
    #print(sep[1], tweetVector[sep[1]])
    #print(tweetLoc[sep[1]])
    #print(filtered)
    if(len(tweetContent) % STEP == 0):
        print("Read {} tweets".format(len(tweetContent)))

fp.close()


X = []
Y = []
for (k, v) in tweetVector.items():
    X.append(v)
    Y.append(tweetLoc[k])

print("Fitting Decision Tree")

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
scores = cross_val_score(clf, X, Y, cv=10, n_jobs=-1) # 10 folds
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
