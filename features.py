import sys
from nltk.corpus import wordnet as wn

DATA_DIR = "./data/"
TYPE = sys.argv[1]

BWD = ['boston', 'redsox', 'ma']
HWD = ['houston', 'jupdicom', 'tx']
SEWD = ['seattle', 'wa', 'cheezburger', 'bellevue']
SDWD = ['diego', 'san', 'chargers', 'sd', 'sdut', 'sandiego']
WWD = ['dc', 'obama', 'health', 'bill']

MAGIC_WORDS = [
    BWD,
    HWD,
    SEWD,
    SDWD,
    WWD
    ]

MAGIC_TOPICS = [
    'travel',
    'sports',
    'celebrity',
    'movie',
    'food',
    'communication',
    'health',
    'location',
    'technology',
    'computer',
    'phone'
    ]
N = 20


def buildVector(line):
    v = []
    #v += best35(line)
    v += magicCount(line)
    v += classicCount(line, TOPIC_VEC)
    v += [wordCount(line)]
    return v

# features

def getTopicWords(topic, n):
    t = wn.synsets(topic)
    l = []

    for f in t:
        q = [w.replace('_', ' ') for s in f.closure(lambda s:s.hyponyms()) for w in s.lemma_names()]
        for wd in q:
            if(" " not in wd):
                l += [wd]
            if(len(l) >= n):
                return l

    return l

def buildTopicVector():
    final = []
    for topic in MAGIC_TOPICS:
        final.append(getTopicWords(topic, N))
    return final

TOPIC_VEC = buildTopicVector()

def wordCount(line):
    return len(line.split())

def classicCount(line, feat):
    v = [0] * len(feat)
    for wd in line.split():
        for i in range(0, len(feat)):
            for fwd in feat[i]:
                if wd in fwd or fwd in wd:
                    v[i] += 1
    return v

def magicCount(line):
    v = [0] * len(MAGIC_WORDS)
    for wd in line.split():
        for i in range(0, len(MAGIC_WORDS)):
            for mwd in MAGIC_WORDS[i]:
                if mwd == wd:
                    v[i] += 1
    return v

# classes: B H SD Se W
def best35(line):
    fvec = FVEC35
    veclen = len(fvec)
    final = [0] * veclen

    for word in line.split():
        for i in range(0, veclen):
            if word == fvec[i]:
                final[i] += 1

    return final

# read only feature and return
def arffRead(fileLoc):
    fp = open(fileLoc)
    l = []
    for line in fp:
        sp = line.split()
        if sp[0] == '@ATTRIBUTE' and 'http' not in sp[1]:
            l.append(sp[1])
        elif sp[0] == '@DATA':
            break
    return l

FVEC35 = arffRead(DATA_DIR + 'best35/'+ TYPE + '-best35.arff')
