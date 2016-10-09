import re
from nltk.corpus import stopwords

STOPS = stopwords.words('english')

def filterTweet(tweet):
    t = filterNameStop(onlyAlpha(tweet))
    return t

def onlyAlpha(line):
    f = ""
    for l in line:
        if(l.isalpha() or l == " "):
            f += l
    return f

def toLowers(line):
    f = ""
    for l in line:
        f += l.lower()
    return f

# filter out stop words
def filterNameStop(line):
    rline = "";
    for word in line.split():
        if word not in STOPS and '@' != word[0]:
            rline += word + " "

    return rline[:-1].strip('\n').strip();

def filterURL(line):
    rline = re.sub('https?:\/\/.*[\r\n]*', '', line)
    return rline
