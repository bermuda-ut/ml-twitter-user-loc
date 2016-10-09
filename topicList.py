import sys
from nltk.corpus import wordnet as wn

topic = sys.argv[1]

t = wn.synsets(topic)
l = []
n = 50
print(t)

for f in t:
    q = [w.replace('_', ' ') for s in f.closure(lambda s:s.hyponyms()) for w in s.lemma_names()]
    for wd in q:
        if(" " not in wd):
            l += [wd]
        if(len(l) >= n):
            break

print(l)

