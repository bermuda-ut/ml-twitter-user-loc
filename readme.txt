Project 2
Geolocation Classification of Tweets with Machine Learning
719577 Max Lee
_________________________

Requirements
scikit-learn 18
numpy
nltk
pickle

Required tweets in ./data/
_________________________

Instructions

STEP 1. Building the vector for the Tweets

$ python3 main.py train build 0
This will use ./data/train-tweets.txt to build vector with SetID 0
(Note SetID below)

$ python3 main.py dev build 0
This will use ./data/dev-tweets.txt to build vector with SetID 0

Vectors are stored in ./build/



STEP 2. Train the classifier
$ python3 main.py train learn 0 2
This will train classifierID 2 with training vector with setID 0.

Trained classifier is stored in ./build/



STEP 3. Development stage - scoring
$ python3 main.py dev score 0 2
This will score classifierID 2 with development vector with setID 0.

________________________

Set in Report | SetID
    A         |  -1
    B         |   0
    C         |  -3
    D         |  -2


ClassifierID | Name
    0        | Decision Tree
    1        | Naive Bayes
    2        | MLP Neural Network
    3        | SVM
    4        | KNN
    5        | Zero-R
    6        | One-R
    7        | Random
    8        | Random Forest

