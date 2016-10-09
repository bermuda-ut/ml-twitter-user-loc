import sys
import re
import pickle
import nltk
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet as wn

DATA_DIR = "./data/"
BUILD_DIR = "./build/"
TYPE = sys.argv[1]

#BWD = ['boston', 'redsox', 'ma']
BWD = ['boston', 'celtics', 'bruins', 'fenway', 'berklee', 'sox', 'roxbury', 'mbta', 'brookline', 'tanglewood', 'tremont', 'charlestown', 'massachusetts', 'fitchburg', 'pawtucket', 'braves', 'fiedler', 'dorchester', 'umass', 'alcs', 'framingham', 'athenaeum', 'globe', 'bulger', 'hingham', 'revere', 'copley', 'provincetown', 'mifflin', 'kenmore', 'pops', 'terriers', 'casco', 'ozawa', 'blackie', 'tufts', 'braintree', 'marathons', 'suisse', 'dropkick', 'stockings', 'longwood', 'brandeis', 'nashua', 'curley', 'lowell', 'haverhill', 'seiji', 'merrimack', 'shakedown', 'mchale', 'providence', 'witham', 'bowdoin', 'doubleheader', '76ers', 'somerville', 'outfielder', 'breakers', 'canadiens', 'lakers', 'schilling', 'sportswriter', 'harvard', 'waltham', 'auerbach', 'hynes', 'ma', 'winthrop', 'minutemen', 'beacon', 'doves', 'yankees', 'pennant', 'houghton', 'woburn', 'fens', 'patriots', '1775', 'ticonderoga', 'baseman', 'quincy', 'watertown', 'farber', 'shamrocks', 'shortstop', 'medford', 'herald', 'prudential', 'lincolnshire', 'scholz', 'refresher', 'malden', 'infielder', 'conservatory', 'blazers', 'longfellow', 'worcester', 'transcript', 'barnstable', 'midseason']

#HWD = ['houston', 'jupdicom', 'tx']
HWD = ['houston', 'astros', 'texans', 'oilers', 'galveston', 'tx', 'cougars', 'nutt', 'rockets', 'whitney', 'isd', 'reliant', 'remi', 'jacinto', 'dynamo', '610', 'tollway', 'comets', 'bayou', 'hillsong', 'texas', 'superdraft', 'beltway', 'piney', 'baylor', 'buffaloes', 'marques', 'lanier', 'enron', 'thelma', 'chronicle', 'hensley', 'brazos', 'zoned', 'wnba', 'moores', 'aretha', 'mariah', 'knicks', 'mavericks', 'lamar', 'texan', 'screwed', 'gamblers', 'renfrewshire', 'bagwell', 'razorbacks', 'babyface', 'huntsville', 'tcu', 'evacuees', 'colt', 'uptown', 'uh', 'mls', 'dallas', 'kilt', 'toros', 'blige', 'metroplex', 'sam', 'grizzlies', 'thurgood', 'burnet', 'smu', 'intercontinental', 'wha', 'hobby', 'arista', 'mcnair', 'celine', 'katy', 'waco', 'timberwolves', 'storefront', 'oilfield', 'elvin', 'ucf', 'christi', 'rusk', 'jaguars', 'chivas', 'alamo', 'tanglewood', 'supersonics', 'chaka', 'undrafted', 'texarkana', 'gordie', 'padres', 'northside', 'lomax', 'galleria', 'harte', 'bobbie', 'incarnate', 'collegiately', 'ike', 'winans', 'petrochemical', 'compaq']

#SEWD = ['seattle', 'wa', 'cheezburger', 'bellevue']
SEWD = ['seattle', 'supersonics', 'sounders', 'seahawks', 'intelligencer', 'mariners', 'qwest', 'tacoma', 'puget', 'sleepless', 'metropolitans', 'grunge', 'bremerton', 'wa', 'soundgarden', 'rainier', 'wto', 'renton', 'bellingham', 'slew', 'spokane', 'ichiro', 'whitecaps', 'nasl', 'kodiak', 'thunderbirds', 'washington', 'uw', 'cobain', 'starbucks', 'bellevue', 'yakima', 'alaskan', 'nordstrom', 'wnba', 'kenmore', 'klondike', 'ballard', 'nfc', 'bnsf', 'drydock', 'portland', 'timbers', 'vedder', 'mls', 'bainbridge', 'whl', 'nirvana', 'pcl', 'superdraft', 'everett', 'coliseum', 'payton', 'blazers', 'brewers', 'eastside', 'monorail', 'undrafted', 'westlake', 'usl', 'colman', 'alcs', 'repertory', 'selig', 'denny', 'nuggets', 'montero', 'staley', 'frye', 'fremont', 'frasier', 'kasey', 'husky', 'walla', 'seward', 'hermon', 'cantrell', 'shipyards', 'timberwolves', 'cascade', 'redmond', 'olympia', 'huskies', 'needle', 'cascades', 'waivers', '49ers', 'espresso', 'sitka', 'eniwetok', 'shipbuilding', 'juneau', 'schell', 'expos', 'starfire', 'cantwell', 'xl', 'vancouver', 'padres', 'gilman', 'grohl']

#SDWD = ['diego', 'san', 'chargers', 'sd', 'sdut', 'sandiego']
SDWD = ['diego', 'san', 'chargers', 'sd', 'sdut', 'sandiego' 'baja', 'obispo', 'northridge', 'yuba', 'chaparral', 'mendocino', 'berkeley', 'menlo', 'pasadena', 'jolla', 'inglewood', 'oxnard', 'redlands', 'modesto', 'fresno', 'simi', 'bakersfield', 'nuys', 'schwarzenegger', 'monterey', 'chula', 'fullerton', 'merced', 'mojave', 'sacramento', 'pomona', 'carlsbad', 'alameda', 'oceanside', 'burbank', 'napa', 'sonoma', 'bernardino', 'torrance', 'verdes', 'irvine', 'cession', 'altos', 'madera', 'glendale', 'vallejo', 'modoc', 'roseville', 'whittier', 'barstow', 'rancho', 'palo', 'shasta', 'alta', 'lassen', 'coachella', 'anaheim', 'klamath', 'mariposa', 'palos', 'joaquin', 'capistrano', 'mateo', 'wildfires', 'indio', 'diego', 'malibu', 'ventura', 'livermore', 'sonoran', 'chino', 'viejo', 'angeles', 'oakland', 'redwood', 'rialto', 'kern', 'cif', 'riverside', 'hanford', 'dorado', 'stockton', 'amador', 'tahoe', 'culver', 'redondo', 'californian', 'assemblyman', 'vandenberg', 'foothill', 'sonora', 'santee', 'fremont', 'nevada', 'redding', 'los', 'shakedown', 'tijuana', 'loma', 'folsom', 'monica', 'francisco', 'chico', 'salinas', 'placer']

#WWD = ['dc', 'obama', 'health', 'bill']
WWD = ['obama', 'president', 'health', 'bill', '20540', 'd.c.', 'bremerton', 'redskins', 'yakima', 'spokane', 'tacoma', 'bellingham', 'puget', 'dulles', 'dc', 'anacostia', 'hdl', 'seattle', 'pnp', 'capitals', 'walla', 'smithsonian', 'potomac', 'renton', 'huskies', 'pasco', 'mystics', 'bolling', 'corcoran', 'longview', 'colville', 'thurgood', 'georgetown', 'rainier', 'bellevue', 'repository', 'hagerstown', 'senators', 'pullman', 'nw', 'parke', 'lobbyist', 'booker', 'brookings', 'olympia', 'tuskegee', 'nationals', 'dinah', 'mcnair', 'kirkland', 'newsweek', 'hanford', 'beltway', 'uw', 'politico', 'coulee', 'wizards', 'd', 'intelligencer', 'loc', 'arlington', 'cascades', 'lobbyists', 'kenmore', 'multnomah', 'redmond', 'watergate', 'pentagon', 'rockville', 'richland', 'capitol', 'c', 'idaho', 'geiger', 'reston', 'grover', 'fredericksburg', 'clackamas', 'cougars', 'rappahannock', 'marysville', 'everett', 'bainbridge', 'grays', 'cascade', 'harpers', 'fairfax', 'bethesda', 'baltimore', 'shales', 'brazos', 'jefferson', 'murrow', 'hillsboro', 'msnbc', 'oregon', 'monongahela', 'caboose', 'abramoff', 'wrc', 'yorktown', 'prints', 'husky', 'correspondent']

CUSTOM = ['blizzcon', 'blizzard', 'con', 'blizz', 'warcraft', 'diablo', 'anaheim', 'mirror', 'edge', 'left', '4', 'dead', 'valve', 'dead', 'space', 'games', 'game', 'video', 'gaming']

CUSTOM2 = ['yogscast', 'england', 'britain', 'uk', 'scotland', 'europe', 'eu']

CUSTOM3 = ['microsoft', 'ny', 'programming', 'google', 'apple', 'iphone', 'ipad', 'android']

if(sys.argv[-2] == '-3'):
    f = open(BUILD_DIR + "train-locToTweet", 'rb')
    SIM_DICT = pickle.load(f)
    f.close()
    SIM_KEYS = list(SIM_DICT.keys())
    SIM_TFIDF = dict()

    for key in SIM_KEYS:
        SIM_TFIDF[key] = TfidfVectorizer().fit(SIM_DICT[key])

MAGIC_WORDS = [
    set(BWD),
    set(HWD),
    set(SEWD),
    set(SDWD),
    set(WWD),
    CUSTOM,
    CUSTOM2,
    CUSTOM3
    ]

SIMPLE = BWD + HWD + SEWD + SDWD + WWD

MAGIC_TOPICS = [
    'movie',
    'location',
    'country',
    'sports',
    'technology',

    #'travel',
    #'computer',
    #'food',
    #'game',
    #'celebrity',
    #'communication',
    #'computer',
    #'phone',
    'music',
    'election',
    #'nature',
    #'shop',
    #'park',
    #'restaurant',
    #'show',
    #'fashion',
    #'accent',
    #'team',
    #'news',
    #'books',
    #'software',
    ]

N = 5

def getFeatures(setId):
    v = []

    if(setId == -2):
        v += ['custom']

    if(setId == -1):
        v += MAGIC_TOPICS

    if(setId >= 0):
        v += ['BWD', 'HWD', 'SWED', 'SDWD', 'WWD', 'CUSTOM', 'CUSTOM2', 'CUSTOM3']
    if(setId >= 1):
        v += MAGIC_TOPICS
    if(setId >= 2 and setId != 3):
        v += ['TwitterLength']

    return v

def buildVector(line, setId):
    v = []

    if(setId == -3):
        #v += simpleCounter(line.lower(), SIMPLE)
        v += calcSim(line)

    if(setId == -2):
        v += structureScore(line)
        v += classicCount(line.lower(), TOPIC_VEC)
        v += magicCount(line.lower())
        #v += simpleCounter(line.lower(), SIMPLE)
        #v += calcSim(line)

    if(setId == -1):
        v += classicCount(line.lower(), TOPIC_VEC)
        v += magicCount(line.lower())
        #v += classicCount(line.lower(), LOADED_WORDS)
        #v += classicCount(line.lower(), TOPIC_VEC)

    if(setId >= 0):
        v += structureScore(line)
        #v += magicCount(line.lower())

    if(setId >= 1):
        v += classicCount(line.lower(), TOPIC_VEC)

    if(setId >= 2 and setId != 3):
        v += [wordCount(line.lower())]
        v += [hasURL(line)]
        v += [countCapitals(line)]
        v += [countSpecial(line)]

    return v

# features


def calcSim(line):
    v = [0] * len(SIM_KEYS)

    for i in range(len(SIM_KEYS)):
        k = SIM_KEYS[i]
        tfidf = SIM_TFIDF[k]
        m = tfidf.transform([line])
        total=0
        c = 0
        for x in (m.A[0]):
            if(x != 0):
                total += x
                c += 1
        if(c != 0):
            score = total/c
        else:
            score = 0

        v[i] = total

    return v

def simpleCounter(line, ref):
    v = [0] * len(ref)

    for wd in line.split():
        for i in range(0, len(ref)):
            if(ref[i] == wd):
                v[i] += 1

    return v

def notPrime(n):
    if n == 1 or n == 2:
        return False
    else:
        for i in range(2, int(math.sqrt(n)) + 1):
            if(n % i == 0):
                return True
    return False

def nextPrime(n):
    n+=1
    while(notPrime(n)):
        n+=1
    return n

def structureScore(line):
    text = nltk.word_tokenize(line);
    start = 2
    sc = [0] * 9
    loc = 1

    for word, cat in nltk.pos_tag(text):
        n = nextPrime(start)
        if('VB' == cat):
            sc[0] += loc*n
        elif('NNS' == cat):
            sc[1] += loc*n
        elif('JJ' == cat):
            sc[2] += loc*n
        elif('NNP' == cat):
            sc[3] += loc*n
        elif('NN' == cat):
            sc[4] += loc*n
        elif('VBG' == cat):
            sc[5] += loc*n
        elif('VBP' == cat):
            sc[6] += loc*n
        elif('PRP' == cat):
            sc[7] += loc*n
        else:
            sc[8] += loc*n

        loc += 1
        start = n

    return sc

def hasURL(line):
    rline = re.sub('https?:\/\/.*[\r\n]*', '', line)
    if(rline == line):
        return 0
    return 1

def countSpecial(line):
    a = 0
    for c in line:
        if(not c.isalpha()):
            a += 1
    return a

def countCapitals(line):
    a = 0
    for c in line:
        if(c.isupper()):
            a += 1
    return a

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
                if wd == fwd:
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


def expandWords():
    t = 30
    for i in range(0, len(MAGIC_WORDS)):
        c = 0
        for wd in MAGIC_WORDS[i]:
            MAGIC_WORDS[i] += getTopicWords(wd, 3)
            c += 1
            if(c > t):
                break


def loadWords():
    t = ['B', 'H', 'SD', 'SE', 'W']
    v = []
    for l in t:
        f = open(BUILD_DIR + l + "-words", "rb")
        v.append(pickle.load(f))
        f.close()
    return v

#LOADED_WORDS = loadWords()
#expandWords()
#FVEC35 = arffRead(DATA_DIR + 'best35/'+ TYPE + '-best35.arff')
