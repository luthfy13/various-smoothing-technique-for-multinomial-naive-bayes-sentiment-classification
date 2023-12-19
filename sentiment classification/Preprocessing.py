import re
import string
import pandas
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk import word_tokenize
from nltk import pos_tag
import logging
import flair
from flair import torch
from flair.data import Sentence
from flair.models import SequenceTagger

logging.getLogger("flair").setLevel(logging.CRITICAL)
logging.getLogger("Sentence").setLevel(logging.CRITICAL)
logging.getLogger("torch").setLevel(logging.CRITICAL)
logging.getLogger("SequenceTagger").setLevel(logging.CRITICAL)

pandas.options.mode.chained_assignment = None  # default='warn'
tagger = SequenceTagger.load("flair/pos-english")
flair.device = torch.device("cuda:0")

def getStopWords():
    df = pandas.read_csv('../data/stoplist-en.csv')
    sw = df.word.to_list()
    new_sw = []
    for w in sw:
        tags = []
        try:
            for x in wn.synsets(w):
                tags.append(x.pos())
        except:
            tags = ['']
        if not 'a' in tags:
            new_sw.append(w)
    return new_sw

def getNegasi():
	df = pandas.read_csv('../data/negasi-en.csv')
	neg = df.word.to_list()
	return neg

def hapusStopword(komentar, stopWords):
    words = komentar.split()
    count = len(words)
    for i in range(0, count):
        if "NOT_" in words[i]:
            word = re.sub("NOT_", "", words[i])
            if word in stopWords:
                words[i] = word
                j = i+1
                if j<count:
                    words[j] = f"NOT_{words[j]}"
            
    result = [word for word in words if word.lower() not in stopWords]
    text = ' '.join(result)
    return text.strip()

def tagNegasi(komentar, negasi):
	hsl = " " + komentar + " "
	for word in negasi:
		hsl = re.sub(" " + word + " ", " NOT_", hsl)
	hsl = hsl[1:-1]
	return hsl

def lemmatization(komentar, lemmatizer):
    words = komentar.split()
    result = []
    for word in words:
        result.append(lemmatizer.lemmatize(word))
    return ' '.join(result)

def rm_html(input_text):
    result = re.sub(r"<(?:\"[^\"]*\"['\"]*|'[^']*'['\"]*|[^'\">])+>", '', input_text)
    return ' '.join(result.split())

def preprocess_tweet(tweet):
    
    # Reading contractions.csv and storing it as a dict.
    contractions = pandas.read_csv('../data/contractions.csv', index_col='Contraction')
    contractions.index = contractions.index.str.lower()
    contractions.Meaning = contractions.Meaning.str.lower()
    contractions_dict = contractions.to_dict()['Meaning']

    # Defining regex patterns.
    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|(www\.)[^ ]*)"
    userPattern       = '@[^\s]+'
    hashtagPattern    = '#[^\s]+'
    alphaPattern      = "[^a-z0-9<>]"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"

    # Defining regex for emojis
    smileemoji        = r"[8:=;]['`\-]?[)d]+"
    sademoji          = r"[8:=;]['`\-]?\(+"
    neutralemoji      = r"[8:=;]['`\-]?[\/|l*]"
    lolemoji          = r"[8:=;]['`\-]?p+"

    tweet = tweet.lower()

    # Replace all URls with '<url>'
    tweet = re.sub(urlPattern,'<url>',tweet)
    # Replace @USERNAME to '<user>'.
    tweet = re.sub(userPattern,'<user>', tweet)

    # Replace 3 or more consecutive letters by 2 letter.
    tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

    # Replace all emojis.
    tweet = re.sub(r'<3', '<heart>', tweet)
    tweet = re.sub(smileemoji, '<smile>', tweet)
    tweet = re.sub(sademoji, '<sadface>', tweet)
    tweet = re.sub(neutralemoji, '<neutralface>', tweet)
    tweet = re.sub(lolemoji, '<lolface>', tweet)

    for contraction, replacement in contractions_dict.items():
        tweet = tweet.replace(contraction, replacement)

    # Remove non-alphanumeric and symbols
    # tweet = re.sub(alphaPattern, ' ', tweet)

    # Adding space on either side of '/' to seperate words (After replacing URLS).
    tweet = re.sub(r'/', ' / ', tweet)
    return tweet

def remove_rb_tag(sentence):
    words = []
    for word in sentence.split():
        w = f" {word} "
        w = re.sub("_RB ", " ", w)
        words.append(w.strip())
    return ' '.join(words)

def pos_tag_nltk(kalimat):
    adj_tags  = ["JJ", "JJR", "JJS"]
    noun_tags = ["NN", "NNS", "NNP", "NNPS", "PRP", "PRP$"]
    adv_tags  = ["RB", "RBR", "RBS"]
    verb_tags = ["VB", "VBG", "VBD", "VBN", "VBP", "VBZ", "MD"]
    
    tokens = word_tokenize(kalimat)
    tag = pos_tag(tokens)
    tagged_words = []
    for n in tag:
        # if (n[1] == 'RB' or n[1] == 'DT') and n[0] not in ['no', 'not']:
        if (n[1] == 'RB') and n[0] not in ['no', 'not']:
            # kata = n[0] + '_' + n[1]
            kata = n[0] + '_RB'
        else:
            kata= n[0]
        tagged_words.append(kata)
    return ' '.join(tagged_words)

def pos_tag_flair(text):
    sentence = Sentence(text)
    tagger.predict(sentence)
    tagged_words = []
    for word in sentence:
        # if (word.tag == 'RB' or word.tag == 'DT') and word.text not in ['no', 'not']:
        if (word.tag == 'RB') and word.text not in ['no', 'not']:
            # kata=word.text + '_' + word.tag
            kata=word.text + '_RB' + word.tag
        else:
            kata=word.text
        tagged_words.append(kata)
    return ' '.join(tagged_words)

def get_tag(text):
    sentence = Sentence(text)
    tagger.predict(sentence)
    tagged_words = []
    for word in sentence:
        tagged_words.append(word.text + "_" + word.tag)
    return ' '.join(tagged_words)

def pos_tag_all(text):
    # sentence = Sentence(text)
    # tagger.predict(sentence)
    # tagged_words = []
    # for word in sentence:
    #     kata=word.text + '_' + word.tag
    #     tagged_words.append(kata)
    # return ' '.join(tagged_words)
    tokens = word_tokenize(text)
    tag = pos_tag(tokens)
    tagged_words = []
    for n in tag:
        kata = n[0] + '_' + n[1]
        tagged_words.append(kata)
    return ' '.join(tagged_words)

# without POS tag
def negation_tag_1(comment):
    
    negations = pandas.read_csv('../data/negasi-en.csv').word.to_list()
    
    #check if negation exist
    is_exist_neg = False
    for negation in negations:
        if negation in comment:
            is_exist_neg = True
            break
    if is_exist_neg==False: return comment


    # hapus redundansi not
    result = " " + comment + " "
    for word in result.split():
        if word in negations:
            result = re.sub(" " + word + " ", " no ", result)

    words = result.split()
    res = []
    for i in range(0, len(words)):
        if (i != len(words)-1):
            if words[i] != words[i+1]:
                res.append(words[i])
        else:
            res.append(words[i])
    
    normalized_text = " ".join(res)
    normalized_text = " " + normalized_text + " " 
    
    # end hapus redundansi not

    adverbs = pandas.read_csv('../data/adverbs-en.csv').word.to_list()

    # adverbs = pandas.read_csv('../data/adverbs-en-dt.csv').word.to_list()
    
    i=0
    words = normalized_text.split()
    for word in words:
        if (i != len(words)-1):
            if (words[i] in negations):
                j=1
                rWords = ""
                while(words[i+j] in adverbs and (i+j+1)<len(words)):
                    rWords += words[i+j] + " "
                    j+=1
                if (j>1):
                    # normalized_text = re.sub(" " + words[i] + " " + rWords, " NOT_", normalized_text) # remove adverbs
                    normalized_text = re.sub(" " + words[i] + " " + rWords, f" {rWords}NOT_", normalized_text) # include adverb
                else:
                    normalized_text = re.sub(" " + words[i] + " " + words[i+1], " NOT_" + words[i+1], normalized_text)
        i+=1
    
    return normalized_text.strip()

# using POS tag
def negation_tag_2(comment, tagger):

    negations = pandas.read_csv('../data/negasi-en.csv').word.to_list()
    
    #check if negation exist
    is_exist_neg = False
    for negation in negations:
        if negation in comment:
            is_exist_neg = True
            break
    if is_exist_neg==False:
        return comment

    
    if tagger == 1:
        result = pos_tag_nltk(comment)
    else:
        result = pos_tag_flair(comment)
    
    result = " " + result + " "
    for word in result.split():
        if word in negations:
            result = re.sub(" " + word + " ", " no ", result)
    
    words = result.split()
    res = []
    for i in range(0, len(words)):
        if (i != len(words)-1):
            if words[i] != words[i+1]:
                res.append(words[i])
        else:
            res.append(words[i])
    
    normalized_text = " ".join(res)
    normalized_text = " " + normalized_text + " " 
    i=0
    words = normalized_text.split()
    for word in words:
        if (i != len(words)-1):
            if (words[i] in negations):
                j=1
                rWords = ""
                while("_RB" in words[i+j] and (i+j+1)<len(words)):
                    rWords += words[i+j] + " "
                    j+=1
                if (j>1):
                    # normalized_text = re.sub(" " + words[i] + " " + rWords, " NOT_", normalized_text) # remove adverb
                    normalized_text = re.sub(" " + words[i] + " " + rWords, f" {rWords}NOT_", normalized_text) # include adverb
                else:
                    normalized_text = re.sub(" " + words[i] + " " + words[i+1], " NOT_" + words[i+1], normalized_text)
        i+=1
    normalized_text = normalized_text[1:-1]
    return remove_rb_tag(normalized_text)

def negation_tag(text):
    # 0 : adv list
    # 1 : nltk
    # 2 : flair
    method = 0
    if method==0:
        return negation_tag_1(text)
    else:
        return negation_tag_2(text, method)

def punctuation_removal(text):
    new_punc = string.punctuation+'‘’“”'
    new_punc = re.sub("_", "", new_punc)
    for p in new_punc:
        text = text.replace(p, " ")

    words = []
    for word in text.split():
        words.append(word.strip())
    
    return ' '.join(words)

def preprocessing(komentar):

    stopWords = getStopWords()
    # lemmatizer = WordNetLemmatizer()
    hsl = preprocess_tweet(komentar)
    hsl = hsl.lower() #case folding
    hsl = punctuation_removal(hsl)
    hsl = negation_tag(hsl)
    # hsl = rm_html(hsl)
    hsl = hsl.translate(str.maketrans('','',string.digits))
    # hsl = lemmatization(hsl, lemmatizer) #lemmatize
    hsl = hapusStopword(hsl, stopWords) #hapus stopwords
    return hsl