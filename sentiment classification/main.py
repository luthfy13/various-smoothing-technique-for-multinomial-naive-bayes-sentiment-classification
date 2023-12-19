import sys
import math
from os import listdir
from os.path import isfile, join
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
import pandas as pd
import pandasql as ps
import re
from Preprocessing import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

pd.options.mode.chained_assignment = None  # default='warn'

def get_text_val(filename):
    with open(filename) as f:
        lines = f.readlines()
    return '\n'.join(lines)

# tweet dataset
i = 0
def preprocess_apply(text):
    global i
    hsl = ""
    hsl = preprocessing(text)
    print((str(i) + ' selesai diproses'), end='\r')
    i+=1
                     
    return hsl

def build_dataset_2(path_data):
    dataset = pd.read_csv(path_data, sep='\t', lineterminator='\n')
    # tags = []
    # print("for loop start")
    # for text in dataset['text']:
    #     tags.append(get_tag(text))
    # print("for loop finish")
    # dataset['tags'] = tags
    # print("begin preprocessing....")
    dataset['processed_text'] = dataset.text.apply(preprocess_apply)
    dataset.rename(columns = {'text':'comment', 'processed_text':'preprocessed'}, inplace = True)
    dataset['sentiment'] = dataset['sentiment'].replace([0], 'negative')
    dataset['sentiment'] = dataset['sentiment'].replace([1], 'positive')
    # dataset.drop(['ids', 'date', 'flag', 'user'], axis=1, inplace=True)
    query = "select * from dataset order by sentiment desc"
    df_result = ps.sqldf(query)
    return df_result

def build_dataset(path_data):
    DATASET_COLUMNS  = ["sentiment", "ids", "date", "flag", "user", "text"]
    DATASET_ENCODING = "ISO-8859-1"
    dataset = pd.read_csv(path_data, encoding=DATASET_ENCODING , names=DATASET_COLUMNS)
    dataset['processed_text'] = dataset.text.apply(preprocess_apply)
    dataset.rename(columns = {'text':'comment', 'processed_text':'preprocessed'}, inplace = True)
    dataset['sentiment'] = dataset['sentiment'].replace([0], 'negative')
    dataset['sentiment'] = dataset['sentiment'].replace([4], 'positive')
    dataset.drop(['ids', 'date', 'flag', 'user'], axis=1, inplace=True)
    return dataset.to_dict('records')

def setup_dataset(df_final_preps):
    # Go et. al dataset setup
    # df_train = df_final_preps.copy()
    # DATASET_COLUMNS  = ["sentiment", "ids", "date", "flag", "user", "text"]
    # DATASET_ENCODING = "ISO-8859-1"
    # df_test = pd.read_csv('data\\testdata.manual.2009.06.14.csv', encoding=DATASET_ENCODING , names=DATASET_COLUMNS)
    # df_test.rename(columns = {'text':'comment'}, inplace = True)
    # df_test['sentiment'] = df_test['sentiment'].replace([0], 'negative')
    # df_test['sentiment'] = df_test['sentiment'].replace([4], 'positive')
    # df_test.drop(['ids', 'date', 'flag', 'user'], axis=1, inplace=True)
    
    df_final_preps_neg = df_final_preps.iloc[:800000]
    df_final_preps_pos = df_final_preps.iloc[800000:]
    df_train = pd.concat([df_final_preps_pos[:640000], df_final_preps_neg[:640000]]) 
    df_test  = pd.concat([df_final_preps_pos[640000:], df_final_preps_neg[640000:]])
    df_train = df_train.reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)
    return df_train, df_test

def get_n_gram(text, ngram):
    if (text==""): return []
    texts = []
    texts.append(text)
    vect = CountVectorizer(ngram_range=(1,ngram))
    X_dtm = vect.fit_transform(texts)
    X_dtm = X_dtm.toarray()
    return vect.get_feature_names_out()

def get_all_features(df_train):
    numbers = []
    for i in range(1, len(df_train)+1):
        numbers.append(i)
        
    df_train['id'] = numbers

    all_features = []
    for i in range(0, len(df_train)):
        print(f"get all featires: {i+1} selesai diproses", end='\r')
        # words = str(df_train['preprocessed'][i]).split() # unigram
        words = get_n_gram(df_train['preprocessed'][i], 2) # n-grams
        for word in words:
            feature = {}
            feature['feature'] = word
            feature['sentimen'] = df_train['sentiment'][i]
            feature['id'] = df_train['id'][i]
            if word != "":
                all_features.append(feature)
    df_all_features = pd.DataFrame(all_features)
    return df_all_features

def get_unique_features(df_all_features):
    df_features = ps.sqldf("select distinct feature from df_all_features order by feature asc")
    return df_features

def get_freq_features(df_all_features, df_train):
    query = "SELECT \
                df_all_features.feature, \
                COUNT( df_all_features.feature ) AS frekuensi, \
                df_train.sentiment  \
            FROM \
                df_all_features \
            INNER JOIN df_train ON df_all_features.id = df_train.id  \
            GROUP BY \
                df_train.sentiment, \
                df_all_features.feature  \
            ORDER BY \
                df_all_features.feature ASC"
    df_result = ps.sqldf(query)
    df_result.to_csv('freq_feature_result.csv', index=False)

    df_freq_feature = pd.read_csv ('freq_feature_result.csv')
    freq_feature = df_freq_feature.to_dict('records')
    dict_freq_feature = {}
    for x in freq_feature:
        dict_freq_feature[x['feature'], x['sentiment']] = x['frekuensi']

    return dict_freq_feature

def get_count_by_class(df_all_features, df_train):
    # query = "SELECT * FROM df_all_features where feature='good'"
    query = "SELECT \
            count(df_all_features.feature) as jml_Fitur, \
            df_train.sentiment  \
            FROM \
            df_all_features \
            INNER JOIN df_train ON df_all_features.id = df_train.id \
            WHERE \
            df_train.sentiment = 'positive' UNION ALL \
            SELECT \
            count(df_all_features.feature), \
            df_train.sentiment  \
            FROM \
            df_all_features \
            INNER JOIN df_train ON df_all_features.id = df_train.id \
            WHERE \
            df_train.sentiment = 'negative'"
    df_result = ps.sqldf(query)
    return df_result

def mnb_weighting(df_features, dict_freq_feature, df_result):
    list_fitur_unik = df_features.feature.tolist()
    alpha = 0.5
    logCondProb={}
    dataFrekFitur = dict_freq_feature
    jmlFitur = {'positive': df_result['jml_Fitur'][0], 'negative': df_result['jml_Fitur'][1]}
    v = len(df_features)
    for sentimen in ['positive', 'negative']:
        # jmlFitur[sentimen] = getJmlFitur(sentimen)
        for fitur in list_fitur_unik:
            # no handling
            # logCondProb[fitur,sentimen] = math.log(float(dataFrekFitur[fitur,sentimen]) / float(jmlFitur[sentimen]), 2)
            # laplace
            try:
                # logCondProb[fitur,sentimen] = math.log(float(dataFrekFitur[fitur, sentimen] + alpha) / float(jmlFitur[sentimen] + (v*alpha)), 2) # additive smootihing
                logCondProb[fitur,sentimen] = math.log(float(dataFrekFitur[fitur, sentimen] + (2*0.5)) / float(jmlFitur[sentimen] + 2), 2) # m-estimate
            except:
                # logCondProb[fitur,sentimen] = math.log(float(0 + alpha) / float(jmlFitur[sentimen] + (v*alpha)), 2)
                logCondProb[fitur,sentimen] = math.log(float(0 + (2*0.5)) / float(jmlFitur[sentimen] + 2), 2)
    return logCondProb

def get_log_prior_prob(jmlRecordTraining, list_sentimen, jmlRecordPerSentimen):
    log_prior_prob = {}
    for sentimen in list_sentimen:
        log_prior_prob[sentimen] = math.log(jmlRecordPerSentimen[sentimen]/jmlRecordTraining, 2)
    return log_prior_prob

def get_sentiment(komentar, modelKlasifikasi, v, jmlFitur, logPriorProb):
    alpha = 0.5
    probAkhir = {}
    m = 2
    
    arrayKata = komentar.split()
    for sentimen in ['positive', 'negative']:
        probAkhir[sentimen] = float(logPriorProb[sentimen])
        for fitur in arrayKata:
            try:
                probAkhir[sentimen] = float(probAkhir[sentimen]) + float(modelKlasifikasi[fitur,sentimen])
            except:
                # probAkhir[sentimen] = float(probAkhir[sentimen]) + (math.log(float(alpha) / float(jmlFitur[sentimen] + (v*alpha)), 2)) # additive smoothing
                probAkhir[sentimen] = float(probAkhir[sentimen]) + math.log(float(0 + (2*0.5)) / float(jmlFitur[sentimen] + 2), 2) # m-estimate
    temp = -9999
    hsl = ''
    for sentimen in ['positive', 'negative']:
        if (probAkhir[sentimen] > temp):
            hsl = sentimen
            temp = probAkhir[hsl]
    return hsl

def conf_matrix(y_pred, y_test):
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)

    categories  = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
                xticklabels = categories, yticklabels = categories)

    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values"   , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)
    
def main():
    # reviews = build_dataset('../data/tweets.csv')
    # df_final_preps = pd.DataFrame(reviews)
    # df_final_preps.to_csv('prep_result.csv', index=False)
    
    df_train          = build_dataset_2("..\\data\\SST\\train.txt")
    print("Data training builded")
    # df_train = build_dataset_2("..\\data\\semeval2013\\train.txt")
    df_train.to_csv('prep_result.csv', index=False)
    df_test           = build_dataset_2("..\\data\\SST\\test.txt")
    print("Data testing builded")
    # df_test  = build_dataset_2("..\\data\\semeval2013\\test.txt")
    df_all_features   = get_all_features(df_train)
    df_features       = get_unique_features(df_all_features)
    dict_freq_feature = get_freq_features(df_all_features, df_train)
    df_result         = get_count_by_class(df_all_features, df_train)
    log_cond_prob     = mnb_weighting(df_features, dict_freq_feature, df_result)

    # test akurasi
    v = len(df_features)
    jmlFitur = {'positive': df_result['jml_Fitur'][0], 'negative': df_result['jml_Fitur'][1]}
    log_prior_prob = get_log_prior_prob(len(df_train), ['positive', 'negative'], jmlFitur)

    y_test = []
    y_pred = []
    for y in df_test['sentiment']:
        y_test.append(y)

    for y in df_test['preprocessed']:
        y_pred.append(get_sentiment(y, log_cond_prob, v, jmlFitur, log_prior_prob))

    # for i in range(0, len(y_test)):
    #     if y_test[i] != y_pred[i]:
    #         print(df_test.iloc[i]['comment'])
    #         print(df_test.iloc[i]['preprocessed'])
    #         print("Test Pred" + y_test[i], y_pred[i])
    #         print("--------------------------------")

    # conf_matrix(y_pred, y_test)
    print(classification_report(y_test, y_pred, digits=4))

if __name__ == "__main__":
    main()