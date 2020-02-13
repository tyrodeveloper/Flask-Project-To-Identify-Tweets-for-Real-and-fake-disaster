import nltk
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
# %matplotlib inline
from nltk.classify import SklearnClassifier
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud,STOPWORDS
w_features =[]

stopwords_set = set(stopwords.words("english"))

def get_word_features(tweets):
    all = []
    for (words, sentiment) in tweets:
        all.extend(words)
    wordlist = nltk.FreqDist(all)
    print(wordlist)
    global w_features
    w_features = wordlist.keys()
    return w_features

def get_cleaned_tweets(train):
    tweets = []
    for index, row in train.iterrows():
        words_filtered = [e.lower() for e in row.text.split() if len(e) > 3]
        words_cleaned = [word for word in words_filtered
            if 'http' not in word
            and not word.startswith('@')
            and not word.startswith('#')
            and word.find("û") == -1
            and word != 'rt']
        words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
#     words_with_lemmatization = [wordnet_lemmatizer.lemmatize(word) for word in words_without_stopwords ]
#     words_with_stemming = [stemmer.stem(word) for word in words_without_stopwords ]
    #words_without_stopwords.apply(lambda x: [stemmer.stem(i) for i in x])
        tweets.append((words_without_stopwords, row.target))
    print(tweets)
    return tweets

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

def wordcloud_draw(data,color,name):
    words= ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
#                                 and not word.startswith('@')
#                                 and not word.startswith('#')
#                                 and word.find("û") == -1
                                and word != 'RT'
                                and word != 'rt'
                            ])
    wordcloud = WordCloud(stopwords=STOPWORDS, 
                        background_color=color,
                        width=2500,
                        height=2000,
                        collocations=False
                        ).generate(cleaned_word)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    # os.remove('static/images/plot1.png')
    # if os.path.exists('static/images/plot1.png'):
    #     print('Exist')
    # else:
    #     print('deleted')
    plt.savefig('static/images/plot_'+name+'.png')
    return name
    