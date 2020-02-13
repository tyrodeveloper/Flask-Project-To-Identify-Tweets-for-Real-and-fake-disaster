from flask import Flask,request, render_template
from main import prediction,prediction_sentiment
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
from functions import wordcloud_draw
app = Flask(__name__)


@app.route("/")
def home():
    return render_template('home.html', name = 'Welcome to Sentiment Analysis of disaster Tweets', url ='static/images/nlp.png')

@app.route("/getWordcloudofPositiveSentiment")
def getWordcloudpositive():
    df = pd.read_csv("train.csv")
    df1 = pd.read_csv("test.csv")
    if os.path.exists('static/images/plot_positive.png'):
        os.remove('static/images/plot_positive.png')
    df['text'] = df['text'].str.replace("[^A-Za-z ]", "")
    train , test = train_test_split(df,test_size=0.3)
    train_1 = train[ train['target'] == 1]
    train_1 = train_1['text']
    train_0 = train[ train['target'] == 0]
    train_0 = train_0['text']
    print("Simple words")
    suf =wordcloud_draw(train_0,'white','positive')
    return render_template('wordcloud.html', name = 'WordCloud For Positive Tweets', url ='static/images/plot_'+suf+'.png')

@app.route("/getWordcloudofNegativeSentiment")
def getWordcloudnegative():
    df = pd.read_csv("train.csv")
    df1 = pd.read_csv("test.csv")
    if os.path.exists('static/images/plot_negative.png'):
        os.remove('static/images/plot_negative.png')
    df['text'] = df['text'].str.replace("[^A-Za-z ]", "")
    train , test = train_test_split(df,test_size=0.3)
    train_1 = train[ train['target'] == 1]
    train_1 = train_1['text']
    train_0 = train[ train['target'] == 0]
    train_0 = train_0['text']
    print("disasterous words")
    suf=wordcloud_draw(train_1,'black','negative')
    return render_template('wordcloud.html', name = 'WordCloud For Negative Tweets', url ='static/images/plot_'+suf+'.png')
    

# @app.route("/getbarofsentiments")
# def getBar():

# @app.route("/mfhashtag")
# def getHashtag():

@app.route("/mlmodelaccuracy")
def getAccuracy():
    accuracy=prediction()
    return render_template('wordcloud.html', name =accuracy , url ='static/images/nlp.png')

@app.route("/predictsentiment", methods=['GET', 'POST'])
def getSentimentPrediction():
    sentiment = request.form['sentiment']
    print(sentiment)
    Sentiment_type=prediction_sentiment(sentiment)
    return render_template('wordcloud.html', name =Sentiment_type , url ='static/images/nlp.png')