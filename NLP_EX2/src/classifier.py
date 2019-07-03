
# coding: utf-8

# ## Sentiment Analyzer 

# ### Students:  
# Carmelo Micciche  
# Vadim Benichou  
# Jennifer Vial   
# Flora Attyasse


#LIBRARIES
import xmltodict
import csv
import pandas as pd
from bs4 import BeautifulSoup 
from nltk.corpus import stopwords
import numpy as np
import time
import matplotlib.pyplot as plt
#Sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import HashingVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import GridSearchCV
from textblob import TextBlob
from nltk.stem import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

nltk.downloader.download('vader_lexicon')
stopwords_nltk = stopwords.words('english')
snowball_stemmer = SnowballStemmer('english')
import re

nltk.download('punkt')
from nltk.stem import PorterStemmer
lancaster=PorterStemmer()

from nltk.corpus import wordnet
nltk.downloader.download('wordnet')


with open('/Users/vadimbenichou/Desktop/NLP_Assignment_2/resources/slang.txt') as file:
    slang_map = dict(map(str.strip, line.partition('\t')[::2])
    for line in file if line.strip())
slang_words = sorted(slang_map, key=len, reverse=True)

sid = SentimentIntensityAnalyzer()

class Classifier:
    def __init__(self, path_data, path_dev):
        self.path_data = path_data
        self.path_dev = path_dev
        self.data = None
        self.data_dev = None
        self.predict = None
        self.accuracy = None
    
    def load_data(self):
        self.data = pd.read_csv(self.path_data, sep = '\t', header = None)
        self.data_dev = pd.read_csv(self.path_dev, sep = '\t', header = None)
                
    def clean_txt(self, txt, stopwords_nltk):
        txt = BeautifulSoup(txt,"html.parser",from_encoding='utf_8').get_text()
        txt = txt.lower()
        querywords = txt.split()
        txt  = [word for word in querywords if word not in stopwords_nltk] 
        return txt
    
    def identity_tokenizer(self, text):
        return text
    
    def stemSentence(self, sentence, snowball_stemmer):
        token_words=word_tokenize(sentence)
        stem_sentence=[]
        for word in token_words:
            stem_sentence.append(snowball_stemmer.stem(word))
            stem_sentence.append(" ")
        return "".join(stem_sentence)
    
    def Lancaster(self, sentence, lancaster):
        token_words=word_tokenize(sentence)
        stem_sentence=[]
        for word in token_words:
            stem_sentence.append(lancaster.stem(word))
            stem_sentence.append(" ")
        return "".join(stem_sentence)
    
    def correctSentence(self, sentence):
        token_words=word_tokenize(sentence)    
        stem_sentence=[]
        for word in token_words:
            stem_sentence.append(str(TextBlob(word).correct()))
            stem_sentence.append(" ")
        return "".join(stem_sentence)

    def countSlang(self, sentence, slang_words):
        """ Input: a text, Output: how many slang words and a list of found slangs """
        slangCounter = 0
        slangsFound = []
        sentence=sentence.lower()
        tokens = nltk.word_tokenize(sentence)
        for word in tokens:
            if word in slang_words:
                slangsFound.append(word)
                slangCounter += 1
        return slangCounter
    
    def countMultiExclamationMarks(self, sentence):
        """ Replaces repetitions of exlamation marks """
        return len(re.findall(r"(\!)\1+", sentence))

    def countMultiQuestionMarks(self, sentence):
        """ Count repetitions of question marks """
        return len(re.findall(r"(\?)\1+", sentence))

    def countMultiStopMarks(self, sentence):
        """ Count repetitions of stop marks """
        return len(re.findall(r"(\.)\1+", sentence))

    def countElongated(self, sentence):
        """ Input: a text, Output: how many words are elongated """
        regex = re.compile(r"(.)\1{2}")
        return len([word for word in sentence.split() if regex.search(word)])

    def countEmoticons(self, sentence):
        return len(re.findall(':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:', sentence))
    
    def addNotTag(self, sentence):
        l=['no', 'never', 'not']
        #sentence=word_tokenize(sentence) 
        sent = []
        s=0
        for word in sentence:
            if word in l:
                s=s+1
                sent.append(word)
                sent.append(" ")
            else:
                sent.append(word)
                sent.append(" ")
        sent = sent[:-1]
        return  s
    
    def addCapTag(self, sentence):
        """ Finds a word with at least 3 characters capitalized and adds the tag ALL_CAPS_ """
        sentence=word_tokenize(sentence) 
        sent = []
        s=0
        for word in sentence:
            if(len(re.findall("[A-Z]{3,}", word))):
                s=s+1
                sent.append(word)
                sent.append(" ")
            else:
                sent.append(word)
                sent.append(" ")
        sent = sent[:-1]
        return  s
    
    def train(self, stopwords_nltk, snowball_stemmer, lancaster, slang_words, sid):
        self.load_data()
        
        data = self.data
        data_dev = self.data_dev
        
        vectorizer = HashingVectorizer(tokenizer=self.identity_tokenizer, lowercase = False, n_features=5000)
        data_reg = data.copy()
        data_test_reg = data_dev.copy()
                                      
        #Rename columns 
        data_reg.columns = ['positivity', 'type', 'word', 'emplacement', 'sentence']
        data_test_reg.columns = ['positivity', 'type', 'word', 'emplacement', 'sentence'] 

        #Drop word and emplacement
        data_reg = data_reg.drop(['word', 'emplacement'], axis = 1)
        data_test_reg = data_test_reg.drop(['word', 'emplacement'], axis = 1)

        #Categories to features
        data_reg = pd.concat([data_reg.drop(['type'], axis = 1), pd.get_dummies(data_reg['type'])], axis = 1)
        data_test_reg = pd.concat([data_test_reg.drop(['type'], axis = 1), pd.get_dummies(data_test_reg['type'])], axis = 1)

        #Emoticons
        data_reg['Emoticons'] = data_reg['sentence'].apply(self.countEmoticons)
        data_test_reg['Emoticons'] = data_test_reg['sentence'].apply(self.countEmoticons)

        #Exclamation Marks
        #data_reg['ExclamationMarks'] = data_reg['sentence'].apply(self.countMultiExclamationMarks)
        #data_test_reg['ExclamationMarks'] = data_test_reg['sentence'].apply(self.countMultiExclamationMarks)

        #QuestionMarks
        #data_reg['QuestionMarks'] = data_reg['sentence'].apply(self.countMultiQuestionMarks)
        #data_test_reg['QuestionMarks'] = data_test_reg['sentence'].apply(self.countMultiQuestionMarks)

        #StopMarks
        data_reg['StopMarks'] = data_reg['sentence'].apply(self.countMultiStopMarks)
        data_test_reg['StopMarks'] = data_test_reg['sentence'].apply(self.countMultiStopMarks)

        #Slang
        data_reg['Slangs'] = data_reg['sentence'].apply(lambda x : self.countSlang(x, slang_words))
        data_test_reg['Slangs'] = data_test_reg['sentence'].apply(lambda x : self.countSlang(x, slang_words))

        #Correct
        data_reg['sentence'] = data_reg['sentence'].apply(self.correctSentence)
        data_test_reg['sentence'] = data_test_reg['sentence'].apply(self.correctSentence) #moins bon score a tester apres le clean

        #Compound polarity
        data_reg['compound'] = data_reg['sentence'].apply(sid.polarity_scores).apply(lambda x : x['compound'])
        data_test_reg['compound'] = data_test_reg['sentence'].apply(sid.polarity_scores).apply(lambda x : x['compound'])
        data_reg['polarity_blob'] = data_reg['sentence'].apply(lambda x: TextBlob(x).polarity)
        data_test_reg['polarity_blob'] = data_test_reg['sentence'].apply(lambda x: TextBlob(x).polarity)

        #posivitypolarity
        #data_reg['posivitypolarity'] = data_reg['sentence'].apply(sid.polarity_scores).apply(lambda x : x['pos'])
        #data_test_reg['posivitypolarity'] = data_test_reg['sentence'].apply(sid.polarity_scores).apply(lambda x : x['pos'])

        #negativepolarity
        #data_reg['negativepolarity'] = data_reg['sentence'].apply(sid.polarity_scores).apply(lambda x : x['neg'])
        #data_test_reg['negativepolarity'] = data_test_reg['sentence'].apply(sid.polarity_scores).apply(lambda x : x['neg'])

        #neutralpolarity
        #data_reg['neutralpolarity'] = data_reg['sentence'].apply(sid.polarity_scores).apply(lambda x : x['neu'])
        #data_test_reg['neutralpolarity'] = data_test_reg['sentence'].apply(sid.polarity_scores).apply(lambda x : x['neu'])

        #Stemming
        data_reg['sentence'] = data_reg['sentence'].apply(lambda x : self.stemSentence(x, snowball_stemmer))
        data_test_reg['sentence'] = data_test_reg['sentence'].apply(lambda x : self.stemSentence(x, snowball_stemmer)) #moins bon score a tester apres le clean

        #data_reg['sentence'] = data_reg['sentence'].apply(lambda x : self.Lancaster(x, lancaster))
        #data_test_reg['sentence'] = data_test_reg['sentence'].apply(lambda x : self.Lancaster(x, lancaster))

        #CountElongated
        data_reg['countElongated']=data_reg['sentence'].apply(self.countElongated)
        data_test_reg['countElongated']=data_test_reg['sentence'].apply(self.countElongated)

        #Majuscule word

        data_reg['majusculenumber']=data_reg['sentence'].apply(self.addCapTag)
        data_test_reg['majusculenumber']=data_test_reg['sentence'].apply(self.addCapTag)

        #Cleaning sentences
        data_reg['sentence'] = data_reg['sentence'].apply(lambda x: self.clean_txt(x, stopwords_nltk))
        data_test_reg['sentence'] = data_test_reg['sentence'].apply(lambda x: self.clean_txt(x, stopwords_nltk))

        #Count negation
        data_reg['negations'] = data_reg['sentence'].apply(self.addNotTag)
        data_test_reg['negations'] = data_test_reg['sentence'].apply(self.addNotTag)


        #CountElongated
        #data_reg['sentence']=data_reg['sentence'].apply(self.replaceElongated)
        #data_test_reg['sentence']=data_test_reg['sentence'].apply(self.replaceElongated)

        data_train = data_reg
        data_test = data_test_reg

        #Hashing : Tokenization - Train 
        response = vectorizer.fit_transform(data_train['sentence'].values)
        data_transform_train = pd.concat([data_train.drop(['positivity', 'sentence'], axis = 1),pd.DataFrame(response.todense())], axis = 1)

        #X And Y - Train
        X_train = data_transform_train.copy()
        Y_train = data_train[['positivity']].copy()
        Y_train.loc[Y_train['positivity']=='positive', 'positivity'] = 1
        Y_train.loc[Y_train['positivity']=='neutral', 'positivity'] = 0
        Y_train.loc[Y_train['positivity']=='negative', 'positivity'] = 2
        Y_train = Y_train.astype('int')

        #Hashing : Tokenization - Test
        response = vectorizer.transform(data_test['sentence'].values)
        data_transform_test = pd.concat([data_test.drop(['positivity', 'sentence'], axis = 1),pd.DataFrame(response.todense())], axis = 1)

        #X and Y Test
        X_test = data_transform_test.copy()
        Y_test = data_test[['positivity']].copy()
        Y_test.loc[Y_test['positivity']=='positive', 'positivity'] = 1
        Y_test.loc[Y_test['positivity']=='neutral', 'positivity'] = 0
        Y_test.loc[Y_test['positivity']=='negative', 'positivity'] = 2
        Y_test = Y_test.astype('int')

        #Logistic Regression
        grid={"C":np.logspace(-3,5,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
        logreg=LogisticRegression()
        logreg_cv=GridSearchCV(logreg,grid,cv=10)
        logreg_cv.fit(X_train.values,Y_train.values.reshape(-1))
        res = logreg_cv.predict(X_test.values)
        acc = accuracy_score(Y_test.values.reshape(-1), res)
        self.predict = res 
        self.accuracy = acc
  
path_data = '/Users/vadimbenichou/Desktop/traindata.csv'
path_dev = '/Users/vadimbenichou/Desktop/devdata.csv'
model = Classifier(path_data, path_dev)
model.train(stopwords_nltk, snowball_stemmer, lancaster, slang_words, sid)
model.accuracy
