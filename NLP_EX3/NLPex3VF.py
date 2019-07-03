__authors__ = ['carmelo_micciche','vadim_benichou','flora_attyasse', 'jennifer_vial']
__emails__  = ['carmelo.micciche@student.ecp.fr','vadim.benichou@gmail.com','flora.attyasse@student-cs.fr', 'jennifer.vial@student-cs.fr']

import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pkl
import numpy as np
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords
from nltk import word_tokenize
import io
import string
import pandas as pd


class DialogueManager:
    def __init__(self):
        self.vect = TfidfVectorizer(analyzer='word',ngram_range=(1,1))
             
        self.embeddings = {}
        
        self.stopwords = set(stopwords.words('english')) - set(['about'])
        
    def load(self,path):
        with open(path,'rb') as f:
            self.vect = pkl.load(f)


    def save(self,path):
        with open(path,'wb') as fout:
            pkl.dump(self.vect,fout)


    def train(self,data, method = "vectorizer", nmax = 100000):
        if method == "vectorizer":
            self.vect.fit(data)

    
   
   
    def best_with_utterance(self,utterance,options):
       
        Xtext = [utterance] + options
        X = self.vect.transform(Xtext)
        X = normalize(X,axis=1,norm='l2') ###
        shape = X.shape[0]
        best_option_utte = [X[0] @ X[i].T for i in range(1, shape)]
        best_option_utte = np.array([best_option_utte[i][0,0] for i in range(len(best_option_utte))])
        return best_option_utte
    
    
    def best_with_yourpers(self, yourper, options):
        
        yourper2 = [' '.join(yourper[i]) for i in range(len(yourper))]
        Xtext = yourper2 + options
        X = self.vect.transform(Xtext)
        X = normalize(X,axis=1,norm='l2') ###
        shape = X.shape[0]
        shape_yourper = len(yourper)
        shape_options = len(options)
        best_option_yourper = np.zeros(shape_options)
        for j in range(shape_yourper):
            matrix_your_options = [X[j] @ X[i].T for i in range(shape_yourper, shape)]
            best_option_yourper = best_option_yourper + np.array([matrix_your_options[i][0,0] for i in range(len(matrix_your_options))])
        
        return best_option_yourper
    
    def best_with_parpers(self, parper, options):
       
        parper2 = [' '.join(parper[i]) for i in range(len(parper))]
        Xtext = parper2 + options
        X = self.vect.transform(Xtext)
        X = normalize(X,axis=1,norm='l2')
        shape = X.shape[0]
        shape_parper = len(parper)
        shape_options = len(options)
        best_option_parper = np.zeros(shape_options)
        for j in range(shape_parper):
            matrix_par_options = [X[j] @ X[i].T for i in range(shape_parper, shape)]
            best_option_parper = best_option_parper + np.array([matrix_par_options[i][0,0] for i in range(len(matrix_par_options))])
        
        return best_option_parper
    
  

    def findBest(self,utterance,options, yourper, parper):

            X2 = self.best_with_yourpers(yourper, options)
            X3 = self.best_with_parpers(parper, options)
            X_par_your = X2 + X3
            idx = np.argsort(X_par_your)[-5:][::-1]
            option_df=pd.DataFrame(options)
            option_df.columns=['sentence']
            option_df=option_df.reset_index(drop=True)
            option_df1=option_df.iloc[idx,:]
            options=list(zip(option_df1['sentence']))
            for i in range(len(options)):
                options[i]=' '.join(options[i])
            X4=self.best_with_utterance(utterance,options)
            idx = np.argmax(X4)

            return options[idx]

def loadData(path):
   
    with open(path) as f:
        descYou, descPartner = [],[]
        dialogue, data_utterance , data_options = [],[],[]
        for l in f:
            l=l.strip()
            lxx = l.split()
            idx = int(lxx[0])
            if idx == 1:
                if len(dialogue) != 0:
                    yield descYou,  descPartner, dialogue , data_utterance, data_options
                # reinit data structures
                descYou, descPartner = [],[]
                dialogue , data_utterance , data_options = [],[],[]

            if lxx[2] == 'persona:':
                # description of people involved
                if lxx[1] == 'your':
                    description = descYou
                elif lxx[1] == "partner's":
                    description = descPartner
                else:
                    assert 'Error, cannot recognize that persona ({}): {}'.format(lxx[1],l)
                l = l.replace(str(lxx[0]) + ' ' + str(lxx[1]) + ' ' + str(lxx[2]) + ' ', '')
                description.append(l)

            else:
                # the dialogue
                lxx = l.split('\t')
                utterance = ' '.join(lxx[0].split()[1:])
                answer = lxx[1]
                options = [o for o in lxx[-1].split('|')]
                dialogue.append( (idx, utterance, answer, options))
                data_utterance.append(utterance)
                data_options.append(options)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='path to model file (for saving/loading)', required=True)
    parser.add_argument('--text', help='path to text file (for training/testing)', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true')
    group.add_argument('--test', action='store_true')
    parser.add_argument('--gen', help='enters generative mode')

    opts = parser.parse_args()

    dm = DialogueManager()
    if opts.train:
        text = []
        for _,_, dialogue, _, _ in loadData(opts.text):
            for idx, _, _,options in dialogue:
                text.extend(options)
        dm.train(text)
        dm.save(opts.model)
    else:
        assert opts.test,opts.test
        dm.load(opts.model)
        i = 0
        j = 0
        for me, parper, dialogue, _, _ in loadData(opts.text):
            for idx, utterance, answer, options in dialogue:
                print(idx,dm.findBest(utterance, options, me, parper))



