
# coding: utf-8

# In[1]:


from __future__ import division
import argparse
import pandas as pd

import time

# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize
import re

import pickle

from nltk.tokenize import RegexpTokenizer

__authors__ = ['Vadim Benichou','Flora Attyasse','Jennifer Vial', 'Carmelo Micciche']
__emails__  = ['vadim.benichou@student-cs.fr','flora.attyasse@student-cs.fr','jennifer.vial@student-cs.fr', 
              'carmelo.micciche@student.ecp.fr']



# In[2]:


### TOKENIZATION OF THE DATA
def stopnumber(corpus):
    for w in range(len(corpus)): 
        corpus[w]=[i for i in corpus[w] if not i.isdigit()]
    return corpus

def dictionnaryoffrequency(corpus):
    wordfreq = {}
    for sent in corpus:
        for raw_word in sent:
            if raw_word not in wordfreq:
                wordfreq[raw_word] = 0 
            wordfreq[raw_word] += 1
    return(wordfreq) 

def stoprecword(corpus, wordfreq):
    num_word=[]
    for sent in corpus:
        num_word += sent
    size = len(num_word)
    del_word=[]
    for l in wordfreq:
        if wordfreq.get(l)>(size*0.002):
            del_word.append(l)
    return set(del_word)

def filteredsentence(corpus,del_word):
    filtered_sentence = []
    for sent in corpus:
        clean_sent=[]
        for word in sent:
            if word not in del_word:
                clean_sent.append(word)
        filtered_sentence.append(clean_sent)
    return filtered_sentence


####LOAD CORPUS + TOKENIZATION
def text2sentences(path): 
    sentences = []
    with open(path) as f:
        for l in f:
            sentences.append(RegexpTokenizer(r'\w+').tokenize(l.lower()))
    
    #Tokenization
    corpus=stopnumber(sentences)
    wordfreq=dictionnaryoffrequency(corpus)
    del_word=stoprecword(corpus, wordfreq)
    sentences=filteredsentence(corpus,del_word)
    
    return sentences


def loadPairs(path):
    
    data = pd.read_csv(path,delimiter='\t')
    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs


# In[5]:


class SkipGram:
    def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize = 5, minCount = 5):
        self.sentences = sentences
        self.nEmbed = nEmbed
        self.negativeRate = negativeRate
        self.winSize = winSize
        self.minCount = minCount
        self.dico_word_vectors = None 

    ### GENERATE THE PAIRS
    def wordlistweigthed(self, corpus):
        wordlistw=list(dictionnaryoffrequency(corpus).values())
        return(wordlistw)

    def wordlist(self, corpus):
        wordlist=list(dictionnaryoffrequency(corpus).keys())
        return(wordlist) 

    def sumwordfreqweighted(self, corpus):
        wordfreqweigted= [i**0.75 for i in self.wordlistweigthed(corpus)]
        return(sum(wordfreqweigted))

    def proba_list(self, wlsum, wlfreq):
        Proba_list=[]
        for word in range(len(wlfreq)):
            probabilitycoeff=wlfreq[word]**0.75/(wlsum)
            Proba_list.append(probabilitycoeff)
        return(Proba_list)

    def positive_and_negative(self, corpus, wlproba, wlword, wlfreq, negativerate=5, winSize=2):
        """winSize : +- 2"""
        w1freq_size = len(wlfreq)
        list_pairs = []    
        for sent in corpus: 
            for i in range(len(sent)): 
                for z in range(1,winSize +1):
                    if i>1 :
                        t=sent[i] 
                        try:
                            c1=sent[i-z] 
                            list_pairs.append([t, c1]) 
                            for _ in range(negativerate): 
                                negative_sample = np.random.choice(w1freq_size, 1, wlproba)  
                                for negative_word in negative_sample: 
                                    list_pairs.append([t, wlword[negative_word]])  
                        except IndexError:
                            resultat = None
                        try:
                            c3=sent[i+z]
                            list_pairs.append([t, c3])
                            for _ in range(negativerate):
                                negative_sample = np.random.choice(w1freq_size, 1, wlproba) 
                                for negative_word in negative_sample:
                                    list_pairs.append([t, wlword[negative_word]])
                        except IndexError:
                            resultat = None
                    if i==0 :
                        t=sent[i]
                        try:
                            c1=sent[i+z]
                            list_pairs.append([t, c1])
                            for _ in range(negativerate):
                                negative_sample = np.random.choice(w1freq_size, 1, wlproba) 
                                for negative_word in negative_sample:
                                    list_pairs.append([t, wlword[negative_word]])
                        except IndexError:
                            resultat = None
                    if i==1 :
                        t=sent[i]
                        try:
                            if z==1:
                                c1=sent[i-1]
                                list_pairs.append([t, c1])
                                for _ in range(negativerate):
                                    negative_sample = np.random.choice(w1freq_size, 1, wlproba) 
                                    for negative_word in negative_sample:
                                        list_pairs.append([t, wlword[negative_word]])
                        except IndexError:
                            resultat = None
                        try:
                            c2=sent[i+z]
                            list_pairs.append([t, c2])
                            for _ in range(negativerate):
                                negative_sample = np.random.choice(w1freq_size, 1, wlproba) 
                                for negative_word in negative_sample:
                                    list_pairs.append([t, wlword[negative_word]])
                        except IndexError:
                            resultat = None

        return(list_pairs)    
        
    ### OPTIMIZATION ALGORITHM 
    ##Initiation - Gradient descent
    def dico_vectors(self, corpus):
        dico=dictionnaryoffrequency(corpus)
        for value in dico:
                dico[value] = np.random.uniform(size = 100)*1e-3
        return(dico)

    ##Mathematical expressions
    def sigmoid(self, x,y):
        return 1/(1+np.exp(-np.dot(x,y)))
    def gradientpostar(self, x,y):
        return(-y*self.sigmoid(-y,x))
    def gradientnegtar(self, x,y):
        return(y*self.sigmoid(y,x))
    def gradientposcon(self, x,y):
        return(-x*self.sigmoid(-y,x))
    def gradientnegaid(self, x,y):
        return(x*self.sigmoid(y,x))
    
    ##Gradient descent
    def dotmat(self, list_pairs, dico_word_vectors, epochs, stepsize , negativerate=5):
        for i in range(epochs):
            for pair_index in range(len(list_pairs)):
                if pair_index % (negativerate + 1) == 0:
                    dico_word_vectors[list_pairs[pair_index][0]]=dico_word_vectors[list_pairs[pair_index][0]]-stepsize*self.gradientpostar(dico_word_vectors[list_pairs[pair_index][0]],dico_word_vectors[list_pairs[pair_index][1]])
                    dico_word_vectors[list_pairs[pair_index][1]]=dico_word_vectors[list_pairs[pair_index][1]]-stepsize*self.gradientposcon(dico_word_vectors[list_pairs[pair_index][0]],dico_word_vectors[list_pairs[pair_index][1]])        
                else:
                    dico_word_vectors[list_pairs[pair_index][0]]=dico_word_vectors[list_pairs[pair_index][0]]-stepsize*self.gradientnegtar(dico_word_vectors[list_pairs[pair_index][0]],dico_word_vectors[list_pairs[pair_index][1]])
                    dico_word_vectors[list_pairs[pair_index][1]]=dico_word_vectors[list_pairs[pair_index][1]]-stepsize*self.gradientnegaid(dico_word_vectors[list_pairs[pair_index][0]],dico_word_vectors[list_pairs[pair_index][1]])        
        return dico_word_vectors

    
    
    def train(self, stepsize = 0.01, epochs = 1):
        
        #Positive and Negative sampling
        wlfreq = self.wordlistweigthed(self.sentences)
        wlword = self.wordlist(self.sentences)
        wlsum = self.sumwordfreqweighted(self.sentences)
        wlproba = self.proba_list(wlsum,wlfreq)
        pairs = self.positive_and_negative(self.sentences, wlproba, wlword, wlfreq, negativerate=5)
        
        dico_word_vectors = self.dico_vectors(self.sentences)
        self.dico_word_vectors = self.dotmat(pairs,dico_word_vectors, epochs, stepsize,negativerate=5)

    def save(self,path):
        with open(path, 'wb') as f:
            pickle.dump(self.dico_word_vectors, f, pickle.HIGHEST_PROTOCOL)
            
    ### ASSESS THE SIMILARITY  
    def similarity(self,word1,word2):
        """
            computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        w1_emb=self.dico_word_vectors[word1] 
        w2_emb=self.dico_word_vectors[word2]  
        cosine_similarity = np.dot(w1_emb, w2_emb.T)/ float(np.linalg.norm(w1_emb)*np.linalg.norm(w2_emb)) 
        return ((1+cosine_similarity)/2)

    @staticmethod
    def load(path):
        with open(path, 'rb') as handle:
            return pickle.load(handle)      
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = SkipGram(sentences)
        sg.train()
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)

        sg = SkipGram.load(opts.model)
        for a,b,_ in pairs:
            print (sg.similarity(a,b))


# In[6]:


model = SkipGram(text2sentences('/Users/jennifervial/Downloads/training 2/news-commentary-v6.fr-en.en'))


# In[7]:


start = time.time()

model.train()

print('TIME RUNNING : ' ,   "{0:.2f}".format((time.time() - start) / 60), 'min')


# In[8]:


model.similarity('war','crimes')


# In[9]:


# 5 Most similar words of the target word
def most_similar(w, dico_word_vectors,k):

    similarity_vector=[] 
    words=[]
    for i in dico_word_vectors.keys():  
        similarity_vector.append(score_similarity(w,i,dico_word_vectors))
        words.append(i)
    similarity_vector = np.array(similarity_vector)
    sorted_similarity=np.argsort(similarity_vector) 
    k_most_similar=[]  
    for i in sorted_similarity[sorted_similarity.shape[0]-(k+1):-1]: 
        k_most_similar.append(words[i])
    return k_most_similar


# In[ ]:


print('5 most similars words:',most_similar('disservice',dico_word_vectors,k=5))

