{\rtf1\ansi\ansicpg1252\cocoartf1561\cocoasubrtf400
{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs36 \cf0 README SKIP GRAM PROJECT \

\fs24 \
	
\b\fs28 Project Description
\b0\fs24 \
\
The skipgram project aims at programming an algorithm in Python to determine for any given word its embedding and to assess its degree of similarity with another given word. At the end of the exercise our class may return for a given word of a given corpus the N-closest ones. This file is a detailed descritpion of the creation process for the algorithm.  \
\
	
\b\fs28 Libraries used
\b0\fs24  \
\

\b For basic manipulation:\

\b0 -numpy  \
-pandas  \
-argparse   \
-pickle  \

\b For tokenisation process:\

\b0 -nltk  \
-future\
\
	
\b\fs28 Tokenisation of the data
\b0\fs24  \
\
Definition of text2sentences function:
\i  in order to separte the sentence, the words of the corpus, remove the ponctuation elements et stock them into a list.
\i0 \
\
Definition of stopnumber function: 
\i to remove the digit elements of the list.
\i0 \
\
Definition of dictionnaryoffrequency function: 
\i to create a dictionary in which the key is the word of the corpus and the value is its frequency. This function will be used to remove the words that appear too many times.
\i0 \
\
Definition of stoprecword function: 
\i to remove the word for which the frequency is above N% of the total number of words in the corpus. Use of dictionnaryoffrequency function. 
\i0 \
\

\b Parameters:   \

\b0 Percentage to remove word: 0.02\
\
	
\b\fs28 Generate the pairs
\b0\fs24 \
\

\b\fs26 Generate the negative pairs\

\b0\fs24 \
Definition of init function: 
\i to initialize the class.
\i0 \
\
Definition of wordlistweigthed function: t
\i o extract and store the values of our dictionnary which are the frequency of the word. Use for our function probalist. \

\i0 \
Definition of wordlist function: 
\i to extract and store the keys of our dictionnary which are the words of our corpus. Use for our function probalist. \

\i0 \
Definition of sumwordfreqweighted function: 
\i to compute  Incremental Skip-gram Model with Negative Sampling rate. \

\i0 \
Definition of proba_list function: 
\i to create a list with the Incremental Skip-gram Model with Negative Sampling rate for each word.\
\

\i0\b\fs26 Generate the positive and negative pairs\

\b0\fs24 \
Definition of positive_and_negative function: 
\i to create our list of pairs; for each positive pair it generates N negative pairs.
\i0 \
\

\b parameters:
\b0 \
negative rate: 5  \
WinSize: 2
\i \

\i0 \

\b\fs28 	Optimization Algorithm \

\b0\fs24 \
Definition of dico_vectors function: 
\i to create an embedding (random vector of size N) for each word. Multiply with 1e-3 to get values small enough for the optimization.
\i0 \
\
Definition of sigmoid function: 
\i to create the function that compute the sigmoid between two embeddings. Use for the computation of the gradients.
\i0 \
\
Definition of gradientpostar function: 
\i to compute the gradient for the target vector update when matched with positive pairs.
\i0 \
\
Definition of gradientnegtar function:
\i  to compute the gradient for the target vector update when matched with negative pairs.
\i0 \
\
Definition of gradientposcon function:
\i  to compute the gradient for the context word vector update.
\i0 \
\
Definition of gradientnegaid function:
\i  to compute the gradient for the negative associated word vector update.
\i0 \
\
Definition of dotmat function: 
\i to actualise the vectors at each iterations. For each pair both vectors are updated. 
\i0 \
\
\

\b Paramaters:
\b0 \
Size of the Vector: 100  \
Step for the gradient: 0.01 to get reasonable embeddings after the optimisation. \
negativerates: 5  \
Number of epochs: 1, get satisfying results with only one epoch.\
\

\b\fs28 	Assess the similarity \

\b0\fs24 \
Definition of score_similarity function: 
\i to compute the similarity between two word , the larger the output the more similar the 2 word are.
\i0 \
\
Definition of most_similar function: to find the N most common word of a word.\
\

\b Parameters:  \

\b0 Most common words: 5 \
\
	
\b\fs28 External References
\b0\fs24 \
\

\b word2vec Explained Deriving Mikolov et al.\'92s Negative-Sampling Word-Embedding Method
\b0  - 
\i Yoav Goldberg and Omer Levy\

\i0 \

\b Speech and Language, Processing vector semantics part II
\b0  - 
\i Dan Jurafsky and James Martin
\i0   \
\

\b Word to Vectors\uc0\u8202 \'97\u8202 Natural Language Processing
\b0  - 
\i Shubham Agarwal
\i0 \
\

\b Distributed Representations of Words and Phrases and their Compositionality - 
\i\b0 Jeffrey Dean, Tomas Mikolov_  
\i0 \
\

\b Word2Vec Tutorial Part I: The SkipGram Model
\b0  - 
\i Alex Minnaar 
\i0 \
\

\b Text Data: Word Embedding
\b0  
\i -Yizhou Sun UCLA CS\
\
ATTYASSE Flora  \
BENICHOU Vadim  \
MICCICHE Carmelo  \
VIAL Jennifer \
 
\i0 \
\
\
\
}