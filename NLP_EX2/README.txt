
# READ ME

#### Introduction:



Through this project, we build a sentiment analyzer based on Natural Processing Language methods. This approach is more and more popular especially for Twitter analysis. For a given topic contextualized in a given review, one should discover if the opinion is positive, negative or neutral.  
The input: Sentences cleaned  
The outputs: Polarity label

### Libraries

The libraries used are the following:  
xmltodict, pandas , BeautifulSoup , nlkt, numpy, sklearn, pysparjk, textblob.

### Pre-processing

This phase aims at tokenizing, cleaning sentences and making each element relevant in order to analyze the context. Please find bellow the different functions created:  
1.TextBlob: Compute the polarity score of the sentences.  

2.Valder: Compute the polarity score of the sentences. 

3.Cleantext: Remove stop stopwords and lower the text.

4.HashingVecorizer: Tokenize sentences. 

5.StemSentence: Convert plural into singular words. 

6.Correctsentence: Correct the typing errors in the review (ex: replace "luve" by "love").  

7.countslang: Count the number of slangs in a review (ex: "OMG"). To use this function, one needs to upload the file _slang.txt_.  

8.countMultiExclamationMarks: Count the number of exclamation marks in a review.  

9.countMultiQuestionMarks: Count the number of question marks in a review.  

10.countMultiStopMarks: Count the number of stop marks in a review.

11.countElongated: Count the number of elongated words in a review (ex: "LLLLOOOVVVEEE"). This type of elements accentuate the sentiment. 

12.countEmoticons: Count the number of emoticons in a review. It also amplifies the opinion. 

13.addNotTag: Count the number of negative elements ("not", "no", "never") in a review.   

14.addCapTag: Count the number of words with at least 3 characters capitalized.   

15.replaceElongated: Replace the elongated word by the real word (ex: "LLLLOOOVVVEEE" replaced by "love").  

### Training 

To label correctly the review "positive", "neutral" or "negative", the model has been trained with support vector machine, a random forest and a logistic regression.  

__-Support Vector Machine__: Sklearn was used to perform the SVM. Cross validation with 5 splits. Grid search to optimize the Gamma and the degree parameters. 

__-Random Forest__: Cross validation. Grid search to optimize the number of trees, the depth, the leaves, the splits, the number of features. 

__-Logistic Regression__:Cross validation. Grid search to see if it is better with Lasso or Ridge. 

### Results

Several simulations with different combinations of features have been run. The valder lexicon from the library nlkt is pretty performant and manages to ensure 80% of accuracy by itself with the logistic regression. Nevertheless, without the valder libraries and with all the features created we managed also to reach 79% of accuracy with the logistic regression.  

The training method that ensures the best results is the Logistic Regression with an accuracy of 80,8%. The accuracy from the random forest is up to 78% so is the one from the Support Vector Machine. 

The combination of features that ensure the best accuracy is the following:  
The Hashing Vectorizer  
The number of emoticons  
The number of stop marks  
The number of slangs  
The correction of typing errors  
The polarity with the Valder lexicon from the NLKT library  
The stemming of the sentence  
The counting of elongated words  
The number of words written in capital letters  
The clean of the sentences  
The number of negative words  

A detailed study of the labels predicted has been realized:   
Our algorithm manages to correctly label 92% of the positive reviews, 63% of the negative's but does not manage to detect the neutral's (0 out of 14 has been detected). This is possibly due to the few amount of neutral data available. 

### Sources

_DataCamp_ https://www.datacamp.com  
_Bing Liu: Sentiment Analysis_ https://www.cs.uic.edu/~liub/FBS/Sentiment-Analysis-tutorial-AAAI-2011.pdf   
_Jurafsky & Martin: POS-tags_ https://web.stanford.edu/~jurafsky/slp3/8.pdf  
_Saif M. Mohammad, Svetlana Kiritchenko, and Xiaodan Zhu: Building the State-of-the-Art in Sentiment Analysis of Tweets_ https://arxiv.org/pdf/1308.6242.pdf

### Students:  
Carmelo Micciche  
Vadim Benichou  
Jennifer Vial   
Flora Attyasse
