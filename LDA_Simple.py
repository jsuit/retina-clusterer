__author__ = 'jsuit'

import Corpus
import numpy as np
from numpy import matlib
import json
#set up Corpus

corpus = Corpus.Corpus()
corpus.get_articles()
corpus.vectorize_articles()
bow = corpus.get_vect_articles()
n_docs = corpus.get_num_articles()
corpus.calc_num_terms()
n_terms = corpus.get_num_terms()
#pick number of topics
k=51
alpha = .01
beta = .001
DTMatrix = matlib.zeros((n_docs,k),dtype='float_')
TTMatrix =matlib.zeros((n_terms,k),dtype='float_')
DocVocab={}
w_tokens = False

if w_tokens:
        word_tokens = np.sum(bow.sum(axis=0))
else:
        word_tokens = n_terms

for doc_num in xrange(bow.shape[0]):
        #get the indexes of word that occur in document
        words_i = np.nonzero(bow[doc_num])[0]
        for indx in words_i:
                #for each time the word occurs in document randomly sample
                p = 0
                if w_tokens:
                        for j in range(bow[doc_num][indx]):
                                z = np.random.multinomial(1, [1/float(k)]*k, size=1).argmax()
                                DTMatrix[doc_num,z]+=1
                                TTMatrix[indx,z]+=1
                                if (doc_num,indx) not in DocVocab:
                                        DocVocab[(doc_num,indx)] = [z]
                                else:
                                        DocVocab[(doc_num,indx)].append(z)
                else:
                        z = np.random.multinomial(1, [1/float(k)]*k, size=1).argmax()
                        DTMatrix[doc_num,z]+=1
                        TTMatrix[indx,z]+=1
                        if (doc_num,indx) not in DocVocab:
                                DocVocab[(doc_num,indx)] = [z]
                        else:
                                DocVocab[(doc_num,indx)].append(z)


iters = 400
#DTMatrix.dump('DTMatrix.txt')
#TTMatrix.dump('TTMatrix.txt')

for i in range(iters):
        print i
        for doc_num in xrange(bow.shape[0]):
                words_i = np.nonzero(bow[doc_num])[0]
                for indx in words_i:
                        topics =  DocVocab[(doc_num,indx)]
                        for count, topic in enumerate(topics):
                                #take the word,topic count and decrement
                                TTMatrix[indx,topic]-=1
                                #take the document and the topic and decrement
                                DTMatrix[doc_num,topic] -=1
                                #math happens here thanks to the dirchlet being a conjugate prior to the multinomial
                                #pz is a vector representing each the probability of each topic k
                                #print DTMatrix[doc_num,:], TTMatrix[indx,:]

                                pz = np.divide(np.multiply(DTMatrix[doc_num,:] + alpha,TTMatrix[indx,:] + beta),DTMatrix.sum(axis=0)+beta*word_tokens)
                                sample_pz  = np.random.multinomial(21, np.asarray(pz/pz.sum())[0],1)
                                topic = sample_pz.argmax()
                                #DocVocab[(doc_num,indx)] = topic
                                topics[count] = topic
                                TTMatrix[indx,topic]+=1
                                DTMatrix[doc_num,topic]+=1
                        DocVocab[(doc_num,indx)] = topics
                        #DTrow = np.nonzero(DTMatrix[doc_num,:])[0]



#compute Document distribution


TopicDict = {}
Topic_DictMax = {}
for doc_num in xrange(bow.shape[0]):
        x = (DTMatrix[doc_num,:] + alpha) / (DTMatrix[doc_num,:].sum() + alpha)
        #theta_d_z
        t = np.asarray(x/x.sum())[0]
        if np.argmax(t) not in Topic_DictMax:
                Topic_DictMax[np.argmax(t)] = [doc_num]
        else:
                Topic_DictMax[np.argmax(t)].append(doc_num)
        #print x
        #print DTMatrix[doc_num,:]
        #print DTMatrix[doc_num,:].sum()
        #print doc_num,DTMatrix[doc_num,:]
        #print doc_num, DTMatrix[doc_num,:].argmax()

        #amax = DTMatrix[doc_num,:].argmax()
        arr = np.asarray(DTMatrix[doc_num,:])
        #amax = np.argpartition(array, -3)[-3:]
        amax = np.argsort(arr[0])[-3:]
        for m in amax:
                if m not in TopicDict:
                        TopicDict[m] = [doc_num]
                else:
                        TopicDict[m].append(doc_num)

from pprint import pprint

#pprint(TopicDict)
pprint(Topic_DictMax)
"""
[17, 29, 38, 46, 48, 53, 101, 122],
 1: [15, 22, 27, 37, 47, 49],
 2: [1, 11, 32, 33, 69],
 3: [77],
 4: [51, 93, 95, 105, 110],
"""


