__author__ = 'jsuit'

from sklearn.feature_extraction.text import CountVectorizer
class Vectorize():

        def __init__(self, v_func, stopWords = None):
                self.v_func = v_func
