__author__ = 'jsuit'
import article_loader
from sklearn.feature_extraction.text import CountVectorizer
class Corpus():
        def __init__(self):
                self.articles = {}
                self.num_docs = 0
                self.num_terms = -1
                self.bow = None
                self.vectorizer = CountVectorizer()

        def get_articles(self):
                articles = article_loader.get_articles()
                for article in articles:
                        self.articles[article[0]['title']] = article[0]['text']
                self.num_docs = len(self.articles.keys())

        def get_num_terms(self):
                if self.num_terms ==-1:
                        print 'no terms'
                else:
                        return self.num_terms

        def get_num_articles(self):
                return self.num_docs

        def calc_num_terms(self):
                if self.bow ==  None:
                        print 'no bag of words'

                else:
                        self.num_terms = self.bow.shape[1]


        def vectorize_articles(self):
                text_c  = [text for text in self.articles.values()]
                self.bow = self.vectorizer.fit_transform(text_c).toarray()

        def get_vect_articles(self):
                return self.bow



