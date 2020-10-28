
import gensim
from gensim.models import LdaMulticore, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.models.nmf import Nmf
from gensim.corpora import Dictionary
import logging

class TM(object):
    """ Class for generating topic model
    """
    def __init__(self, k, method, passes=20, workers=8, alpha=0.001, eta=0.001, no_below=20, no_above=0.6,
                 mallet_path='mallet-2.0.8/bin/mallet', onepass=True, power_iters=1000, iterations=500, 
                 random_state=42, normalize=True, kappa=.1, chunksize=2000):
        
        self.k = k
        self.method = method
        self.passes = passes
        self.workers = workers
        self.alpha = alpha
        self.eta = eta
        self.no_below = no_below
        self.no_above = no_above
        self.onepass = onepass
        self.power_iters = power_iters
        self.mallet_path = mallet_path
        self.iterations = iterations
        self.random_state = random_state
        self.normalize = normalize
        self.chunksize = chunksize
        self.kappa = kappa
        self.model = None
        self.dictionary = None
        self.corpus = None

    def fit(self, token_lists):
        """ Generate topic model, dictionary, corpus from token lists 
        
        :param token_lists: list of document tokens
        """
        try:
            if self.method in {'LDA', 'LDA_MALLET', 'LSI', "HDP", 'NMF' }:
                
                # create dictionary & filter common words 
                self.dictionary = Dictionary(token_lists)
                self.dictionary.filter_extremes(no_below=self.no_below, no_above=self.no_above)
                # generate bow-corpus
                self.corpus = [self.dictionary.doc2bow(text) for text in token_lists]

                if self.method == "LDA":
                    self.model = LdaMulticore(
                        corpus=self.corpus, id2word=self.dictionary,
                        num_topics=self.k, alpha=self.alpha, eta=self.eta, 
                        passes=self.passes, workers=self.workers, random_state=self.random_state
                    )
                elif self.method == "LDA_MALLET":
                    self.model = LdaMallet(
                        corpus=self.corpus, id2word=self.dictionary, 
                        num_topics=self.k, alpha=self.alpha, mallet_path=self.mallet_path, 
                        iterations=self.iterations, workers=self.workers
                    )
                elif self.method == "LSI":
                    self.model = LsiModel(
                        corpus=self.corpus, id2word=self.dictionary, 
                        num_topics=self.k, onepass=self.onepass, power_iters=self.power_iters
                    )
                elif self.method == "HDP":
                    self.model = HdpModel(
                        corpus=self.corpus, id2word=self.dictionary,
                        alpha=self.alpha, eta=self.eta, random_state=self.random_state
                    )
                else:
                    self.model = Nmf(
                        corpus=self.corpus, id2word=self.dictionary, num_topics=self.k,
                        chunksize=self.chunksize, passes=self.passes, kappa=self.kappa,
                        minimum_probability=0.01, w_max_iter=300, w_stop_condition=0.0001,
                        h_max_iter=100, h_stop_condition=0.001, eval_every=10, normalize=self.normalize,
                        random_state=self.random_state
                    )
            else:
                raise Exception('method not exist')
        except Exception:
            logging.error("exception occured", exc_info=True)   

            