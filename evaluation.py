from gensim.models.coherencemodel import CoherenceModel
from visualization import save_topic_visualization, save_topic_wordclouds
from shared.utils import load_from_pickle
from shared.utils import dump_to_pickle
from shared.utils import dump_to_json
from shared.utils import dump_to_txt
from shared.utils import make_dirs 
from tqdm import tqdm
import pandas as pd
import numpy as np
from tm import TM
import logging
import time
import glob
import os

pd.set_option('display.max_columns', None)
pd.set_option("max_rows", None)

class Evaluation(object):
    """ Class for generating topic model and evaluation files.
    """
    def __init__(self, lang_code, method, version="1.1", norm_type="LEMMA", k=20, num_words=10, passes=20, workers=8, alpha=0.01, 
                 eta=0.01, no_below=20, no_above=0.6, mallet_path='mallet-2.0.8/bin/mallet', onepass=True, power_iters=1000, 
                 iterations=500, random_state=42, kappa=0.1, normalize=True, chunksize=2000, num_wordcloud_words=150):

        self.k = k                   
        self.lang_code = lang_code
        self.method = method
        self.version = version
        self.norm_type = norm_type
        self.num_words = num_words
        self.passes = passes
        self.workers = workers
        self.alpha = alpha
        self.eta = eta
        self.no_below = no_below
        self.no_above = no_above
        self.mallet_path = mallet_path
        self.random_state = random_state
        self.onepass = onepass
        self.power_iters = power_iters
        self.iterations = iterations
        self.normalize = normalize
        self.kappa = kappa
        self.chunksize = chunksize
        self.num_wordcloud_words = num_wordcloud_words
       
    def create_model(self, token_lists, output_path):
        """ Generate model & evaluation file to a given output path 
        
        :param token_lists: list of document tokens
        :param output_path: path to save model, dictionary, corpus, evaluation files
        """
        try:
            # define output path
            subdir = "{}_k_{}_{}_{}".format(self.lang_code, str(self.k), self.method, self.version)
            models_path = output_path + "/models/" + subdir
            dict_path = output_path + "/dictionary/"
            corpus_path = output_path + "/corpus/"
            eval_path = output_path + "/evaluation/" + subdir
            
            # create directories
            make_dirs(output_path)
            make_dirs(models_path)
            make_dirs(dict_path)
            make_dirs(corpus_path)
            make_dirs(eval_path)
            make_dirs(eval_path + "/wordcloud")

            # create topic model & fit token_lists
            tm = TM(
                k=self.k, method=self.method, passes=self.passes, workers=self.workers, alpha=self.alpha, eta=self.eta, 
                no_below=self.no_below, no_above=self.no_above, onepass=self.onepass, power_iters=self.power_iters, 
                iterations=self.iterations, random_state=self.random_state, kappa=self.kappa, normalize=self.normalize, 
                chunksize=self.chunksize
            )

            tm.fit(token_lists)
           
            # evaluate model
            c_v = self.compute_coherence(tm.model, tm.dictionary, tm.corpus, token_lists, measure='c_v')
            u_mass = self.compute_coherence(tm.model, tm.dictionary, tm.corpus, token_lists, measure='u_mass')
            perplexity = self.compute_perplexity(tm.model, tm.corpus)
     
            # hyperparameters & evaluation metrics
            metrics = dict()
            if self.method == "LDA":
                metrics = {
                    "k"             : self.k,
                    "passes"        : self.passes,
                    "workers"       : self.workers,
                    "eta"           : self.eta,
                    "alpha"         : self.alpha,
                    "random_state"  : self.random_state,
                    "lang_code"     : self.lang_code,
                    "num_docs"      : len(token_lists),
                    "num_words"     : self.num_words,
                    "method"        : self.method,
                    "version"       : self.version,
                    "norm_type"     : self.norm_type,
                    "c_v"           : c_v,
                    "u_mass"        : u_mass,
                    "perplexity"    : perplexity
                }
            elif self.method == "LDA_MALLET":
                metrics = {
                    "k"             : self.k,
                    "workers"       : self.workers,
                    "num_docs"      : len(token_lists),
                    "num_words"     : self.num_words,
                    "iterations"    : self.iterations,
                    "lang_code"     : self.lang_code,
                    "method"        : self.method,
                    "norm_type"     : self.norm_type,
                    "version"       : self.version,
                    "c_v"           : c_v,
                    "u_mass"        : u_mass,
                }
            elif self.method == "LSI":
                metrics = {
                    "k"             : self.k,
                    "num_docs"      : len(token_lists),
                    "num_words"     : self.num_words,
                    "onepass"       : self.onepass,
                    "power_iters"   : self.power_iters,
                    "lang_code"     : self.lang_code,
                    "method"        : self.method,
                    "norm_type"     : self.norm_type,
                    "version"       : self.version,
                    "c_v"           : c_v,
                    "u_mass"        : u_mass,
                }
            elif self.method == "HDP":
                metrics = {
                    "k"             : self.k,
                    "num_docs"      : len(token_lists),
                    "num_words"     : self.num_words,
                    "lang_code"     : self.lang_code,
                    "method"        : self.method,
                    "norm_type"     : self.norm_type,
                    "version"       : self.version,
                    "eta"           : self.eta,
                    "alpha"         : self.alpha,
                    "random_state"  : self.random_state,
                    "c_v"           : c_v,
                    "u_mass"        : u_mass,
                }
            else:
                metrics = {
                    "k"             : self.k,
                    "num_docs"      : len(token_lists),
                    "num_words"     : self.num_words,
                    "iterations"    : self.iterations,
                    "random_state"  : self.random_state,
                    "normalize"     : self.normalize,
                    "lang_code"     : self.lang_code,
                    "method"        : self.method,
                    "norm_type"     : self.norm_type,
                    "version"       : self.version,
                    "c_v"           : c_v,
                    "u_mass"        : u_mass,
                }
            
            # save topic model, dictionary, corpus 
            tm.model.save(models_path + "/model.gensim")
            tm.dictionary.save(dict_path + "/dict.gensim")
            dump_to_pickle(tm.corpus, corpus_path + "/corpus.pkl")
            
            # save metrics, topic_terms dataframe 
            dump_to_json(metrics, eval_path + "/evaluation.json", sort_keys=False)
            self.save_topic_terms(tm.model, eval_path + '/topic_terms.txt')

            # save topic visualization, topic wordclouds
            save_topic_visualization(self.method, tm.model, tm.corpus, tm.dictionary, eval_path + "/topics.html")
            save_topic_wordclouds(tm.model, self.num_wordcloud_words, eval_path + "/wordcloud")

        except Exception:
            logging.error('error occured', exc_info=True)

    def compute_coherence(self, model, dictionary, corpus, token_lists, measure):
        """ Compute coherence score for a given topic model 
        
        :param model: topic model
        :param dictionary: generated dictionary
        :param corpus: generated bow corpus
        :param token_lists: list of document tokens
        :param measure: coherence score
        """
        coherence = 0.0
        cm = CoherenceModel(
            model=model, 
            dictionary=dictionary, 
            corpus=corpus, 
            texts=token_lists, 
            coherence=measure
        )
        coherence = cm.get_coherence()
        coherence = "{0:.3f}".format(coherence)
        return coherence

    def compute_perplexity(self, model, corpus):
        """ Compute model perplexity 
        
        :param model: topic model
        :param corpus: bow corpus
        :return: perplexity score
        """
        perplexity = None
        if self.method == "LDA":
            perplexity = model.log_perplexity(corpus)
            perplexity = "{0:.3f}".format(perplexity)
        return perplexity

    def get_topics(self, model):
        """ Get topics for given model 
        
        :param model: topic model
        :return: list of topics
        """
        topics = []
        if self.method == "HDP":
            num_topics = len(model.get_topics().tolist())
            topics = [[(term, round(wt, 3))
                            for term, wt in model.show_topic(n, topn=self.num_words)]
                                for n in range(0, num_topics)]
        else:
            topics = [[(term, round(wt, 3))
                            for term, wt in model.show_topic(n, topn=self.num_words)]
                                for n in range(0, model.num_topics)]
        return topics
        
    def get_topic_terms_df(self, model):
        """ Generate topic_term dataframe for a given model 
        
        :param model: topic model
        :return: dataframe
        """
        topics = self.get_topics(model)
        topic_terms_df = None
        if self.method == "HDP":
            num_topics = len(model.get_topics().tolist())
            topic_terms_df = pd.DataFrame([', '.join([term for term, wt in topic])
                                for topic in topics], columns = ['Topic terms'],
                                    index=[t for t in range(0, num_topics)])
            topic_terms_df.reset_index(inplace=True)
            topic_terms_df = topic_terms_df.rename(columns={'index': 'id'})
        else:
            topic_terms_df = pd.DataFrame([', '.join([term for term, wt in topic])
                                for topic in topics], columns = ['Topic terms'],
                                    index=[t for t in range(0, model.num_topics)])
            topic_terms_df.reset_index(inplace=True)
            topic_terms_df = topic_terms_df.rename(columns={'index': 'id'})
        return topic_terms_df

    def save_topic_terms(self, model, output_path):
        """ Generate and save topic_terms dataframe to a given output_path
        
        :param model: topic model
        :param output_path: output path
        """
        topic_terms_df = self.get_topic_terms_df(model)
        topic_terms_df.to_string(output_path, index=False)


if __name__ == "__main__":

    chunks_prep_path = abs_path = os.path.dirname(os.path.abspath("__file__")) + "/chunks_prep"
    output_path = os.path.dirname(os.path.abspath("__file__")) + "/output"	
    
    token_lists = []	
    for filename in os.listdir(chunks_prep_path):	
        chunk_no = filename.split("_")[1]	
        token_list = load_from_pickle(chunks_prep_path + "/" + filename + "/token_lists.pkl")	
        token_lists.append(token_list)	

    token_lists = [item for sublist in token_lists for item in sublist]	
    print("NUM DOCS: ", len(token_lists))	

    start_time = time.time()	
    ev = Evaluation(
        lang_code="en", method="LDA", version="1.1", 
        k=50, alpha=0.1, eta=0.1, num_words=15
    )	
    ev.create_model(token_lists, output_path=output_path)	
    print("--- LDA: %s seconds ---" % (time.time() - start_time))

    start_time = time.time()	
    ev = Evaluation(
        lang_code="en", method="LDA_MALLET", version="1.1", 
        k=50,  alpha=0.1, eta=0.1, num_words=15
    )	
    ev.create_model(token_lists, output_path=output_path)	
    print("--- LDA_MALLET: %s seconds ---" % (time.time() - start_time))

    start_time = time.time()	
    ev = Evaluation(
        lang_code="en", method="LSI", version="1.1", 
        k=50,  alpha=0.1, eta=0.1, num_words=15
    )	
    ev.create_model(token_lists, output_path=output_path)	
    print("--- LSI: %s seconds ---" % (time.time() - start_time))

    start_time = time.time()	
    ev = Evaluation(
        lang_code="en", method="NMF", version="1.1", 
        k=50, alpha=0.1, eta=0.1, num_words=15, passes=5
    )	
    ev.create_model(token_lists, output_path=output_path)	
    print("--- NMF: %s seconds ---" % (time.time() - start_time))

    start_time = time.time()	
    ev = Evaluation(
        lang_code="en", method="HDP", version="1.1", 
        k=50, alpha=0.1, eta=0.1, num_words=15
    )	
    ev.create_model(token_lists, output_path=output_path)	
    print("--- HDP: %s seconds ---" % (time.time() - start_time))