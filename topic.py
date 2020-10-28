from gensim.models import LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.models.nmf import Nmf
from gensim.models.phrases import Phraser
from gensim.models import Phrases
from gensim.corpora import Dictionary
from shared.utils import load_from_pickle
from preprocessing import Preprocessing
from utility import Utility
from functools import lru_cache
import pandas as pd 
import numpy as np 
import logging
import re
import os

abs_path = os.path.dirname(os.path.abspath("__file__")) + "/output"

class Topic(object):
    """ Class for predicting topics in new documents

    :param lang_code: language text
    :param method: topic method
    :param version: model version number
    :param k: number of topics
    :param top_k: top k predictions
    :param clean_text: boolean flag for cleaning text
    :param num_words: num words per topic 
    :param min_words: min number of words for prediction
    :param min_conf_score: minimum confidence threshold 
    """
    def __init__(self, lang_code, method="LDA", version="1.1", k=50, top_k=1, clean_text=False, num_words=10, 
                min_words=10, min_conf_score=0.10):
          
        self.lang_code = lang_code
        self.method = method
        self.version = version
        self.k = k                      
        self.top_k = top_k 
        self.clean_text = clean_text
        self.num_words = num_words     
        self.min_words = min_words
        self.min_conf_score = min_conf_score
        
        self.models_path = abs_path + "/models"
        self.dict_path = abs_path + "/dictionary"
        subdir = "{}_k_{}_{}_{}".format(self.lang_code, str(self.k), self.method, self.version)

        self.valid_langs = ["en"]
        if lang_code in self.valid_langs:
            self.p = Preprocessing(lang_code=lang_code)
            if method in {"LDA", "LDA_MALLET", "LSI", "HDP", "NMF" }:
                self.filepath_model = self.models_path + "/" + subdir + "/model.gensim"
                self.filepath_dict = self.dict_path + "/dict.gensim"
                if os.path.isfile(self.filepath_model) and os.path.isfile(self.filepath_dict): 
                    self.model = self.load_model(self.filepath_model)
                    self.dictionary = self.load_dictionary(self.filepath_dict)
            
    @lru_cache(maxsize=128)
    def load_model(self, filepath):
        """ Load model from filepath

        :param filepath: 
        :return: Gensim model
        """
        model = None
        if self.method == "LDA":
            model = LdaModel.load(filepath)
        elif self.method == "LDA_MALLET":
            model = LdaMallet.load(filepath)
        elif self.method == "LSI":
            model = LsiModel.load(filepath)
        elif self.method == "NMF":
            model = Nmf.load(filepath)
        else:
            model = HdpModel.load(filepath)
        return model

    @lru_cache(maxsize=128)
    def load_dictionary(self, filepath):
        """ Load dictionary from filepath 
        
        :param filepath: 
        :return: Gensim dictionary
        """
        dictionary = Dictionary.load(filepath)
        return dictionary
    
    def text_preprocessing_pipeline(self, text):
        """ Text normalization pipeline 
        
        :param text: 
        :return: list of tokens
        """ 
        token_list = self.p.text_preprocessing(text)
        token_list = self.p.make_bigrams([token_list])
        return token_list

    def generate_bow_corpus(self, token_list, dictionary):
        """ Generate bag-of-words corpus  
        
        :param token_list: list of tokens
        :param dictionary: word to number mappings 
        :return: bag-of-words vectors
        sample:
        """
        corpus = [dictionary.doc2bow(text) for text in token_list]
        return corpus
    
    def get_top_k_preds(self, model, corpus):
        """ Get top k predictions for a given model and corpus
        
        :param model: Gensim model
        :param corpus: bag-of-words
        :return: list of list of tuples
        """
        top_k_preds = []
        topic_predictions = model[corpus]
        top_k_preds = [
            [(topic, round(wt, 3)) for topic, wt in sorted(
                                        topic_predictions[i], key=lambda row: -row[1])[:self.top_k]]
                                            for i in range(len(topic_predictions))]
        return top_k_preds
    
    def predict_topic(self, text):
        """ Predict topic for new documents 
        
        :param text: text to predict
        :return: python dictionary
        Sample:
            {
            "predictions":[
                {
                    "id":40,
                    "confidence":0.317,
                    "topic_terms":[
                        {
                            "term":"task",
                            "weight":"0.055"
                        },
                        {
                            "term":"training",
                            "weight":"0.031"
                        },
                        {
                            "term":"train",
                            "weight":"0.022"
                        },
                        {
                            "term":"expert",
                            "weight":"0.018"
                        },
                        {
                            "term":"architecture",
                            "weight":"0.010"
                        }
                    ]
                }
            ],
            "message":"successful"
            }
        """
        try:
            prediction = dict()
    
            if text:
                if Utility.get_doc_length(text) > self.min_words:
                    if self.lang_code in self.valid_langs:
                        if self.clean_text:
                            text = Utility.clean_text(text)
                            
                        # generate token list from given text
                        token_list = self.text_preprocessing_pipeline(text)
                        
                        topic_preds = []
                        if self.method in { "LDA", "LDA_MALLET", "LSI", "HDP", "NMF" }:
                            # check if model and dictionary files exist
                            if os.path.isfile(self.filepath_model) and os.path.isfile(self.filepath_dict): 
                                # generate bow corpus as [(token_id: freq)]                
                                corpus = self.generate_bow_corpus(token_list, self.dictionary)  # [(4, 1), (14, 2), (20, 1), (28, 1), (33, 1)]
                                # get top k predictions as list of list of tuples               # [[(doc_id, confidence)]]  
                                top_k_preds = self.get_top_k_preds(self.model, corpus)          # [[(40, 0.317), (5, 0.144), (34, 0.085)]]
                                top_k_preds = top_k_preds[0]                                    # [(40, 0.317), (5, 0.144), (34, 0.085)]
                                # loop through all top k predictions and get metadata
                                for id, confidence in top_k_preds:
                                    topic_pred = dict()
                                    topic_terms = []
                                    for term, weight in self.model.show_topic(id, topn=self.num_words):
                                        term_weight = dict()
                                        term_weight["term"] = term
                                        term_weight["weight"] = "{0:.3f}".format(weight)
                                        topic_terms.append(term_weight)
                                    topic_pred['id'] = id
                                    topic_pred['confidence'] = "{0:.3f}".format(confidence)
                                    topic_pred["topic_terms"] = topic_terms
                                    topic_preds.append(topic_pred)

                                if topic_preds:
                                    max_conf_topic = max(topic_preds, key=lambda k: k["confidence"])
                                    max_conf_score = max_conf_topic.get("confidence")
                                    
                                    if float(max_conf_score) <= self.min_conf_score:
                                        return "unknown topic, confidence below threshold"
                                    
                                    prediction["predictions"] = topic_preds
                                    prediction["message"] = "successful"
                                else:
                                    return "no topics found"
                            else:
                                return "model not found"
                        else:
                            return "method not exist"  
                    else:
                        return "language not supported"
                else:
                    return 'required at least {} words for prediction'.format(self.min_words)
            else:
                return "required textual content"
            return prediction
        except Exception:
            logging.error("exception occured", exc_info=True)    