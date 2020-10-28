
import gensim
import pyLDAvis.gensim

def save_topic_visualization(method, model, corpus, dictionary, output_path):
    """ Save topics visualization to a given output path 
    
    :param method: method type
    :param model: topic model
    :param corpus: generated bow corpus
    :param dictionary: generated dictionary
    :param output_path: path to save visualization
    """
    if method == "LDA":
        vis_data = pyLDAvis.gensim.prepare(model, corpus, dictionary)
        pyLDAvis.save_html(vis_data, output_path)
    elif method == "LDA_MALLET": 
        model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(model)
        vis_data = pyLDAvis.gensim.prepare(model, corpus, dictionary)
        pyLDAvis.save_html(vis_data, output_path)
    elif method == "HDP":
        for (token, uid) in dictionary.token2id.items():
            dictionary.id2token[uid] = token
        vis_data = pyLDAvis.gensim.prepare(model, corpus, dictionary)
        pyLDAvis.save_html(vis_data, output_path)
    else:
        pass