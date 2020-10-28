
import gensim
import pyLDAvis.gensim
import matplotlib.pyplot as plt
from wordcloud import WordCloud

plt.rcParams['figure.figsize'] = (10, 5)

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

def save_topic_wordclouds(model, num_topics, num_wordcloud_words, output_path):
    """ Save topic wordclouds to a given output path 
    
    :param model: topic model
    :param num_topics: num topics
    :param num_wordcloud_words: total num words in wordcloud
    """
    topic = 0
    for topic in range(num_topics):
        topic += 1
        topic_words_freq = dict(model.show_topic(topic-1, topn=num_wordcloud_words))
        plt.figure()
        wordcloud = WordCloud(background_color="white").generate_from_frequencies(topic_words_freq) 
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.savefig(output_path + "/topic_" + str(topic-1))
        plt.clf()