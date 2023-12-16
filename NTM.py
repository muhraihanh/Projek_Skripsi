from gensim import corpora
from gensim.models import LdaModel
import re


class LDAModel:
    def __init__(self, num_topics=10, passes=15, no_below=2, no_above=0.5, keep_n=100000):
        self.num_topics = num_topics
        self.passes = passes
        self.no_below = no_below
        self.no_above = no_above
        self.keep_n = keep_n
        self.lda_model = None
        self.dictionary = None

    def build_lda_model(self, noun_phrase):

        texts = [re.findall(r'\b\w+\b', d.lower()) for d in noun_phrase]
        # Building Dictionary based on texts
        self.dictionary = corpora.Dictionary(texts)

        # Filter tokens that are too short or too frequent/rare
        self.dictionary.filter_extremes(
            no_below=self.no_below, no_above=self.no_above, keep_n=self.keep_n)

        # Building Corpus, representation of each document in the form of "bag of words" (BoW)
        corpus = [self.dictionary.doc2bow(text) for text in texts]

        # Building LDA Model with specified number of topics and passes
        self.lda_model = LdaModel(
            corpus, num_topics=self.num_topics, id2word=self.dictionary, passes=self.passes)
        
        return corpus

    def get_topic_distributions(self, corpus):
        num_topics = self.num_topics
        topic_distributions = []
        ntm_index = []

        for i, doc_bow in enumerate(corpus):
            doc_topics = self.lda_model.get_document_topics(
                doc_bow, minimum_probability=0)
            topic_vector = [0] * num_topics

            for topic_num, prop_topic in doc_topics:
                topic_vector[topic_num] = prop_topic

            topic_distributions.append(topic_vector)
            dominant_topic_num = max(doc_topics, key=lambda x: x[1])[0]
            ntm_index.append((i, dominant_topic_num))

        return topic_distributions, ntm_index



