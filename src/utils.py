def topic_diversity(topics, top_n=10):
    topics_truncated = []
    for topic in topics:
        topics_truncated.append(topic[:top_n])
    
    topics_union = [word for topic in topics_truncated for word in topic]
    
    num_unique = len(set(topics_union))
    td = num_unique/len(topics_union)
    
    return td


# code for c-TF-IDF from: https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6
def c_tf_idf(documents, m, ngram_range=(1, 1)):
    from sklearn.feature_extraction.text import CountVectorizer
    
    count = CountVectorizer(ngram_range=ngram_range).fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)
    
    return tf_idf, count


def extract_top_n_words_per_cluster(tf_idf, count, docs_per_cluster, n=20):
    words = count.get_feature_names_out()
    labels = list(docs_per_cluster.cluster)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words


def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['cluster'])
                     .preprocessed_text
                     .count()
                     .reset_index()
                     .rename({"preprocessed_text": "size"}, axis='columns')
                     .sort_values("size", ascending=False))
    return topic_sizes