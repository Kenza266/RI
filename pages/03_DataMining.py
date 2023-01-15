import streamlit as st
import json
import numpy as np
import pandas as pd
from Index import Index 
from collections import Counter
from sklearn.manifold import TSNE 
from matplotlib import pyplot as plt 
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN as skDB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from utils import DBscan, NaiveBayes, grid_search, plot_graphs
import warnings
warnings.filterwarnings("ignore") 

st.set_page_config(layout='wide')

if 'index' not in st.session_state:
    st.session_state.index = Index(('DS\\index.json', 'DS\\inverted.json', 'DS\\queries.json', 'DS\\ground_truth.csv', 'DS\\raw_queries.json', 'DS\\raw_docs.json'), preprocessed=True)
    st.session_state.data = np.array(list([list(i.keys()) for i in st.session_state.index.index.values()]), dtype=object) 

    X = list([list(i.keys()) for i in st.session_state.index.index.values()])
    flat = [i for j in X for i in j] 
    features = np.unique(flat)
    token_to_index_mapping = {t:i for t, i in zip(features, range(len(features)))}
    def message_to_count_vector(i, doc):
        count_vector = np.zeros(len(features))
        for token in doc:
            if token in features:
                id = token_to_index_mapping[token]
                count_vector[id] = st.session_state.index.get_weight(str(i), token)
        return count_vector.tolist()
    st.session_state.data_new = [np.array(message_to_count_vector(i, x)) for i, x in enumerate(X)]
    st.session_state.data_new = np.array(st.session_state.data_new) 

    st.session_state.reduced = PCA().fit_transform(st.session_state.data_new) 
    st.session_state.tsne = TSNE(random_state=42, n_components=2, verbose=0, perplexity=20, n_iter=3000, learning_rate='auto').fit_transform(st.session_state.data_new)


model = st.sidebar.selectbox(
    'Choose a model',
    ('Similarity on raw data', 'Jaccard on raw data', 'Hamming on one hot', 'Manhattan on one hot', 'Euclidean on reduced', 'Manhattan on reduced', 'GloVe with cosine'))

models = {'Similarity on raw data': ['Sim', 'sim', st.session_state.data], 
          'Jaccard on raw data': ['Jac', 'jaccard', st.session_state.data], 
          'Hamming on one hot': ['Ham', 'hammig', st.session_state.data_new], 
          'Manhattan on one hot': ['Manh', 'manhattan', st.session_state.data_new], 
          'Euclidean on reduced': ['Euc_Red', 'euclidean', st.session_state.reduced], 
          'Manhattan on reduced': ['Manh_Red', 'manhattan', st.session_state.reduced]}

if model == 'GloVe with cosine':
    clusters = [-1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, 3, 2, -1, -1, -1, -1, 2, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, 2, 0, 0, -1, -1, -1, -1, -1, 3, -1, -1, 2, -1, -1, -1, -1, -1, 0, 3, -1, -1, -1, 0, -1, 3, 2, 0, 1, 3, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 3, 3, 3, -1, -1, 3, 2, -1, 0, 0, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 2, 2, 3, -1, -1, 2, 3, -1, -1, -1, 2, -1, 1, -1, 1, 0, 1, -1, -1, 2, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 3, 3, 0, 0, -1, -1, 2, -1, -1, 0, 3, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, 1, -1, -1, -1, -1, -1, 3, 2, 1, -1, 3, -1, -1, 1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, 2, 2, 0, 1, -1, -1, -1, -1, -1, -1, 2, -1, -1, -1, 1, 1, 1, -1, 0, -1, 2, -1, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 3, -1, -1, -1, 3, -1, -1, -1, -1, -1, 3, -1, -1, 2, -1, 3, 3, -1, -1, -1, -1, 2, 3, 3, 2, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 0, 0, 2, -1, -1, 2, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1, 3, 0, -1, -1, 0, -1, -1, -1, 3, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, 1, 2, -1, 2, 1, -1, -1, -1, -1, 0, 3, 0, -1, 3, -1, 3, -1, 1, -1, -1, 3, 0, 1, -1, 1, 2, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 0, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 2, -1, -1, -1, -1, 1, -1, 1, -1, -1, 2, 1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 0, 1, 0, -1, 0, 2, -1, 0, -1, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, 3, 1, -1, -1, -1, -1, 1, 1, -1, -1, -1, 3, 1, 2, -1, -1, -1, 3, -1, -1, -1, 2, 2, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 0, 2, 2, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, 1, 2, -1, 1, -1, -1, -1, 3, -1, -1, -1, -1, -1, 3, 2, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, 2, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, 1, -1, -1, 2, -1, -1, 2, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, 2, -1, -1, 2, 1, -1, -1, -1, 3, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 0, -1, 3, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 0, -1, -1, 2, 2, -1, 1, -1, 1, -1, 3, -1, -1, -1, -1, -1, -1, 0, -1, 3, -1, -1, -1, 3, -1, -1, -1, -1, 0, 0, 1, -1, -1, 1, 3, -1, 0, -1, 3, -1, -1, 0, 3, -1, -1, -1, -1, 1, -1, -1, 0, 3, 0, 0, -1, -1, 0, -1, -1, -1, 3, -1, 0, 2, -1, -1, -1, -1, -1, 2, -1, -1, 0, -1, 0, 1, -1, 2, 1, 0, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1, -1, 2, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, 2, -1, -1, -1, 3, 2, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 2, 3, -1, -1, 2, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 2, -1, 1, -1, -1, -1, 0, -1, 0, 1, 0, 3, 1, 3, 1, 2, -1, -1, 0, 0, -1, 1, 0, -1, -1, 1, 3, 3, 1, 0, 1, -1, -1, -1, 0, -1, 3, 1, 0, 3, 0, -1, -1, -1, -1, 2, 1, -1, -1, -1, -1, 2, -1, 0, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 2, -1, -1, -1, 1, -1, 0, 3, -1, 0, -1, 0, 0, 1, 1, 1, 0, 2, 2, -1, -1, 3, 1, -1, -1, 2, -1, -1, 1, 0, -1, -1, -1, -1, 1, 2, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, 3, -1, -1, 2, -1, -1, -1, 1, 1, -1, 1, 1, 1, 2, 1, -1, 1, 1, 3, 1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, 1, -1, 0, 0, 1, -1, 3, -1, 0, 1, -1, 0, 1, 0, -1, -1, -1, -1, 3, -1, 0, 0, -1, -1, 2, 0, 2, -1, 1, -1, -1, -1, -1, 1, -1, 2, -1, -1, -1, 0, 0, -1, 0, -1, -1, 2, -1, 3, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 2, -1, 0, -1, -1, -1, -1, 1, 1, 0, -1, -1, -1, 3, 2, 2, -1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 3, -1, 0, -1, 1, -1, 3, -1, 0, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1, 0, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 3, 3, -1, 3, 0, -1, -1, -1, 3, 3, -1, -1, 3, -1, -1, -1, -1, 0, 2, -1, 2, 2, -1, 1, -1, 1, -1, 2, -1, -1, -1, 1, 0, 1, 3, -1, -1, 0, -1, -1, -1, 3, -1, 3, 1, -1, -1, 2, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 3, -1, 2, -1, -1, -1, -1, 1, 3, -1, 1, -1, -1, -1, 3, 1, -1, -1, -1, -1, -1, 2, -1, -1, 0, -1, -1, -1, -1, 1, 3, -1, -1, 1, -1, 3, -1, -1, 0, 1, -1, 0, 1, 0, 0, -1, 2, 0, -1, -1, -1, -1, -1, -1, -1, 2, -1, 1, 3, -1, -1, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, 1, 2, 3, -1, -1, -1, -1, 1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 3, 1, -1, 1, 1, 2, -1, -1, -1, 0, -1, 3, -1, 1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, 1, -1, 3, 3, 1, -1, 1, -1, 0, -1, 3, -1, -1, -1, -1, -1, -1, 2, 0, -1, -1, -1, -1, -1, -1, 2, -1, 1, 0, 1, 1, -1, -1, -1, -1, -1, 1, 2, -1, -1, -1, -1, -1, -1, -1, 2, 0, 2, 1, 3, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, 0, 0, 3, 0, -1, 1, -1, 0, -1, 2, -1, 2, -1, -1, 1, -1, -1, 2, 1, -1, -1, 2, -1, -1, -1, -1, 0]
    
    col1, col2 = st.columns(2)
    col1.header('Clusters distribution')
    col1.write(dict(Counter(clusters))) 
    
    
    col2.header('Silhouette score')
    col2.write(0.003)
    col2.header('2D distribution')
    col2.image('tsne_GloVe.png')

    index = index = Index(('DS\\index.json', 'DS\\inverted.json', 'DS\\queries.json', 'DS\\ground_truth.csv', 'DS\\raw_queries.json', 'DS\\raw_docs.json'), preprocessed=True)
    
    y = index.ground_truth 
    y['Relevent document'] = [clusters[doc-1] for doc in y['Relevent document']]
    y.drop(y[y['Relevent document'] == -1].index, inplace=True)
    grouped = y.groupby('Query')
    def group_to_list(group):
        return np.array(group['Relevent document'])
    ground_truth = np.array(grouped.apply(group_to_list))
    ground_truth = [np.unique(row)[np.argmax(np.unique(row, return_counts=True)[1])] for row in ground_truth]
    
    nb = NaiveBayes()
    data = [index.queries[str(i)] for i in np.unique(y['Query'])]
    labels = ground_truth
    X_train, X_test, y_train, y_test = train_test_split(data, ground_truth, test_size=0.2, random_state=86, stratify=ground_truth)
    nb.train(X_train, y_train) 
    pred = []
    for query, gt in zip(X_test, y_test):
        prediction = nb.predict(query)
        pred.append(prediction)
    report = classification_report(y_test, pred)
    col1.text("Report\n"+report)

else:
    matrix, similarity, X = models[model]
    dist_matrix = np.load('Distances//Distances_'+matrix+'.npy')
    s = not_inf = ~np.isinf(dist_matrix) 
    st.sidebar.write(round(np.min(dist_matrix), 3), round(np.max(dist_matrix[s]), 3))

    eps = st.sidebar.number_input('Eps', step=0.1, min_value=0.0, max_value=2500.0, value=105.0)
    min_samples = st.sidebar.number_input('MinPts', step=1, min_value=1, max_value=30, value=2)

    db = DBscan(eps=eps, min_samples=min_samples, similarity=similarity) 
    clusters = db.cluster(X, dist_matrix=dist_matrix)

    col1, col2 = st.columns(2)
    col1.header('Clusters distribution')
    col1.write({int(k): v for k, v in Counter(clusters).items()}) 
    if model not in ['Similarity on raw data', 'Jaccard on raw data']:
        fig, ax = plt.subplots()
        tsne = st.session_state.tsne
        plt.scatter(tsne[:, 0], tsne[:, 1], s=5, c=clusters, cmap='Spectral', label=str(clusters)) 
        cbar = plt.colorbar()
        cbar.set_ticks(np.unique(clusters))
        
        try:
            col2.header('Silhouette score')
            col2.write(round(silhouette_score(X, clusters), 4))
        except:
            pass
        col2.header('2D distribution')
        col2.pyplot(fig)
    try:

        index = index = Index(('DS\\index.json', 'DS\\inverted.json', 'DS\\queries.json', 'DS\\ground_truth.csv', 'DS\\raw_queries.json', 'DS\\raw_docs.json'), preprocessed=True)
        
        y = index.ground_truth 
        y['Relevent document'] = [clusters[doc-1] for doc in y['Relevent document']]
        y.drop(y[y['Relevent document'] == -2].index, inplace=True)
        grouped = y.groupby('Query')
        def group_to_list(group):
            return np.array(group['Relevent document'])
        ground_truth = np.array(grouped.apply(group_to_list))
        ground_truth = [np.unique(row)[np.argmax(np.unique(row, return_counts=True)[1])] for row in ground_truth]
        
        nb = NaiveBayes()
        data = [index.queries[str(i)] for i in np.unique(y['Query'])]
        labels = ground_truth
        X_train, X_test, y_train, y_test = train_test_split(data, ground_truth, test_size=0.2, random_state=15, stratify=ground_truth)
        nb.train(X_train, y_train) 
        pred = []
        for query, gt in zip(X_test, y_test):
            prediction = nb.predict(query)
            pred.append(prediction)
        report = classification_report(y_test, pred)
        col1.text("Report\n"+report)
    except:
        pass