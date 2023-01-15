import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from matplotlib import pyplot as plt
from scipy.spatial.distance import hamming
from sklearn.metrics.pairwise import manhattan_distances

class DBscan():
    def __init__(self, eps, min_samples, similarity):
        self.eps = eps
        self.min_samples = min_samples
        if similarity == 'sim':
            self.distance = self.sim
        elif similarity == 'hamming':
            self.distance = hamming
        elif similarity == 'jaccard':
            self.distance = self.jaccard_similarity
        elif similarity == 'manhattan':
            self.distance = manhattan_distances
        elif similarity == 'euclidean':
            self.distance = lambda x, y: np.linalg.norm(x - y)

    def sim(self, a, b):
        return len(a) + len(b) - 2 * len(set(a) & set(b))

    def jaccard_similarity(self, a, b):
        set1 = set(a)
        set2 = set(b)
        return len(set1.intersection(set2)) / len(set1.union(set2))

    def calculate_matrix(self):
        self.dist_matrix = np.full((self.X.shape[0], self.X.shape[0]), float('inf'))
        for i in tqdm(range(self.X.shape[0])):
            for j in range(i+1, self.X.shape[0]):
                if self.distance == manhattan_distances:
                    self.dist_matrix[i, j] = self.distance(self.X[i].reshape(1, -1), self.X[j].reshape(1, -1))
                else:
                    self.dist_matrix[i, j] = self.distance(self.X[i], self.X[j])

    def cluster(self, X, stop=False, dist_matrix=None):
        self.X = X
        if dist_matrix is not None:
            self.dist_matrix = dist_matrix
        else:
            print('Calculating distance matrix...')
            self.calculate_matrix()
        if stop:
            return
            
        self.cluster_labels = np.full(self.X.shape[0], -1)   
        cluster_label = 0
        for i in range(self.X.shape[0]):
            if self.cluster_labels[i] == -1:
                self.cluster_labels[i] = -2
                neighbors = self.get_neighbors(i)
                if len(neighbors) >= self.min_samples:
                    self.cluster_labels[i] = cluster_label 
                    self.expand_cluster(neighbors, cluster_label)
                    cluster_label += 1
        return self.cluster_labels 

    def get_neighbors(self, i):
        neighbors = [i]
        for j in range(i + 1, self.X.shape[0]):
            if self.dist_matrix[i, j] <= self.eps and self.cluster_labels[j] == -1:
                neighbors.append(j)
        return neighbors

    def expand_cluster(self, neighbors, cluster_label):
        shit = 0
        for j in neighbors:
            if self.cluster_labels[j] == -1:
                shit += 1
                self.cluster_labels[j] = cluster_label
                new_neighbors = self.get_neighbors(j)
                if len(new_neighbors) >= self.min_samples:
                    neighbors += new_neighbors 

def plot_graphs(results, title, cmap='Spectral'):
    fig = plt.figure() 
    fig.set_size_inches(10, 6)

    X = results['eps']
    Y = results['min_samples']

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    Z = results['noise']
    ax.scatter(X, Y, Z, c=Z, cmap=cmap)
    ax.set_xlabel('Eps')
    ax.set_ylabel('MinPts')
    #ax.set_zlabel('#Noise')
    ax.set_title('Number of noise points')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    Z = results['nb_clusters']
    ax.scatter(X, Y, Z, c=Z, cmap=cmap)
    ax.set_xlabel('Eps')
    ax.set_ylabel('MinPts')
    #ax.set_zlabel('#Clusters') 
    ax.set_title('Number of Clusters')

    fig.text(0.5, 0.05, title, ha='center', fontsize=13)
    plt.show()

def grid_search(eps_list, min_samples_list, X, dist_matrix, similarity):
    results = pd.DataFrame(columns=['eps', 'min_samples', 'nb_clusters', 'noise'])
    for eps in tqdm(eps_list):
        for min_samples in min_samples_list:
            dbscan = DBscan(eps=eps, min_samples=min_samples, similarity=similarity) 
            clusters = dbscan.cluster(X, dist_matrix=dist_matrix) 
            pred = np.copy(clusters)
            pred = np.delete(pred, np.where(pred == -2))
            results = results.append({'eps': eps,
                                    'min_samples': min_samples,
                                    'nb_clusters': len(np.unique(pred)), 
                                    'noise': list(clusters).count(-2)}, ignore_index=True)
    return results 

class NaiveBayes:
    def __init__(self):
        pass

    def train(self, data, labels):
        self.class_counts = [0] * len(np.unique(labels))
        self.word_counts = defaultdict(lambda: self.class_counts)
        for x, y in zip(data, labels):
            self.class_counts[y] += 1
            for word in x:
                self.word_counts[word][y] += 1

    def predict(self, x):
        log_probs = [math.log(self.class_counts[0]/sum(self.class_counts))]
        log_probs.append(math.log(self.class_counts[1]/sum(self.class_counts)))
        for word in x:
            log_probs[0] += math.log((self.word_counts[word][0] + 1) / (self.class_counts[0] + len(self.word_counts)))
            log_probs[1] += math.log((self.word_counts[word][1] + 1) / (self.class_counts[1] + len(self.word_counts)))
        return log_probs.index(max(log_probs))