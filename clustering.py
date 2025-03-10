# This script performs clustering on a dataset using various ML techniques.
# It loads data, preprocesses it (scaling), and then applies several clustering algorithms.
# For each algorithm, it calculates a silhouette score to evaluate the clustering quality
# and generates a dendrogram (for hierarchical clustering methods) to visualize the results.
# The clustering methods include: SVM-RBF, Random Forest, MLPRegressor + KMeans, KNN, Gaussian Mixture Model, Decision Tree, and Bagging Classifier.


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform, cdist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# Load data
file_path = "path/to/yourfile" # data/data.csv
data = pd.read_csv(file_path)

# Separate cultivar names and features
names = data['Cultivar']
X = data.drop('Cultivar', axis=1)

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Function to calculate proximity matrix
def proximity_matrix_from_trees(clf, X):
    terminals = [tree.apply(X) for tree in clf.estimators_]
    n_trees = len(clf.estimators_)
    proximity_matrix = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(i, X.shape[0]):
            proximity_matrix[i, j] = sum(1 for t in terminals if t[i] == t[j]) / n_trees
            proximity_matrix[j, i] = proximity_matrix[i, j]
    return proximity_matrix


# Function to calculate clustering and silhouette score
def perform_clustering_and_evaluate(clustering_method, X_scaled, names, title):
    try:
        if callable(clustering_method):
            labels, Z = clustering_method(X_scaled, names)
        else:
            labels, Z = clustering_method.fit_predict(X_scaled), None

        if labels is not None and len(np.unique(labels)) > 1:
            silhouette_avg = silhouette_score(X_scaled, labels)
            print(f"{title} Silhouette Score: {silhouette_avg}")

        if Z is not None:
            plt.figure(figsize=(10, 5))
            dendrogram(Z, labels=names.values, orientation='top', truncate_mode='lastp', p=30, leaf_rotation=45., leaf_font_size=8., show_contracted=True)
            plt.xlabel('Cultivar')
            plt.ylabel('Distance')
            plt.title(title)
            plt.show()

    except Exception as e:
        print(f"{title} Error during clustering: {e}")


# 1. SVM-RBF clustering
def svm_rbf_clustering(X_scaled, names):
    clf = SVC(kernel='rbf', gamma='scale')
    clf.fit(X_scaled, names)
    dist_matrix = clf.decision_function(X_scaled)
    Z = linkage(dist_matrix, method='ward')
    labels = clf.predict(X_scaled)
    return labels, Z

perform_clustering_and_evaluate(svm_rbf_clustering, X_scaled, names, "SVM-RBF")

# 2. Random Forest clustering
def random_forest_clustering(X_scaled, names):
    clf = RandomForestClassifier()
    clf.fit(X_scaled, names)
    proximity_matrix = 1 - (1 - clf.oob_decision_function_) if hasattr(clf, "oob_decision_function_") else proximity_matrix_from_trees(clf, X_scaled)
    Z = linkage(squareform(1 - proximity_matrix), method='ward')
    labels = clf.predict(X_scaled)
    return labels, Z

perform_clustering_and_evaluate(random_forest_clustering, X_scaled, names, "Random Forest")

# 3. MLPRegressor + KMeans Clustering
def mlpregressor_kmeans_clustering(X_scaled, names, n_clusters=3, random_state=42):
    encoder = MLPRegressor(hidden_layer_sizes=(10, 5), activation='relu', solver='adam', max_iter=1000, random_state=random_state)
    encoder.fit(X_scaled, X_scaled)
    encoded_data = encoder.predict(X_scaled)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(encoded_data)
    Z = linkage(pdist(encoded_data), method='ward')
    return labels, Z

perform_clustering_and_evaluate(lambda x, y: mlpregressor_kmeans_clustering(x, y), X_scaled, names, "MLPRegressor + KMeans")

# 4. KNN clustering
def knn_clustering(X_scaled, names):
    knn = KNeighborsClassifier()
    knn.fit(X_scaled, names)
    distances, _ = knn.kneighbors(X_scaled)
    knn_distances = distances.max(axis=1)
    condensed_dist_matrix = squareform(pdist(X_scaled, metric='euclidean'))
    np.fill_diagonal(condensed_dist_matrix, knn_distances)
    Z = linkage(condensed_dist_matrix, method='ward')
    labels = knn.predict(X_scaled)
    return labels, Z

perform_clustering_and_evaluate(knn_clustering, X_scaled, names, "KNN-based")

# 5. GaussianMixture clustering
def gmm_clustering(X_scaled, names, n_components=3, random_state=42):
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    labels = gmm.fit_predict(X_scaled)
    centers = gmm.means_
    dist_matrix = cdist(X_scaled, centers, metric='mahalanobis', VI=np.linalg.inv(gmm.covariances_[labels]))
    Z = linkage(dist_matrix, method='ward')
    return labels, Z

perform_clustering_and_evaluate(lambda x, y: gmm_clustering(x, y), X_scaled, names, "GaussianMixture (Original Data)")


# 6. Decision Tree clustering
def decision_tree_clustering(X_scaled, names):
    dt = DecisionTreeClassifier()
    dt.fit(X_scaled, names)
    Z = linkage(X_scaled, method='ward', optimal_ordering=True, metric='euclidean')
    labels = dt.predict(X_scaled)
    return labels, Z

perform_clustering_and_evaluate(decision_tree_clustering, X_scaled, names, "Decision Tree")


# 7. Bagging Classifier Clustering
def bagging_clustering(X_scaled, names):
    y = data['Cultivar']
    bagging_clf = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10, random_state=0)
    bagging_clf.fit(X_scaled, y)
    proximity_matrix = proximity_matrix_from_trees(bagging_clf, X_scaled)
    np.fill_diagonal(proximity_matrix, 1)
    condensed_dist_matrix = squareform(1 - proximity_matrix)
    Z = linkage(condensed_dist_matrix, method="ward")
    labels = bagging_clf.predict(X_scaled)
    return labels, Z

perform_clustering_and_evaluate(bagging_clustering, X_scaled, names, "Bagging Classifier")




