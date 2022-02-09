import pytest
import numpy as np
from sklearn.metrics import silhouette_score
from cluster import (KMeans, Silhouette, make_clusters, plot_clusters, plot_multipanel)

def test_silhouette():
	clusters, labels = make_clusters(k=5, scale=1.2, m = 2)
	km = KMeans(k=5)
	km.fit(clusters)
	pred = km.predict(clusters)
	scores = Silhouette().score(clusters, pred)
	scores2 = silhouette_score(clusters, pred)
	assert abs(np.mean(scores) - scores2) < 1e-6

def test_silhouette_small():
	X = np.array([[0,1,3,4],[1,0,3,4],[3,4,0,1],[4,3,1,0]])
	y = np.array([1,1,2,2])
	assert abs(np.mean(Silhouette().score(X,y)) - silhouette_score(X,y)) < 1e-6
	

def test_silhouette2():
	clusters, labels = make_clusters(k = 4, m = 3)
	km = KMeans(k = 4)
	km.fit(clusters)
	pred = km.predict(clusters)
	scores = Silhouette().score(clusters, pred)
	scores2 = silhouette_score(clusters, pred)
	assert abs(np.mean(scores) - scores2) < 1e-6	

