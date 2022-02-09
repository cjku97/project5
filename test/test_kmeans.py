import pytest
import numpy as np
from cluster import (KMeans, Silhouette, make_clusters, plot_clusters, plot_multipanel)


def test_kmeans():
    clusters, labels = make_clusters(k=5, scale=1.2, m = 2)
    km = KMeans(k=5)
    assert km.k == 5
    assert km.n_obs == 0
    assert km.n_feats == 0
    assert km.clusters == []
    km.fit(clusters)
    assert km.n_obs == len(clusters)
    assert km.n_feats == len(clusters[0])
    assert len(km.clusters) == len(clusters)
    pred = km.predict(clusters)
    assert pred.all() == km.clusters.all()


def test_kmeans_failure():
	with pytest.raises(ValueError, match = "k must be at least 1"):
		KMeans(k = 0)

def test_toomanyclusters():
	clusters2, labels2 = make_clusters(k = 2, n = 10, m = 2)
	km = KMeans(k = 12)
	with pytest.raises(ValueError, match = "You cannot have more clusters than observations"):
		km.fit(clusters2)

def test_no_obs():
	km = KMeans(k = 1)
	with pytest.raises(ValueError, match = "You must have at least one observation"):
		km.fit(np.array([]))

def test_no_feats():
	km = KMeans(k = 1)
	with pytest.raises(ValueError, match = "You must have at least one feature"):
		km.fit(np.array([[]]))
    
    
    