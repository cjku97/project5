import pytest
import numpy as np
from cluster import (KMeans, Silhouette, make_clusters, plot_clusters, plot_multipanel)


def test_kmeans():
    clusters, labels = make_clusters(k=7, scale=0.5, m = 3)
    km = KMeans(k=4)
    km.fit(clusters)
    pred = km.predict(clusters)
    scores = Silhouette().score(clusters, pred)
    print(np.mean(scores))
    plot_multipanel(clusters, labels, pred, scores)
    