import pytest
import numpy as np
from cluster import (KMeans, Silhouette, make_clusters, plot_clusters, plot_multipanel)


def test_kmeans():
    clusters, labels = make_clusters(k=5, scale=1.2, m = 2)
    km = KMeans(k=5)
    km.fit(clusters)
    pred = km.predict(clusters)
    # scores = Silhouette().score(clusters, pred)
    # print(np.mean(scores))
    # plot_multipanel(clusters, labels, pred, scores)
    pass
    