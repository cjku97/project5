import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score

class Silhouette:
    def __init__(self, metric: str = "euclidean"):
        """
        inputs:
            metric: str
                the name of the distance metric to use
        """
        self.metric = metric

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features. 

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        # get matrix parameters
        n_obs = len(X)
        n_feats = len(X[0])
        n_clusts = len(np.unique(y))
        # initialize scores array
        scores = np.zeros(n_obs)
        # generate cluster dictionary
        cluster_dict = {}
        for c in range(1, n_clusts+1):
        	cluster_dict[c] = []
        for i in range(0, n_obs):
        	c = y[i]
        	cluster_dict[c] = cluster_dict[c] + [i]
        
        # get datapoint distances
        dists = cdist(X, X, metric = self.metric)
        
        # get silhouette scores for each observation
        for i in range(0, n_obs):
        	# find intra-cluster distances 
        	# i.e. the mean distance between i and all other data points in the same cluster
        	cluster_i = cluster_dict[y[i]]
        	a_sum = 0
        	for j in cluster_i:
        		if j == i:
        			continue
        		else:
        			a_sum = a_sum + dists[i][j]
        	a_i = (1/(len(cluster_i)-1)) * a_sum
        	# find inter-cluster distances
        	# i.e. the smallest mean distance of i to all points in any other cluster, 
        	# of which i is not a member
        	b_sum = 0
        	b_i = np.inf
        	for k in range(1, n_clusts+1):
        		if k == y[i]:
        			continue
        		else:
        			cluster_k = cluster_dict[k]
        			for j in cluster_k:
        				b_sum = b_sum + dists[i][j]
        			b_k = (1/len(cluster_k)) * b_sum
        		b_i = min(b_k, b_i)
        	scores[i] = (b_i - a_i)/max(a_i,b_i)
        
        return(scores)
    
    		
    	
    	


