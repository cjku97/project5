import numpy as np
from scipy.spatial.distance import cdist

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
        
        # get datapoint distances
        dists = cdist(X, X, metric = self.metric)
        
        # get silhouette scores for each observation
        for i in range(0, n_obs):       	
        	size_c_i = sum(y == y[i])
        	if size_c_i == 1:
        		scores[i] = 0
        	else:
        		# find intra-cluster distances 
        		# i.e. the mean distance between i and all other data points 
        		# in the same cluster
        		a_i = np.sum(dists[y == y[i]][:,i])/(size_c_i - 1)
        		# find inter-cluster distances
        		# i.e. the smallest mean distance of i to all points in any other cluster, 
        		# of which i is not a member
        		b_i = np.inf
        		for k in range(1, n_clusts+1):
        			if k == y[i]:
        				continue
        			else:
        				b_k = np.mean(dists[y == k][:,i])
        			b_i = min(b_k, b_i)
        		scores[i] = (b_i - a_i)/max(a_i,b_i)
        
        return(scores)
    
    		
    	
    	


