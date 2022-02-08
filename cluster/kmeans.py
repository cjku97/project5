import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error

class KMeans:
    def __init__(
            self,
            k: int,
            metric: str = "euclidean",
            tol: float = 1e-6,
            max_iter: int = 100):
        """
        inputs:
            k: int
                the number of centroids to use in cluster fitting
            metric: str
                the name of the distance metric to use
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        if k <= 0:
        	raise ValueError("k must be at least 1")
        self.k = k
        self.metric = metric
        self.tol = tol
        self.max_iter = max_iter
        self.mat = [[]]
        self.n_obs = 0
        self.n_feats = 0
        self.clusters = []
        self.centroids = [[]]
        self.old_centroids = [[]]
        self.cluster_dict = {}
        self.dists = [[]]
    
    def fit(self, mat: np.ndarray):
        """
        fits the kmeans algorithm onto a provided 2D matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        self.mat = mat
        # number of observations = number of rows
        self.n_obs = len(mat)
        print("Observations: " + str(self.n_obs))
        # number of features = number of columns
        self.n_feats = len(mat[0])
        print("Features: " + str(self.n_feats))
        
        # initial parameter check
        if self.n_obs < self.k:
        	raise ValueError("You cannot have more clusters than observations")
        if self.n_obs < 1:
        	raise ValueError("You must have at least one observation")
        if self.n_feats < 1:
        	raise ValueError("You must have at least one feature")
        
        # randomly assign observations an initial cluster from 1 to k
        self.clusters = np.random.randint(low = 1, high = self.k+1, size = self.n_obs)
        self.get_clusters()

        # get centroids for random assignments
        self.centroids = self.get_centroids()
        print("RANDOM CENTROIDS")
        print(self.centroids)
        
        # initialize iteration count and error
        n_iter = 0
        error = np.inf
                        
        print("FITTING OBSERVATIONS TO " + str(self.k) + " CLUSTERS")
        # fit model
        while error > self.tol and n_iter < self.max_iter:
        	print("iteration: " + str(n_iter + 1))
        	# 1. find the distance of each point to each centroid
        	self.dists = cdist(self.mat, self.centroids, metric = self.metric) 
        	# 2. reassign observations to nearest centroid and update cluster dictionary
        	self.clusters = self.predict(self.mat)
        	self.get_clusters()
        	# 3. find the centroids of each new cluster
        	self.old_centroids = self.centroids
        	self.centroids = self.get_centroids()
        	print(self.centroids)
        	# 4. update error of model (measure of cluster stability)
        	error = self.get_error(self.old_centroids, self.centroids)
        	print("Error: " + str(error))
        	# 5. increase iteration counter
        	n_iter = n_iter + 1 
        	
        	

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        predicts the cluster labels for a provided 2D matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        print("GETTING CLUSTERS")
        new_clusters = np.zeros(self.n_obs)
        for i in range(0, self.n_obs):
        	min_dist = min(self.dists[i])
        	new_clusters[i] = np.where(self.dists[i] == min_dist)[0] + 1
        return(new_clusters)

    def get_error(self, old, new) -> float:
        """
        returns the final mean-squared error of the fit model

        outputs:
            float
                the mean-squared error of the fit model
        """
        return(mean_squared_error(old, new))

    def get_centroids(self) -> np.ndarray:
        """
        returns the centroid locations of the fit model

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        print("GETTING CENTROIDS")
        cent_mat = np.zeros(shape = (self.k,self.n_feats))
        for c in range(1,self.k+1):
        	s = self.cluster_dict[c]
        	s_sum = np.zeros(self.n_feats)
        	for i in s:
        		s_sum = s_sum + self.mat[i]
        	cent_mat[c-1] = (1/len(s)) * s_sum
        return(cent_mat)
        	
    
    def get_clusters(self):
    	for c in range(1, self.k+1):
        	self.cluster_dict[c] = []
    	for i in range(0, self.n_obs):
    		c = self.clusters[i]
    		self.cluster_dict[c] = self.cluster_dict[c] + [i]
        	
