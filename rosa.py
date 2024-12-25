from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import numpy as np


def norm(x):
    """Calculate the Euclidean norm of a vector x."""

    return np.sqrt(np.dot(x, x))



class Rosa(BaseEstimator, RegressorMixin):
    """The Response Optimal Sequential Alternation (ROSA) algorithm as
    proposed by Liland, K. H., Næs, T., and Indahl, U. G. (2016),
    ROSA—a fast extension of par- tial least squares regression for
    multiblock data analy- sis, J. Chemometrics, doi: 10.1002/cem.2824.
    """
    def __init__(self, group_ids, n_components=None):

        self.group_ids = group_ids
        self.n_components = n_components

        self.num_blocks = None
        self.scores = None
        self.weights = None

    def fit(self, X, y):

        # Number of data points 
        N, _ = np.shape(X[0])

        # Number of variable blocks
        n_blocks = len(X)

        # Num variables per block
        pk = [x.shape[1] for x in X]
        
        block_idx = [np.arange(X[0].shape[1])]
        for m in range(1, n_blocks):
            block_idx.append(np.arange(X[m].shape[1]) + 1 + block_idx[m - 1][-1])
 
        # Count the number of times a block is active 
        count = np.zeros(n_blocks, dtype=int)
        
        # Order of active blocks 
        order = np.zeros(self.n_components)

        # Orthonormal scores
        T = np.zeros((N, self.n_components))
        
        # Regression coeffs
        q = np.zeros(self.n_components)

        # Orthonormal block−loadings and −weights
        Pb = []
        
        # Global weights
        W = np.zeros((sum(pk), self.n_components))
        Wb = [np.zeros((n, N)) for n in pk]
    
        # Competing scores and residuals
        r = np.zeros((N, n_blocks))
        t = np.zeros((N, n_blocks))
         
        for a in range(self.n_components):
            
            # Placeholder for loading weight candidates
            v = [float(np.nan)] * n_blocks

            for k in range(n_blocks):
                
                # Compute the loading weight candidates
                v[k] = X[k].T @ y
                # Modify the associated competing candidate scores
                t[:, k] = X[k] @ v[k]

            if a > 0:
                t -= T[:, :a-1] @ (T[:, :a-1].T @ t)
            
            for j in range(n_blocks):
                
                # Normalize scores 
                t[:, j] /= norm(t[:, j])
                # Compute residuals 
                r[:, j] = y - t[:, j] * (t[:, j].T * y)
            
            i = np.argmin(np.sum(r ** 2, axis=0))
            
            count[i] += 1 
            order[a] = i 
            
            # Selected score vector 
            T[:, a] = t[:, i]
            # Regression coefficient 
            q[a] = y @ T[:, a]
            # Update to the smallest residual 
            y = r[:, i]
   
            v[i] -= Wb[i][:, :count[i]] @ (Wb[i][:, :count[i]].T @ v[i])
            
            Wb[i][:, count[i]] = v[i] / norm(v[i])
    
            W[block_idx[i], a] = Wb[i][:, count[i]]
    
        # Postprocessing
        for k in range(n_blocks):
            Pb.append(X[k].T @ T)
        
        PtW = np.triu(np.concatenate(Pb, axis=0).T @ W)
        
        C = np.dot(W, np.linalg.inv(PtW))
        #C = np.linalg.solve(PtW.T, W.T).T
    
        # Regression coefficients 
        self.coef_ = np.cumsum(C * q, axis=1) 
        
        # Intercept 
        self.intercept_ = np.mean(y) - np.mean(np.concatenate(X, axis=1), axis=0) @ self.coef_
        
    def transform(self, X):
        
        return self.intercept_ + np.concatenate(X, axis=1) @ self.coef_ 


if __name__ == '__main__':

    # Demo run.
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True)

    group_idx = [0] * X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=0, test_size=0.2
    )
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    
    y_train = y_train - np.mean(y_train)
    
    Z_train = np.random.random(X_train_std.shape)
    Z_test = np.random.random(X_test_std.shape)

    group_idx = [0] * Z_train.shape[0] + [1] * X_train.shape[0] + [2] * X_train.shape[1]
    X_train_std = [Z_train, X_train_std, X_train_std]
    X_test_std = [Z_test, X_test_std, X_test_std]
    
    model = Rosa(group_ids=group_idx, n_components=4)
    model.fit(X_train_std, y_train)
    print(model.transform(X_train_std).shape)
    #print(score(y_test, model.predict(X_test))
