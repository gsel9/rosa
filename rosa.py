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
        Pb = np.zeros(n_blocks)
        
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
            Pb[k] = X[k].T @ T 

        PtW = np.dot(Pb.T, W)

        """

            for k in range(self.num_blocks):

                t[:, k] = t[:, k] / (norm(t[:, k]) + self._eps)

                r[:, k] = y - np.dot(t[:, k], np.dot(t[:, k].T, y))

            # Index of winning block = smallest residual.
            i = np.argmin(np.sum(r ** 2), axis=0)

            count[i] = count[i] + 1

            # Order of winning blocks.
            order[a] = i

            T[:, a] = t[:, i]

            q[a] = np.dot(y.T, T[:, a])

            # Update y to smallest residual.
            y = r[:, i]

            # Orthogonalise and normalise the winning weights.
            v[i] = v[i] - np.dot(Wb, np.dot(Wb.T, v[i]))

            Wb = v[i] /  (norm(v[i]) + self._eps)

            #W[a] = Wb

        Pb = []
        for k in range(self.num_blocks):

            Pb.append(np.dot(X.T, T))

            #beta.append(np.dot(W, np.linalg.inv(np.dot(P, W))))

        print(np.shape(Pb))

        PtW = np.triu(np.dot(Pb, W))

        print(np.shape(Pb))
        print(np.shape(W))
        print(np.shape(PtW))

        for j in range(self.num_components):
            W[k] / PtW * q

            #X: (569, 60)
            #Pb: (30, 4)
            #PtW: (30, 30)
            #W: (4, 30)
            #q: (4)

        beta.append(np.cumsum(W / PtW[k] * q, axis=1))

        beta = np.mean(y, axis=0) - np.dot(np.mean(X, axis=0), beta)

        print(np.shape(beta))

        return beta



        A = self.num_components

        #for Xi in X:
        #    _, _ = check_X_y(Xi, y)

        self._org_X, self._org_y = X, y

        nb = 2#len(X)
        n, _ = np.shape(X)

        pk = [X.shape[1]] * 2 #[np.shape(X[num])[1] for num in range(nb)]

        count = np.zeros(nb)
        order = np.zeros(A)
        T = np.zeros((n, A))
        q = np.zeros(A)

        Pb = [0] * nb

        Wb = [np.zeros((x, n)) for x in pk]
        W = np.zeros((sum(pk), A))

        X_cent, y_cent = X, y#self.centering(X, y)

        inds = [np.arange(var) for var in pk]

        for i in range(1, nb):
            inds[i] += int(np.sum(pk[:i]))

        v = [0] * nb
        t = np.zeros((n, nb))
        r = np.zeros((n, nb))

        for a in range(A):

            for k in range(nb):
                v[k] = np.dot(X_cent[k].T, y_cent)
                t[:, k] = np.dot(X_cent[k], v[k])

            if a > 0:
                t = t - T[:, :a].dot(np.dot(T[:, :a].T, t))

            for k in range(nb):
                t[:, k] = t[:, k] / np.linalg.norm(t[:, k])

                offset = np.dot(np.dot(t[:, k], t[:, k].T), y_cent)
                r[:, k] = y_cent - offset

            i = np.argmin(np.sum(r ** 2))
            count[i] += 1
            order[a] = i

            T[:, a] = t[:, i]
            q[a] = np.dot(np.transpose(y_cent), T[:, a])

            y_cent = r[:, i]

            weight = Wb[i][:, :int(count[i])]

            v[i] -= weight.dot(np.dot(weight.T, v[i]))
            normalized = v[i] / np.linalg.norm(v[i])
            Wb[i][:, int(count[i])] = normalized
            W[inds[i], a] = Wb[i][:, int(count[i])]

        ## Postprocessing
        for k in range(nb):
            Pb[k] = np.dot(X_cent[k].T, T)

        PtW = np.triu(np.dot((np.concatenate(Pb, axis=0)).T, W))
        """


    def update_scores(self, scores):
        pass

    def predict(self, X):
        pass



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

    #print(score(y_test, model.predict(X_test))
