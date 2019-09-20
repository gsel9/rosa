from sklearn.base import BaseEstimator

import numpy as np


def norm(x):
    """Calculate the Euclidean norm of a vector x."""

    return np.sqrt(np.dot(x, x))



class Rosa(BaseEstimator):
    """

    """
    def __init__(self, group_ids, num_components=None):

        self.group_ids = group_ids
        self.num_components = num_components

        self.num_blocks = None
        self.scores = None
        self.weights = None

        self._eps = 1e-10

    def fit(self, X, y):

        n, _ = np.shape(X)

        self.num_blocks = len(np.unique(self.group_ids))

        self.scores = np.zeros((n, self.num_components), dtype=float)
        self.weights = np.zeros(self.num_components, dtype=float)

        # Z-scoring.
        X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        order = np.zeros(self.num_components, dtype=int)
        count = np.zeros(self.num_blocks, dtype=int)

        v = []
        q = np.zeros(self.num_components, dtype=float)
        t = np.zeros((n, self.num_blocks), dtype=float)
        r = np.zeros((n, self.num_blocks), dtype=float)
        T = np.zeros((n, self.num_components), dtype=float)

        # Block loading weights.
        Wb = np.asarray(0)

        inds = []

        W = [0] * self.num_components
        Pb = []

        for a in range(self.num_components):

            # Compute the loading weight candidates.
            for k in range(self.num_blocks):
                v.append(np.dot(X_std[:, self.group_ids == k].T, y))

                t[:, k] = np.dot(X_std[:, self.group_ids == k], v[k])

            if a > 0:
                t = t - np.dot(T[:, :a - 1], np.dot(T[:, :a - 1].T, t))

            # Compute the residuals after regressing y onto each t.
            for k in range(self.num_blocks):
                t[:, k] = t[:, k] / (norm(t[:, k]) + self._eps)

                r[:, k] = y - np.dot(t[:, k], np.dot(t[:, k].T, y))

            # Index of winning block = smallest residual.
            i = np.argmin(np.sum(r ** 2), axis=0)

            count[i] = count[i] + 1

            # Order of winning blocks.
            order[a] = i

            # Winning score vector.
            T[:, a] = t[:, i]

            # Regression coefficient wrt. Ta.
            q[a] = np.dot(y.T, T[:, a])

            # Update y to smallest residual.
            y = r[:, i]

            # Orthogonalise and normalise the winning weights.
            v[i] = v[i] - np.dot(Wb, np.dot(Wb.T, v[i]))

            Wb = v[i] /  (norm(v[i]) + self._eps)

            W[a] = Wb

        Pb = []
        for k in range(self.num_blocks):

            Pb.append(np.dot(X[:, self.group_ids == k].T, T))

            #beta.append(np.dot(W, np.linalg.inv(np.dot(P, W))))

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
        """
        beta.append(np.cumsum(W / PtW[k] * q, axis=1))

        beta = np.mean(y, axis=0) - np.dot(np.mean(X, axis=0), beta)
        """
        print(np.shape(beta))

        return beta

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

    # Dummy blocks.
    group_idx = np.concatenate(([0] * X.shape[1], [1] * X.shape[1]))
    X = np.append(X, X, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=0, test_size=0.2
    )
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    model = Rosa(group_ids=group_idx, num_components=4)
    model.fit(X_train, y_train)

    #print(score(y_test, model.predict(X_test))
