from sklearn.base import BaseEstimator

import numpy as np


def norm(x):
    """Calculate the Euclidean norm of a vector x."""

    return np.sqrt(np.dot(x, x))



class Rosa(BaseEstimator):
    """

    """
    def __init__(self, groups, num_components=None):

        self.groups = groups
        self.num_components = num_components

        self.num_blocks = None
        self.scores = None
        self.weights = None

    def fit(self, X, y):

        n, _ = np.shape(X)

        self.num_blocks = len(np.unique(self.groups))

        self.scores = np.zeros((n, self.num_components), dtype=float)
        self.weights = np.zeros(self.num_components, dtype=float)

        # Z-scoring.
        X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        v = []
        q = np.zeros(self.num_components, dtype=float)
        t = np.zeros((n, self.num_blocks), dtype=float)
        r = np.zeros((n, self.num_blocks), dtype=float)
        T = np.zeros((n, self.num_components), dtype=float)

        for a in range(self.num_components):

            for k in range(self.num_blocks):
                v.append(np.dot(X_std[:, self.groups == k].T, y))

                t[:, k] = np.dot(X_std[:, self.groups == k], v[k])

            if a > 0:
                t = t - np.dot(T[:, :a - 1], np.dot(T[:, :a - 1].T, t))

            for k in range(self.num_blocks):
                t[:, k] = t[:, k] / norm(t[:, k])

                r[:, k] = y - np.dot(t[:, k], np.dot(t[:, k].T, y))

            i = np.argmin(np.sum(r ** 2), axis=0)

            T[:, a] = t[:, i]

            q[a] = np.dot(y.T, T[:, a])

            y = r[:, i]

            v[i] = v[i] - np.dot(Wb[i], np.dot(Wb[i].T, v[i]))

            Wb[i] = v[i] / norm(v[i])


    def update_scores(self, taus):

        for j in range(self._num_blocks):

            taus[j] = taus[j] - np.dot(self.scores, np.dot(np.transpose(self.scores), taus[j]))

            taus[j] = taus[j] / norm(taus[j])

        return taus[j]


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

    model = Rosa(groups=group_idx, num_components=4)
    model.fit(X_train, y_train)
