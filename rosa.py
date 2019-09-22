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
    def __init__(self, group_ids, num_components=None):

        self.group_ids = group_ids
        self.num_components = num_components

        self.num_blocks = None
        self.scores = None
        self.weights = None

        self._eps = 1e-10

    def fit(self, X, y):

        A = self.num_components

        # Number of samples
        n, _ = np.shape(X[0])

        # Number of blocks.
        nb = len(X)

        # Num features per block.
        pk = [30, 30]

        p = sum(pk)

        count = np.zeros((nb), dtype=int)

        order = np.zeros(A)

        T = np.zeros((n, A))

        q = np.zeros(A)

        Pb = np.zeros((p, nb))

        W = np.zeros((p, A))

        # NB: Check this with MATLAB example.
        Wb = [np.zeros((x, n)) for x in pk]

        # Column indices of blocks.
        inds = np.arange(60)

        # Column indices per blocks.
        inds_b = [np.arange(30), np.arange(30, 60)]

        v = [0] * nb
        t = np.zeros((n, nb))
        r = np.zeros((n, nb))

        for a in range(A):

            for k in range(nb):
                v[k] = np.dot(X[k].T, y)

                t[:, k] = np.dot(X[k], v[k])

            if a > 0:
                t = t - np.dot(T[:, :a - 1], np.dot(T[:, :a - 1].T, t))

            for k in range(nb):

                t[:, k] = t[:, k] / (norm(t[:, k]) + 1e-15)

                i = np.argmin(np.sum(r ** 2), axis=0)

                count[i] += 1

                order[a] = i

                T[:, a] = t[:, i]

                q[a] = np.dot(y.T, T[:, a])

                y = r[:, i]

                """
                * omega_i = np.dot(np.zeros(p).T, np.dot(v[i].T, np.zeros().T)) (p-dims)
                * p = sum(pk)
                * W = [w_1, ..., w_(a-1)]
                * w_a = omega_i - W * (W.T * omega_i)
                * w_a != 0 only for the i-th index segment of length p_i.

                omega_i = np.zeros(p)
                omega_i[inds_b[k]] = v[i]

                # * W = [w_1, ..., w_(a-1)]
                # * v[i] != 0 only at i-th entry of pi entries.

                v[i] = omega_i - np.dot(W[:, :a], np.dot(W[:, :a].T, omega_i))

                # * W = [w_1, ..., w_(a-1), w_a]
                W[inds_b[i], a] = v[i][inds_b[i]] / norm(v[i][inds_b[i]] + 1e-15)

                """

                v[i] = v[i] - np.dot(Wb[i][:, :count[i]], np.dot(Wb[i][:, :count[i]].T, v[i]))

                Wb[i][:, count[i]] = v[i] / (norm(v[i]) + 1e-15)

                W[inds_b[i], a] = Wb[i][:, count[i]]

        # Postprocessing
        for k in range(nb):
            Pb[inds_b[k], k] = np.dot(X[k].T, T)

        #PtW = np.dot(Pb.T, W)


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

    # Dummy blocks.
    #group_idx = np.concatenate(([0] * X.shape[1], [1] * X.shape[1]))
    group_idx = [0] * X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=0, test_size=0.2
    )
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)


    model = Rosa(group_ids=group_idx, num_components=2)
    model.fit([X_train_std, X_train_std], y_train)

    #print(score(y_test, model.predict(X_test))
