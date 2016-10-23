
import numpy as np


def fnmtf(X, k, l, num_iter=100, norm=False, orthogonal_strategy=False):
    m, n = X.shape

    U = np.random.rand(m, k)
    S = np.random.rand(k, l)
    V = np.random.rand(n, l)

    error_best = np.inf
    error = error_best

    if norm:
        X = Normalizer().fit_transform(X)
    for _ in xrange(num_iter):
        S = np.linalg.pinv(U.T.dot(U)).dot(U.T).dot(X).dot(V).dot(np.linalg.pinv(V.T.dot(V)))

        # solve subproblem to update V
        U_tilde = U.dot(S)
        V_new = np.zeros(n*l).reshape(n, l)
        for j in xrange(n):
            errors = np.zeros(l)
            for col_clust_ind in xrange(l):
                errors[col_clust_ind] = ((X[:][:, j] - U_tilde[:][:, col_clust_ind])**2).sum()
            ind = np.argmin(errors)
            V_new[j][ind] = 1
        V = V_new

        if orthogonal_strategy:
            while np.linalg.det(V.T.dot(V)) <= 0:
                if np.isnan(np.sum(V)):
                    break

                erros = (X - U.dot(S).dot(V.T)) ** 2
                erros = np.sum(erros.dot(V), axis=0) / np.sum(V, axis=0)
                erros[np.where(np.sum(V, axis=0) <= 1)] = -np.inf
                quantidade = np.sum(V, axis=0)
                indexMin = np.argmin(quantidade)
                indexMax = np.argmax(erros)
                indexes = np.nonzero(V[:, indexMax])[0]
                end = len(indexes)
                indexes_p = np.random.permutation(end)
                V[indexes[indexes_p[0:np.floor(end/2.0)]], indexMax] = 0.0
                V[indexes[indexes_p[0:np.floor(end/2.0)]], indexMin] = 1.0

        # solve subproblem to update U
        V_tilde = S.dot(V.T)
        U_new = np.zeros(m*k).reshape(m, k)
        for i in xrange(m):
            errors = np.zeros(k)
            for row_clust_ind in xrange(k):
                errors[row_clust_ind] = ((X[i][:] - V_tilde[row_clust_ind][:])**2).sum()
            ind = np.argmin(errors)
            U_new[i][ind] = 1
        U = U_new

        if orthogonal_strategy:
            while np.linalg.det(U.T.dot(U)) <= 0:
                if np.isnan( np.sum(U) ):
                    break

                erros = (X - U.dot(V_tilde)) ** 2
                erros = np.sum(U.T.dot(erros), axis=1) / np.sum(U, axis=0)
                erros[np.where(np.sum(U, axis=0) <= 1)] = -np.inf
                quantidade = np.sum(U, axis=0)
                indexMin = np.argmin(quantidade)
                indexMax = np.argmax(erros)
                indexes = np.nonzero(U[:, indexMax])[0]

                end = len(indexes)
                indexes_p = np.random.permutation(end)
                U[indexes[indexes_p[0:np.floor(end/2.0)]], indexMax] = 0.0
                U[indexes[indexes_p[0:np.floor(end/2.0)]], indexMin] = 1.0

        error_ant = error
        error = np.sum((X - U.dot(S).dot(V.T)) ** 2)

        if error < error_best:
            U_best = U
            S_best = S
            V_best = V
            error_best = error

        if np.abs(error - error_ant) <= 0.000001:
            break

    rows_ind = np.argmax(U, axis=1)
    cols_ind = np.argmax(V, axis=1)

    return U, S, V, rows_ind, cols_ind, error
