
import numpy as np


def onmtf(X, U, S, V):
    U *= np.true_divide(X.dot(V).dot(S.T), U.dot(S).dot(V.T).dot(X.T).dot(U))
    V *= np.true_divide(X.T.dot(U).dot(S), V.dot(S.T).dot(U.T).dot(X).dot(V))
    S *= np.true_divide(U.T.dot(X).dot(V), U.T.dot(U).dot(S).dot(V.T).dot(V))
    return U, S, V


def onm3f(X, U, S, V):
    U = U * (X.dot(V).dot(S.T)) / np.sqrt(U.dot(U.T).dot(X).dot(V).dot(S.T))
    V = V * (X.T.dot(U).dot(S)) / np.sqrt(V.dot(V.T).dot(X.T).dot(U).dot(S))
    S = S * (U.T.dot(X).dot(V)) / np.sqrt(U.T.dot(U).dot(S).dot(V.T).dot(V))
    return U, S, V


def nbvd(X, U, S, V):
    U = U * (X.dot(V).dot(S.T)) / U.dot(U.T).dot(X).dot(V).dot(S.T)
    V = V * (X.T.dot(U).dot(S)) / V.dot(V.T).dot(X.T).dot(U).dot(S)
    S = S * (U.T.dot(X).dot(V)) / U.T.dot(U).dot(S).dot(V.T).dot(V)
    return U, S, V


def matrix_factorization_clustering(X, k, l, factorization_func=onmtf, norm=False, num_iters=100):
    m, n = X.shape
    U = np.random.rand(m, k)
    S = np.random.rand(k, l)
    V = np.random.rand(n, l)

    # if norm:
    #     X = Normalizer().fit_transform(X)

    XV = np.random.rand(m, l)
    XVSt = np.random.rand(m, k)
    US = np.random.rand(m, l)
    USVt = np.random.rand(m, n)
    USVtXt = np.random.rand(m, m)
    USVtXtU = np.random.rand(m, k)

    XtUS = np.random.rand(n, l)
    VSt = np.random.rand(n, k)
    VStUt = np.random.rand(n, m)
    UtX = np.random.rand(k, n)
    VStUtXV = np.random.rand(n, l)

    UtXV = np.random.rand(k, l)
    UtUS = np.random.rand(k, l)
    UtUSVt = np.random.rand(k, n)
    UtUSVtV = np.random.rand(k, l)

    error_best = np.inf
    error = np.inf

    for i in range(num_iters):
        # compute U
        XV = np.dot(X, V)
        XVSt = np.dot(XV, S.T)

        if i is 0:
            US = np.dot(U, S)
            USVt = np.dot(US, V.T)
        USVtXt = np.dot(USVt, X.T)
        USVtXtU = np.dot(USVtXt, U)

        U *= np.true_divide(XVSt, USVtXtU)

        # compute V
        US = np.dot(U, S)
        XtUS = np.dot(X.T, US)
        VSt = np.dot(V, S.T)
        VStUt = np.dot(VSt, U.T)
        VStUtXV = np.dot(VStUt, XV)

        V *= np.true_divide(XtUS, VStUtXV)

        # compute S
        UtX = np.dot(U.T, X)
        UtXV = np.dot(UtX, V)

        UtUS = np.dot(U.T, US)
        UtUSVt = np.dot(UtUS, V.T)
        UtUSVtV = np.dot(UtUSVt, V)

        S *= np.true_divide(UtXV, UtUSVtV)

        error_ant = error

        US = np.dot(U, S)
        USVt = np.dot(US, V.T)

        error_ant = error
        error = np.sum((X - USVt) ** 2)

        print error

        if error < error_best:
            U_best = U
            S_best = S
            V_best = V
            error_best = error

        if np.abs(error - error_ant) <= 0.000001:
            break

    Du = np.diag(np.ones(m).dot(U_best))
    Dv = np.diag(np.ones(n).dot(V_best))

    U_norm = U_best.dot( np.diag(S_best.dot(Dv).dot(np.ones(l))) )
    V_norm = V_best.dot( np.diag(np.ones(k).dot(Du).dot(S_best)) )

    rows_ind = np.argmax(U_best, axis=1)
    cols_ind = np.argmax(V_best, axis=1)

    return U_norm, S_best, V_norm, rows_ind, cols_ind, error_best
