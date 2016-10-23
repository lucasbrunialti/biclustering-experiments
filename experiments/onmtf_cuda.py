
import numpy as np
import cudamat as cm


def matrix_factorization_clustering(X_aux, k, l, norm=False, num_iters=100):
    cm.cublas_init()

    m, n = X_aux.shape
    U = cm.CUDAMatrix(np.random.rand(m, k))
    S = cm.CUDAMatrix(np.random.rand(k, l))
    V = cm.CUDAMatrix(np.random.rand(n, l))

    X = cm.CUDAMatrix(X_aux)

    # if norm:
    #     X = Normalizer().fit_transform(X)

    XV = cm.CUDAMatrix(np.random.rand(m, l))
    XVSt = cm.CUDAMatrix(np.random.rand(m, k))
    US = cm.CUDAMatrix(np.random.rand(m, l))
    USVt = cm.CUDAMatrix(np.random.rand(m, n))
    USVtXt = cm.CUDAMatrix(np.random.rand(m, m))
    USVtXtU = cm.CUDAMatrix(np.random.rand(m, k))
    U_aux = cm.CUDAMatrix(np.random.rand(m, k))

    XtUS = cm.CUDAMatrix(np.random.rand(m, l))
    VSt = cm.CUDAMatrix(np.random.rand(n, k))
    VStUt = cm.CUDAMatrix(np.random.rand(n, m))
    UtX = cm.CUDAMatrix(np.random.rand(k, n))
    VStUtXV = cm.CUDAMatrix(np.random.rand(n, l))
    V_aux = cm.CUDAMatrix(np.random.rand(n, l))

    UtXV = cm.CUDAMatrix(np.random.rand(k, l))
    UtUS = cm.CUDAMatrix(np.random.rand(k, l))
    UtUSVt = cm.CUDAMatrix(np.random.rand(k, n))
    UtUSVtV = cm.CUDAMatrix(np.random.rand(k, l))
    S_aux = cm.CUDAMatrix(np.random.rand(k, l))

    error_best = np.inf
    error = np.inf

    for i in range(num_iters):
        # compute U
        cm.dot(X, V, target=XV)
        cm.dot(XV, S.T, target=XVSt)

        if i is 0:
            cm.dot(U, S, target=US)
            cm.dot(US, V.T, target=USVt)
        cm.dot(USVt, X.T, target=USVtXt)
        cm.dot(USVtXt, U, target=USVtXtU)

        cm.divide(XVSt, USVtXtU, U_aux)
        cm.mult(U, U_aux, U)

        # compute V
        cm.dot(U, S, target=US)
        cm.dot(X.T, US, target=XtUS)
        cm.dot(V, S.T, target=VSt)
        cm.dot(VSt, U.T, target=VStUt)
        cm.dot(VStUt, XV, target=VStUtXV)

        cm.divide(XtUS, VStUtXV, target=V_aux)
        cm.mult(V, V_aux, V)

        # compute S
        cm.dot(U.T, X, target=UtX)
        cm.dot(UtX, V, target=UtXV)

        cm.dot(U.T, US, target=UtUS)
        cm.dot(UtUS, V.T, UtUSVt)
        cm.dot(UtUSVt, V, target=UtUSVtV)

        cm.divide(UtXV, UtUSVtV, target=S_aux)
        cm.mult(S, S_aux, target=S)

        error_ant = error

        cm.dot(U, S, target=US)
        cm.dot(US, V.T, target=USVt)
        error = cm.sum(cm.pow(cm.subtract(X, USVt), 2), axis=0)

        if error < error_best:
            U_best_cm = U
            S_best_cm = S
            V_best_cm = V
            error_best = error

        if np.abs(error - error_ant) <= 0.000001:
            break

        U_best = U_best_cm.asarray()
        S_best = S_best_cm.asarray()
        V_best = V_best_cm.asarray()

    Du = np.diag(np.ones(m).dot(U_best))
    Dv = np.diag(np.ones(n).dot(V_best))

    U_norm = U_best.dot( np.diag(S_best.dot(Dv).dot(np.ones(l))) )
    V_norm = V_best.dot( np.diag(np.ones(k).dot(Du).dot(S_best)) )

    rows_ind = np.argmax(U_best, axis=1)
    cols_ind = np.argmax(V_best, axis=1)

    cm.shutdown()

    return U_norm, S_best, V_norm, rows_ind, cols_ind, error_best
