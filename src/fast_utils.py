import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
#from dpmatchsrvf import dpmatch
#from dpsrvf.match_utils import match

# reformat for fury display                       
def tract_reformat(tract):                        
    new_tract = tract.transpose((0,2,1))
    tract_list = []                               
    for element in new_tract:                     
        tract_list += [element]                   
                                                                          
    return tract_list                             


def inner_product_L2(u,v):
    '''
    Computes the standard inner product on L2
    Input:
    - u: A (dimension: n, number_of_points) matrix representation of function u: D --> R^n
    - v: A (dimension: n, number_of_points) matrix representation of function v: D --> R^n
    
    Outputs:
    <u,u> = int_[0,1] (u(t), v(t))_R^n dt
    '''

    _, number_of_points = u.shape

    return np.trapz(np.sum(np.multiply(u,v), axis = 0), dx = 1/(number_of_points -1 )) # changed

def induced_norm_L2(u):
    '''
    Computes the norm induced by the standard L2 inner product
    Inputs:
    - u: An (dimension, number_of_points) matrix representation of the function u: D --> R^n

    Outputs:
    - ||u|| = sqrt(<u,u>) = sqrt(int_[0,1] (u(t), u(t))_R^ndt)
    '''

    return np.sqrt(inner_product_L2(u,u))

def project_unit_ball(srvf):
    '''
    Projects a srvf to a point on the unit ball in L^2
    Inputs:
    - srvf: A (dimension, number_of_points) matrix representation of the srvf function of 
    f : D --> R^n
    
    Outputs:
    An (dimension, number_of_points) matrix representation of srvf projected on the Hilbert
    Sphere
    '''

    induced_norm = induced_norm_L2(srvf)
    return srvf/induced_norm

def curve_to_srvf(curve):
    '''
    Given a curve f, we get the srvf representation
    Inputs:
    - f An (dimnension, number_of_points) matrix representation of the function f: D --> R^n

    Outputs:
    - An (dimension, number_of_shapes) matrix representation of the srvf of f
    '''

    dimension, number_of_points = curve.shape

    # Taking the derivative of curve
    beta_dot = np.zeros((dimension, number_of_points))
    for i in range(dimension):
        beta_dot[i,:] = np.gradient(curve[i,:], 1/(number_of_points -1))

    # Initializing srvf and dividing by the norm of its derivative
    srvf = np.zeros((dimension, number_of_points))
    eps = np.finfo(float).eps
    for i in range(number_of_points):
        srvf[:,i] = beta_dot[:,i]/(np.sqrt(np.linalg.norm(beta_dot[:,i])) + eps)


    srvf = project_unit_ball(srvf)

    return srvf

def srvf_to_curve(srvf):
    '''
    Given an srvf, recovers original curve. Note that translation, and scale is lost.
    Inputs:
    - srvf: ans (dimension, number_of_points) matrix representation of srvf: D --> R^n

    Outputs:
    - An (dimension, number_of_points) matrix representation of the original curve
    '''

    dimension, number_of_points = srvf.shape

    srvf_norms = np.linalg.norm(srvf, axis = 0)
    curve = np.zeros((dimension, number_of_points))
    for i in range(dimension):
        curve[i,:] = cumtrapz(np.multiply(srvf[i,:], srvf_norms), dx = 1/(number_of_points - 1), initial = 0)

    return curve

def batch_curve_to_srvf(curves):
    '''
    Given a collection of curves, gets their srvf representation. Assumes that all matrix
    representations of the curves are of the same size.
    Input:
    - curves: A (number_of_curves, dimension, number_of_points) array of curves

    Outputs:
    - A (number_of_curves, dimension, number_of_points) list of srvf representations of the
    curves
    '''
    
    return np.array([curve_to_srvf(curve) for curve in curves])

def batch_srvf_to_curve(srvfs):
    '''
    Given a collection of curves, gets their srvf representation. Assumes that all matrix
    representations of the curves are of the same size.
    Input:
    - curves: A (number_of_curves, dimension, number_of_points) array of curves

    Outputs:
    - A (number_of_curves, dimension, number_of_points) list of srvf representations of the
    curves
    '''
    
    return np.array([srvf_to_curve(srvf) for srvf in srvfs])

def find_best_rotation(srvf1, srvf2):
    '''
    Solves the Procrusted problem to find optimal rotation
    Inputs:
    - srvf1: An (dimension, number_of_points) matrix
    - srvf2: An (dimension, number_of_points) matrix

    Outputs:
    - srvf2n: An (dimension, number_of_points) matrix representing the rotated srvf2
    - R: An (dimension, dimension) matrix representing the rotation matrix
    '''
    dimension, number_of_points = srvf1.shape
    A = np.matmul(srvf1, srvf2.T)
    [U, S, V] = np.linalg.svd(A)
    V = V.T

    S = np.eye(dimension)
    if (np.abs(np.linalg.det(U)*np.linalg.det(V) - 1) > 10*np.spacing(1)):
        S[:,-1] = -S[:,-1]

    R = np.matmul(U, np.matmul(S, V.T))
    srvf2n = np.matmul(R, srvf2)

    return srvf2n, R



def curve_length(X):

    pgrad = np.gradient(X, axis=1)
    arc_length = np.linalg.norm(pgrad, axis=0)
    return np.sum(arc_length)

def est_pose(X):

    n, T = X.shape
    Y = X
    L = curve_length(Y)
    POS = np.mean(Y, axis = 1)
    Y = Y - POS[:, np.newaxis]
    Y /= L

    XYZaxis = np.tile(np.linspace(0, 1, T, True), (n, 1))
    POSaxis = np.mean(XYZaxis, axis=1)
    XYZaxis = XYZaxis - POSaxis[:, np.newaxis]
    L1 = curve_length(XYZaxis)
    XYZaxis = XYZaxis / L1

    return L, POS

def est_repose(X, L, POS):
    n, T = X.shape
    Y = X
    curPOS = np.mean(Y, axis=1)
    Y = Y/curve_length(Y)

    # zero-out the current position is
    Y = Y - curPOS[:, np.newaxis]

    # Scale
    Xnew = Y
    Xnew = (Xnew*L)/(curve_length(Xnew))

    # Translation
    Xnew = Xnew + POS[:, np.newaxis]

    return Xnew

def group_action_by_gamma(q, gamma):
    '''
    Computes composition of q and gamma and normalizes by gradient
    Inputs:
    -q: An (n,T) matrix
    -gamma: A (T,) dimensional vector representing the warp to apply to q
    '''
    n, T = q.shape
    gamma_t = np.gradient(gamma, 1/(T-1))
    f = interp1d(np.linspace(0, 1, T, True), q, kind = 'linear', fill_value = 'extrapolate')
    q_composed_gamma = f(gamma)

    sqrt_gamma_t = np.tile(np.sqrt(gamma_t), (n,1))
    qn = np.multiply(q_composed_gamma, sqrt_gamma_t)

    return qn

def gram_schmidt(X):
    epsilon = 5e-6
    N, n, T = np.shape(X)
    #N = T

    i = 0
    r = 0
    Y = np.zeros_like(X)
    Y[0] = X[0]

    while (i < N):
        temp_vec = 0
        for j in range(i):
            temp_vec += inner_product_L2(Y[j], X[r])*Y[j]
        Y[i] = X[r] - temp_vec
        temp = inner_product_L2(Y[i], Y[i])
        if (temp > epsilon):
            Y[i] /= np.sqrt(temp)
            i += 1
            r += 1
        else:
            if (r < i):
                r += 1
            else:
                break
    return Y


def project_tangent(f, q):
    w = f - inner_product_L2(f, q) * q
    return w


def parallel_transport(w, q1, q2):
    
    w_norm = induced_norm_L2(w)
    w_new = w

    if w_norm > 1e-4:
        w_new = project_tangent(w, q2)
        w_new = w_norm*w_new / induced_norm_L2(w_new)
    return w_new


def array_parallel_transport(alpha_t_arr, qmean1, qmean2):
    v_transport_array = np.zeros((alpha_t_arr.shape))

    for i, vec in enumerate(alpha_t_arr):
        v_new = parallel_transport(vec, qmean1, qmean2)
        v_transport_array[i] = v_new
    return v_transport_array

def form_basis_L2_R3(d, T):
    '''
    Returns basis for L_2(R^3)
    Note basis elements will be 6 x d
    '''

    x = np.linspace(0, 1, T, True)
    sqrt_2 = np.sqrt(2)
    constB = np.zeros((3,3,T))

    constB[0] = np.array([sqrt_2 * np.ones(T), np.zeros(T), np.zeros(T)])
    constB[1] = np.array([np.zeros(T), sqrt_2 * np.ones(T), np.zeros(T)])
    constB[2] = np.array([np.zeros(T), np.zeros(T), sqrt_2 * np.ones(T)])

    B = np.zeros((6*d, 3, T))
    k = 0
    for j in np.arange(1, d+1):
        B[0 + 6*k] = np.array([np.sqrt(2) * np.cos(2 * np.pi * j * x), np.zeros(T), np.zeros(T)])
        B[1 + 6*k] = np.array([np.zeros(T), np.sqrt(2) * np.cos(2 * np.pi * j * x), np.zeros(T)])
        B[2 + 6*k] = np.array([np.zeros(T), np.zeros(T), np.sqrt(2) * np.cos(2 * np.pi * j * x)])
        B[3 + 6*k] = np.array([np.sqrt(2) * np.sin(2 * np.pi * j * x), np.zeros(T), np.zeros(T)])
        B[4 + 6*k] = np.array([np.zeros(T), np.sqrt(2) * np.sin(2 * np.pi * j * x), np.zeros(T)])
        B[5 + 6*k] = np.array([np.zeros(T), np.zeros(T), np.sqrt(2) * np.sin(2 * np.pi * j * x)])
        k = k + 1

    B = np.concatenate((constB, B))

    return B

def form_basis_D(d, T):
    
    x = np.linspace(0,2*np.pi,T)
    xdarray = np.arange(1,d+1)
    xdarray = np.outer(xdarray,x)
    V_cos = np.cos(xdarray) / np.sqrt(np.pi)
    V_sin = np.sin(xdarray) / np.sqrt(np.pi)
    V = np.concatenate((V_cos, V_sin))
    x = np.reshape(np.linspace(0, 2*np.pi, T, True), (1,T))
    return V

def form_basis_O_q(B,q):
    d = len(B)
    n, T = q.shape

    # assumes dimension is n == 3
    R0 = np.array([[0,1,0], [-1,0,0], [0,0,0]])
    R1 = np.array([[0,0,1], [0,0,0], [-1,0,0]])
    R2 = np.array([[0,0,0], [0,0,1], [0,-1,0]])

    G = np.zeros((n,n,T))
    G[0] = R0 @ q
    G[1] = R1 @ q
    G[2] = R2 @ q

    # calculate derivatives of q
    qdiff = np.zeros(q.shape)
    for i in range(0, n):
        qdiff[i,:] = np.gradient(q[i,:], 2*np.pi/(T-1))
    
    # calculate the derivative of V
    V = form_basis_D(d,T)
    Vdiff = np.zeros(V.shape)
    for i in range(0,d):
        Vdiff[i,:] = np.gradient(V[i,:], 2*np.pi/(T-1))

    D_q = np.zeros((d,n,T))
    for i in range(0,d):
        tmp1 = np.tile(V[i,:], (n,1))
        tmp2 = np.tile(Vdiff[i,:], (n,1))
        D_q[i] = np.multiply(qdiff, tmp1) + (1/2)*np.multiply(q,tmp2)

    O_q = np.concatenate((G, D_q))

    return O_q


def form_basis_of_tangent_space_of_S_at_q(Bnew, G_O_q):
    '''
    T_q(S) = T_q + T_q(O_q)^{\perp}
    S is this case refers to the orbits of C^o
    S = {[q] | q in C^o
    Subtract the projection of the basis of T_q(C) onto T_q(O_q) from itself
    i.e. basis(T_q(C)) - <basis(T_q(C)), basis(T_q(O_q))> * basis(T_q(O_q))
    '''

    Gnew = Bnew.copy()
    for jj in np.arange(0, np.shape(Bnew)[0]):
        tmp = 0
        for kk in np.arange(0, np.shape(G_O_q)[0]):
            tmp += inner_product_L2(Bnew[jj], G_O_q[kk]) * G_O_q[kk]
        # tmp calculates projection of vectors in T_q(C) onto T_q(O_q)
        # by iteratively summing up over the projections along the
        # orthonormal basis of T_q(O_q)
        Gnew[jj] = Bnew[jj] - tmp

    return Gnew

def project_to_basis(alpha_t_array, Y):
    
    V = np.zeros(alpha_t_array.shape)
    A = np.zeros((alpha_t_array.shape[0], Y.shape[0]))
    d,n,T = Y.shape
    for ii in np.arange(0, alpha_t_array.shape[0]):
        V[ii] = np.zeros((n,T))
        for jj in np.arange(0, Y.shape[0]):
            A[ii, jj] = inner_product_L2(alpha_t_array[ii], Y[jj])
            V[ii] = V[ii] + A[ii, jj] * Y[jj]
    return A, V


def tpca_from_pre(qmean, tangent_vectors):
    
    epsilon = 0.0001
    N,n,T = tangent_vectors.shape
    d = 20

    B = form_basis_L2_R3(d,T)
    Bnew = form_basis_of_tangent_space_of_S_at_q(B, qmean)

    Bnew = gram_schmidt(Bnew)
    G = Bnew

    Aproj, A = project_to_basis(tangent_vectors, G)
    C = np.cov(Aproj.T)
    U, S, V = np.linalg.svd(C)

    sDiag = np.diag(S)
    tmp = np.identity(len(S))
    tmp = epsilon*tmp
    Cn = U*(tmp+sDiag)*U.T

    Eigproj = np.dot(Aproj, U)
    Y = gram_schmidt(tangent_vectors)

    return Aproj, A, G, Eigproj, U, S, V

def geodesic_flow(q1, w, stp):
    '''
    w represents the mean geodesic

    Output: qt (n x T), alpha: (stp+1, n, T)
    '''
    n, T = q1.shape
    qt = q1
    w_norm = induced_norm_L2(w)
    alpha = []
    alpha.append(q1)

    if w_norm < 1e-3:
        return qt, alpha
    
    for i in range(stp):
        qt = project_unit_ball(qt + w/stp)
        alpha.append(qt)
        w = project_tangent(w, qt)
        w = w_norm*w/induced_norm_L2(w)
    return qt, alpha

def geodesic_sphere(x_init, g, dt):
    g_norm = induced_norm_L2(g)
    X = np.cos(dt*g_norm)*x_init + np.sin(dt*g_norm)*g/g_norm
    return X

def dAlpha_dt(alpha):
    k,n,T = alpha.shape
    stp = k-1
    alpha_t = np.zeros_like(alpha)
    for tau in np.arange(1, k):
        alpha_t[tau] = stp*(alpha[tau] - alpha[tau-1])
        alpha_t[tau] = project_tangent(alpha_t[tau], alpha[tau])

    return alpha_t

def compute_geodesic(q1, q2, stp, d, dt):
    theta = np.arccos(inner_product_L2(q1,q2))
    f = (q2 - (inner_product_L2(q1,q2))*(q1))
    f = theta * f/induced_norm_L2(f)

    alpha = np.array([project_unit_ball(geodesic_sphere(q1, f, tau/stp)) for tau in range(stp+1)])
    alpha_t = dAlpha_dt(alpha)

    return alpha_t
    

def geodesic_distance_all(qarr, stp = 7, d = 5):
    '''
    '''
    stp = 7
    dt = 0.1
    d = 5

    num_shapes, _, _ = qarr.shape
    alpha_t_arr = []

    for i in range(num_shapes):
        q1 = qarr[i]
        for j in range(i+1, num_shapes):
            q2 = qarr[j]
            alpha_t = compute_geodesic(q1, q2, stp, d, dt)
            alpha_t_arr.append(alpha_t)

    return alpha_t_arr 
    


def karcher_mean(qarr, num_itr = 1, stp = 7, d = 5, dt =0.1):
    N, n, T = qarr.shape

    #stp = 6
    #dt = 0.1
    #d = 5 # number of Fourier coefficients divided by 2

    # Initialize mean to extrinsic average
    # Estimate the intrinsic average
    qmean = np.mean(qarr, axis = 0)
    qmean = project_unit_ball(qmean)
    qmean_and_shapes = np.zeros((2, n, T))

    for itr in range(num_itr):
        alpha_t_mean = np.zeros((n,T))
        qmean_and_shapes[0] = qmean

        for i in range(N):
            qmean_and_shapes[1] = qarr[i]
            alpha_t_arr_i = geodesic_distance_all(qmean_and_shapes, stp, d)
            alpha_t_mean += alpha_t_arr_i[0][1]

        alpha_t_mean /= N
        qmean, _ = geodesic_flow(qmean, alpha_t_mean, stp)

    qmean_and_shapes[0] = qmean
    alpha_t_arr = np.zeros((N,n,T))
    for i in range(N):
        qmean_and_shapes[1] = qarr[i]
        alpha_t_arr_i = geodesic_distance_all(qmean_and_shapes)
        alpha_t_arr[i] = alpha_t_arr_i[0][1]

    return qmean, alpha_t_arr
