import numpy as np
from fast_utils import *
from fury import window, actor
from dipy.segment.bundles import bundle_shape_similarity
from dipy.align.streamlinear import StreamlineLinearRegistration
import scipy 

# Synthetic Bundles
tract1 = np.load('synthetic_tract_5_target.npy')
tract2 = np.load('synthetic_tract_5_moving.npy')

tract1 = tract1.transpose((2,0,1))
tract1 = tract1[:1000]

tract2 = tract2.transpose((2,0,1))
tract2 = tract2[:1000]

# Record the coordinates
coords1 = []
for curve in tract1:
    L, POS = est_pose(curve)
    coords1 += [[L, POS]]
# convert to srvf space
srvf_tract1 = np.zeros((tract1.shape))
for index, curve in enumerate(tract1):
    srvf_tract1[index] = curve_to_srvf(curve)

# compute the Karther mean and the tangent vectors
qmean, alpha_t_arr = karcher_mean(srvf_tract1, 15, 7, 5)

# Record the coordinates
coords2 = []
for curve in tract2:
    L, POS = est_pose(curve)
    coords2 += [[L, POS]]
# convert to srvf space
srvf_tract2 = np.zeros((tract2.shape))
for index, curve in enumerate(tract2):
    srvf_tract2[index] = curve_to_srvf(curve)

# compute the Karther mean and the tangent vectors
qmean2, alpha_t_arr2 = karcher_mean(srvf_tract2, 15, 7, 5)

# Compute the coefficient matrix Aproj
Aproj, A, G, Eigproj, U, S, V = tpca_from_pre(qmean, alpha_t_arr)

# Parallel transport vectors between mean geodesic between qmeans
alpha_t_arr2_transport = array_parallel_transport(alpha_t_arr2, qmean2, qmean)
Aproj2, A2, G2, Eigproj2, U2, S2, V2 = tpca_from_pre(qmean, alpha_t_arr2_transport)

# Perform rotational alignment in coefficient space
Aproj2, _ = find_best_rotation(Aproj, Aproj2)

# Aproj2 @ x = Aproj need to solve this
# Apply a linear transform in coefficient space
x = np.linalg.pinv(Aproj2) @ Aproj
Aproj2 = Aproj2 @ x

# 20 principal components (Max is 123)
di = 100 #123
recon = np.matmul(Aproj2, U[:,:di])
recon = np.matmul(recon, U[:,:di].T)
#print("recon:", recon.shape)

# Once we register shape features, we reconstruct the bundle, rescale and retranslate back to the original space in R^3
N, coeff = Aproj.shape
p_new = np.zeros((N,3,100))
for i, gp in enumerate(recon):
    recon_alpha_t_arr = np.zeros((3,100))
    for t in range(123): # iterating over Fourier coefficients
        recon_alpha_t_arr += gp[t] * G[t]

    q_recon, _ = geodesic_flow(qmean, recon_alpha_t_arr, stp = 100)
    p_final = srvf_to_curve(q_recon)
    p_final = est_repose(p_final, coords1[i][0], coords1[i][1])
    p_new[i] = p_final


# Formatting to work with Dipy Streamline utilities
p_new = tract_reformat(p_new)
tract1 = tract_reformat(tract1)
tract2 = tract_reformat(tract2)

# Calling Dipy to perform alignment
#srr = StreamlineLinearRegistration(x0="affine", num_threads = 4)
#srm = srr.optimize(static=tract1, moving=tract2)
#tract2_dipy = srm.transform(tract2)

# Compute Bundle Similarity Score From BUAN
rng = np.random.RandomState()
score_t1_t2 = bundle_shape_similarity(tract1, tract2, rng, [0], threshold = 6)
print("tract1 to tract2 score: ", score_t1_t2)
score_t1_tnew = bundle_shape_similarity(tract1, p_new, rng, [0], threshold = 6)
print("tract1 to tract2-parallel-transport score: ", score_t1_tnew)
#score_t1_tnew = bundle_shape_similarity(tract1, tract2_dipy, rng, [0], threshold = 6)
#print("tract1 to tract2-dipy  score: ", score_t1_tnew)

def plot_tract(tract, rotate = 90, color = None, width = 0.3, alpha = 1):
    t = tract
    line_actor = actor.streamtube(t, color, opacity = alpha, linewidth = width)
    line_actor.RotateX(-rotate)
    line_actor.RotateZ(rotate)
    scene.add(line_actor)

# repositioned tract curves                                                         
scene = window.Scene()
#scene.SetBackground(1., 1,1)
plot_tract(tract1, rotate = 90, color = window.colors.red, width = .5, alpha = .9)
plot_tract(tract2, rotate = 90, color = window.colors.green, width = .5, alpha = .9)
#plot_tract(tract2_dipy, rotate = 90, color = window.colors.orange, width = .5, alpha = 1)
plot_tract(p_new, rotate = 90, color = window.colors.blue, width = .5, alpha = 1)
window.show(scene)


