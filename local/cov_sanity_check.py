import numpy as np
import argparse, itertools
from phoneclassification.phoneclassification.multicomponent_binary_sgd import sparse_dot
from acoustic_cues.patch_computations import patch_sums, summed_area_table


def get_tpr_fpr(z,y,true_class):
    sorted_classes = y[np.argsort(z)[::-1]]
    true_class_locs = np.where(sorted_classes == true_class)[0]
    tpr = np.arange(true_class_locs.shape[0],dtype=float)+1
    tpr /= len(true_class_locs)
    fpr = true_class_locs.astype(float)  -np.arange(true_class_locs.shape[0],dtype=float)
    return tpr, fpr
        


parser = argparse.ArgumentParser("""
Finding candidate locations of parts
""")
parser.add_argument('--W',type=str,help='path to classifier matrix')
parser.add_argument('--meta',type=str,help='path to the metadata for the classifier matrix')
parser.add_argument('--patch_side_length',type=int,default=13,help='side length for the extracted patch') #TODO extend to rectangular patches
parser.add_argument('--data_prefix',type=str,help='path from where we load the data')
parser.add_argument('--data_suffix',type=str,help='suffix for the saved binary data--ending indicates parameter settings usually')
parser.add_argument('--save_prefix',type=str,help='path to where we save the data')
parser.add_argument('--save_suffix',type=str,help='suffix for the saved patches data--ending indicates parameter settings usually')
parser.add_argument('--cov',type=str,help='path to where the general covariance matrix is saved')
parser.add_argument('--part_stats',type=str,help='path to where the stats matrix indicated the coordinates rank, energy fraction and other metadata for the hypothesized discriminative parts')
parser.add_argument('--mean',type=str,help='path to where the feature means are saved')
parser.add_argument('--dim',type=str,help='path to where the dimensions for the block are saved')
parser.add_argument('--leehon',type=str,help='path to the 48 to 39 mapping')
# parser.add_argument('--',type=,help='')



args = parser.parse_args()

X_indices = np.load('%sX_indices_%s' % (args.data_prefix,args.data_suffix))
rownnz = np.load('%sX_rownnz_%s' % (args.data_prefix,args.data_suffix))
rowstartidx = np.load('%sX_rowstartidx_%s' % (args.data_prefix,args.data_suffix))
y = np.load('%sy_%s' % (args.data_prefix,args.data_suffix))

leehon = np.loadtxt(args.leehon,dtype=str)
W = np.load(args.W)
meta = np.load(args.meta)
dim = tuple(np.load(args.dim).astype(int))
# things to store about patches:
# 1. coordinates - 4 numbers
# 2. rank - simply location within the list
# 3. energy fraction and absolute energy 2 numbers
# 4. metadata -- class id and component id 2 numbers

n_models = W.shape[0]
part_stats = np.load(args.part_stats)
cov = np.load(args.cov)
mean = np.load(args.mean)

n_times, n_freqs, n_parts = dim[:]

# find the most promising par
big_fraction_ids = part_stats[:,5].argsort()[::-1]
part_id = big_fraction_ids[0]
component_id = int(part_stats[part_id][8])
v = W[component_id,:-1].reshape( * (dim))
v_patch = np.zeros(v.shape)
v_patch[part_stats[part_id][0]:part_stats[part_id][2],
        part_stats[part_id][1]:part_stats[part_id][3]] = v[
            part_stats[part_id][0]:part_stats[part_id][2],
            part_stats[part_id][1]:part_stats[part_id][3]] - mean

v_patch = np.ascontiguousarray(v_patch)

z = np.zeros(len(rownnz))
sparse_dot(X_indices, rownnz,
             rowstartidx, v_patch.ravel(),z,len(rownnz))

tpr, fpr = get_tpr_fpr(z,y,12)

#strongest patch example
strong_example_inside_id = z[y==12].argmax()
strong_example_id = np.arange(len(y))[y==12][strong_example_inside_id]
# get the patch associated with that
x = np.zeros(np.prod(dim),dtype=float)
x[X_indices[rowstartidx[strong_example_id]:rowstartidx[strong_example_id+1]]] = 1.0
x = x.reshape(*dim)
x_patch = np.zeros(v.shape)
x_patch[part_stats[part_id][0]:part_stats[part_id][2],
        part_stats[part_id][1]:part_stats[part_id][3]] = x[
            part_stats[part_id][0]:part_stats[part_id][2],
            part_stats[part_id][1]:part_stats[part_id][3]] - mean
x_patch = np.ascontiguousarray(x_patch)

z2 = np.zeros(len(rownnz))
sparse_dot(X_indices, rownnz,
             rowstartidx, x_patch.ravel(),z2,len(rownnz))

tpr2, fpr2 = get_tpr_fpr(z2,y,12)

vllsvm_patch = np.zeros(v.shape)
mean_clip = np.clip(mean,.0001,1-.0001)
mean_kernel = np.log(mean_clip/(1-mean_clip))
vllsvm_patch[part_stats[part_id][0]:part_stats[part_id][2],
        part_stats[part_id][1]:part_stats[part_id][3]] = v[
            part_stats[part_id][0]:part_stats[part_id][2],
            part_stats[part_id][1]:part_stats[part_id][3]] - mean_kernel

vllsvm_patch = np.ascontiguousarray(vllsvm_patch)

zllsvm = np.zeros(len(rownnz))
sparse_dot(X_indices, rownnz,
             rowstartidx, vllsvm_patch.ravel(),zllsvm,len(rownnz))

tprllsvm, fprllsvm = get_tpr_fpr(zllsvm,y,12)


vll_patch = np.zeros(v.shape)
patch_kernel = np.clip(x[
            part_stats[part_id][0]:part_stats[part_id][2],
            part_stats[part_id][1]:part_stats[part_id][3]],.0001,1-.0001)
patch_kernel = np.log(patch_kernel/(1-patch_kernel))
mean_kernel = np.log(mean_clip/(1-mean_clip))
vll_patch[part_stats[part_id][0]:part_stats[part_id][2],
        part_stats[part_id][1]:part_stats[part_id][3]] = patch_kernel - mean_kernel

vll_patch = np.ascontiguousarray(vll_patch)

zll = np.zeros(len(rownnz))
sparse_dot(X_indices, rownnz,
             rowstartidx, vll_patch.ravel(),zll,len(rownnz))

tprll, fprll = get_tpr_fpr(zll,y,12)

# now we make use of the covariance matrix and ultimately
# we will produce a plot showing how each of these different 
# methods work

cov_sq = np.dot(cov,cov)
example_x_diff = (x[
            part_stats[part_id][0]:part_stats[part_id][2],
            part_stats[part_id][1]:part_stats[part_id][3]] - mean).ravel()
cov_example_x_diff = np.dot(cov,example_x_diff)
fld0p01 = np.linalg.solve(.01*np.eye(example_x_diff.shape[0]) + cov_sq,cov_example_x_diff)

fld0p01 = fld0p01.reshape(args.patch_side_length,args.patch_side_length,mean.shape[0])

x_patch = np.zeros(v.shape)
x_patch[part_stats[part_id][0]:part_stats[part_id][2],
        part_stats[part_id][1]:part_stats[part_id][3]] = fld0p01
x_patch = np.ascontiguousarray(x_patch)

z_fld0p01 = np.zeros(len(rownnz))
sparse_dot(X_indices, rownnz,
             rowstartidx, x_patch.ravel(),z_fld0p01,len(rownnz))

tpr_fld0p01, fpr_fld0p01 = get_tpr_fpr(z_fld0p01,y,12)


fld0p001 = np.linalg.solve(.001*np.eye(example_x_diff.shape[0]) + cov_sq,cov_example_x_diff)

fld0p001 = fld0p001.reshape(args.patch_side_length,args.patch_side_length,mean.shape[0])

x_patch = np.zeros(v.shape)
x_patch[part_stats[part_id][0]:part_stats[part_id][2],        part_stats[part_id][1]:part_stats[part_id][3]] = fld0p001
x_patch = np.ascontiguousarray(x_patch)

z_fld0p001 = np.zeros(len(rownnz))
sparse_dot(X_indices, rownnz, rowstartidx, x_patch.ravel(),z_fld0p001,len(rownnz))

tpr_fld0p001, fpr_fld0p001 = get_tpr_fpr(z_fld0p001,y,12)

fld0p0001 = np.linalg.solve(.0001*np.eye(example_x_diff.shape[0]) + cov_sq,cov_example_x_diff)

fld0p0001 = fld0p0001.reshape(args.patch_side_length,args.patch_side_length,mean.shape[0])

x_patch = np.zeros(v.shape)
x_patch[part_stats[part_id][0]:part_stats[part_id][2],        part_stats[part_id][1]:part_stats[part_id][3]] = fld0p0001
x_patch = np.ascontiguousarray(x_patch)

z_fld0p0001 = np.zeros(len(rownnz))
sparse_dot(X_indices, rownnz, rowstartidx, x_patch.ravel(),z_fld0p0001,len(rownnz))

tpr_fld0p0001, fpr_fld0p0001 = get_tpr_fpr(z_fld0p0001,y,12)


import pdb; pdb.set_trace()


