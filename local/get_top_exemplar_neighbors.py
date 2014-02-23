import numpy as np
from acoustic_cues.ambiguity import get_top_exemplar_neighbors

class Jeery():
    pass

args = Jeery()
args.data_prefix='/home/mark/Research/phoneclassification/exp/parts_pegasos/'
args.data_suffix='bsparse.npy'

X_indices = np.load('%sX_indices_%s' % (args.data_prefix,args.data_suffix))
rownnz = np.load('%sX_rownnz_%s' % (args.data_prefix,args.data_suffix))
rowstartidx = np.load('%sX_rowstartidx_%s' % (args.data_prefix,args.data_suffix))
y = np.load('%sy_%s' % (args.data_prefix,args.data_suffix))
dim = np.load('%sdim_%s' % (args.data_prefix,args.data_suffix))

args.leehon='/home/mark/Research/acoustic_cues/conf/phones.48-39'

leehon = np.loadtxt(args.leehon,dtype=str)

cov = np.load('/home/mark/Research/acoustic_cues/exp/discriminative_parts/general_covariance_cov_9T9F_parts.npy')
mean = np.load('/home/mark/Research/acoustic_cues/exp/discriminative_parts/general_covariance_feature_counts_all_9T9F_parts.npy')


n_times, n_freqs, n_parts =dim
cov = cov[:n_parts,:n_parts]
M = np.ascontiguousarray(np.dot(np.linalg.inv( .01*np.eye(n_parts) + np.dot(cov,cov)),cov))
Mmean = np.dot(M,mean)
bgd_mean = np.ascontiguousarray(np.zeros(dim))
for t in xrange(n_times):
    for f in xrange(n_freqs):
        bgd_mean[t,f] = Mmean.copy()

D = n_times*n_freqs*n_parts
n_neighbors= 50
neighbor_indices, neighbor_scores=get_top_exemplar_neighbors(
    X_indices,
    rownnz,
    rowstartidx,
    M.ravel(),    
    n_parts, 
    D,
    bgd_mean.ravel(),
    n_neighbors)
