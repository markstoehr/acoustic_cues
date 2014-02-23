import numpy as np
import argparse

from acoustic_cues.patch_computations import patch_autocovariance

parser = argparse.ArgumentParser("""
Finding candidate locations of parts
""")
parser.add_argument('--data_prefix',type=str,help='prefix for sparse data path')
parser.add_argument('--data_suffix',type=str,help='suffix for sparse data path')
parser.add_argument('--patch_time_length',type=np.intc,default=13,help='side length for the extracted patch') #TODO extend to rectangular patches
parser.add_argument('--patch_freq_length',type=np.intc,default=13,help='side length for the extracted patch') #TODO extend to rectangular patches
parser.add_argument('--save_prefix',type=str,help='path to where we save the data')
parser.add_argument('--save_suffix',type=str,help='suffix for the saved patches data--ending indicates parameter settings usually')
# parser.add_argument('--',type=,help='')

args = parser.parse_args()

X_indices = np.load('%sX_indices_%s' % (args.data_prefix,args.data_suffix))
rownnz = np.load('%sX_rownnz_%s' % (args.data_prefix,args.data_suffix))
rowstartidx = np.load('%sX_rowstartidx_%s' % (args.data_prefix,args.data_suffix))
dim = np.load('%sdim_%s' % (args.data_prefix,args.data_suffix))


feature_counts, feature_cooccurrences = patch_autocovariance(
    X_indices,
    rownnz,
    rowstartidx,
    rownnz.shape[0],
    dim[0],
    dim[1],
    dim[2],
    args.patch_time_length,
    args.patch_freq_length)

np.save('%sfeature_cooccurrences_all_%s' % (args.save_prefix,args.save_suffix), feature_cooccurrences)
np.save('%sfeature_counts_all_%s' % (args.save_prefix,args.save_suffix), feature_counts)

Pdim = args.patch_time_length*args.patch_freq_length*dim[2]
general_cov = np.zeros((Pdim,Pdim))

feat_sq = dim[2]**2
for i in xrange(Pdim):
    d0 = int(i % dim[2])
    f0 = int((i / dim[2]) % args.patch_freq_length)
    t0 = int(i  / (dim[2] * args.patch_freq_length))
    for j in xrange(i,Pdim):
        d1 = int(j % dim[2])
        f1 = int((j / dim[2]) % args.patch_freq_length)
        t1 = int(j  / (dim[2] * args.patch_freq_length))
        time_offset = t1 - t0
        freq_offset = f1 - f0
        if t1 == t0 and freq_offset < 0:
            import pdb; pdb.set_trace()
        if time_offset > 0:
            offset_idx = time_offset *( 2*args.patch_freq_length -1) + freq_offset + args.patch_freq_length - 1
        elif time_offset < 0:
            import pdb; pdb.set_trace()
        else:
            offset_idx = freq_offset
        cooccur_idx = offset_idx * feat_sq + d0*dim[2] + d1
        general_cov[i,j] = feature_cooccurrences[time_offset,freq_offset,d0,d1] - feature_counts[d0]*feature_counts[d1]
        general_cov[j,i] = general_cov[i,j]

np.save('%scov_%s' % (args.save_prefix,args.save_suffix), general_cov)
            
import pdb; pdb.set_trace()
