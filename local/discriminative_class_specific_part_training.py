import numpy as np
from phoneclassification.phoneclassification.binary_sgd import binary_to_bsparse, add_final_one
from phoneclassification.phoneclassification._fast_EM import EM, e_step, m_step
from phoneclassification.phoneclassification.multicomponent_binary_sgd import BinaryArrayDataset, multiclass_sgd, sparse_dotmm

import argparse, collections
# load in the data

parser = argparse.ArgumentParser("""File to run a basic test of the uncertainty
EM algorithm""")
parser.add_argument('--root_dir',default='/home/mark/Research/phoneclassification',type=str,help='root directory for where to look for things')
parser.add_argument('--data_dir',default='data/local/data',type=str,
                    help='relative path to where the data is kept')

parser.add_argument('--use_sparse_suffix',default=None,
                    type=str,help='If not included then we do not assume a sparse save structure for the data otherwise this is the suffix for where the data are stored in sparse format')
parser.add_argument('--dev_sparse_suffix',default=None,
                    type=str,help='If not included then we do not assume a sparse save structure for the data otherwise this is the suffix for where the data are stored in sparse format')

parser.add_argument('--out_prefix',type=str,help='prefix for path to save the output to')
parser.add_argument('--out_suffix',type=str,help='suffix for path to save the output to')
parser.add_argument('--total_iter',type=np.intc,help='Number of iterations to run this for')
parser.add_argument('--total_init',type=np.intc,help='Number of initializations to use in estimating the models')
parser.add_argument('--min_counts',type=np.intc,default=30,help='Minimum number of examples for each component')
parser.add_argument('--ncomponents_per_class',type=int,default=5,help='number of components per class')
parser.add_argument('--part_size',type=int,nargs='+',help='time then frequency dimension for patch')
parser.add_argument('--noverlap',type=int,default=4,help='number of overlapping pixels in time and frequency')
# parser.add_argument('--min_counts',type=int,default=30,help='minimum number of examples assigned to each class')
parser.add_argument('--tol',type=float,help='Convergence criterion')
# parser.add_argument('--ncomponents',type=np.intc,help='Maximum number of components per model')

# parser.add_argument('--',type=,help='')
args = parser.parse_args()

rootdir = args.root_dir[:]
confdir='%s/conf'%rootdir
datadir=args.data_dir

leehon=np.loadtxt('%s/phones.48-39' % confdir,dtype=str)
phones39 = np.unique(np.sort(leehon[:,1]))
phones39_dict = dict( (v,i) for i,v in enumerate(phones39))
phones48_dict = dict( (v,i) for i,v in enumerate(leehon[:,0]))
leehon_dict = dict( (phones48_dict[p],
                     phones39_dict[q]) for p,q in leehon)
leehon_dict_array = np.zeros(48,dtype=int)
for k,v in leehon_dict.items():
    leehon_dict_array[k] = int(v)


leehon_phn_dict = dict( (p,q) for p,q in leehon)

leehon39to48 = collections.defaultdict(list)

for phn in leehon[:,0]:
    leehon39to48[leehon_phn_dict[phn]].append(phn)

print "loading in data now"

feature_ind = np.load('%sX_indices_%s' % (args.data_dir,
                                              args.use_sparse_suffix),
                          )
rownnz = np.load('%sX_rownnz_%s' % (args.data_dir,
                                              args.use_sparse_suffix),
                          )
dim = np.intc(np.prod(np.load('%sdim_%s' % (args.data_dir, args.use_sparse_suffix))))
dimensions = np.load('%sdim_%s' % (args.data_dir, args.use_sparse_suffix)).astype(np.intc)
X_n_rows = rownnz.shape[0]
rowstartidx = np.load('%sX_rowstartidx_%s' % (args.data_dir,
                                              args.use_sparse_suffix),
                          )

y = np.load('%sy_%s' % (args.data_dir,
                                              args.use_sparse_suffix),
                          ).astype(np.int16)

print "loaded in training data"

feature_ind_test = np.load('%sX_indices_%s' % (args.data_dir,
                                              args.dev_sparse_suffix),
                          )
rownnz_test = np.load('%sX_rownnz_%s' % (args.data_dir,
                                              args.dev_sparse_suffix),
                          )
rowstartidx_test = np.load('%sX_rowstartidx_%s' % (args.data_dir,
                                              args.dev_sparse_suffix),
                           )
feature_ind_test, rownnz_test,rowstartidx_test = add_final_one(feature_ind_test,rownnz_test,rowstartidx_test,dim)

y_test = np.load('%sy_%s' % (args.data_dir,
                                              args.dev_sparse_suffix),
                          ).astype(np.int16)
X_n_rows_test = y_test.shape[0]
y_test39 = np.array([ leehon_dict[phone_id] for phone_id in y_test]).astype(np.int16)
test_accuracy = lambda W : np.sum(leehon_dict_array[weights_classes[sparse_dotmm(feature_ind_test,rownnz_test,rowstartidx_test,W.ravel().copy(),X_n_rows_test,W.shape[1],W.shape[0]).argmax(1)]] == y_test39)/float(len(y_test39))

class_n_indices = np.zeros(48,dtype=np.intc)
for i in xrange(48):
    class_n_indices[i] = rownnz[y == i].sum()


class_n_data = np.zeros(48,dtype=np.intc)
for i in xrange(48):
    class_n_data[i] = (y==i).sum()


hop_factor = args.part_size[0] - args.noverlap

patch_time_pairs = np.array([
    (k*hop_factor, min(k*hop_factor+args.part_size[0],dimensions[0]))
    for k in xrange(dimensions[0]/hop_factor )])

hop_factor = args.part_size[1] - args.noverlap
patch_freq_pairs = np.array([
    (k*hop_factor, min(k*hop_factor+args.part_size[1],dimensions[1]))
    for k in xrange(dimensions[1]/hop_factor )])

for tpair_id, patch_time_pair in enumerate(patch_time_pairs):
    for fpair_id, patch_freq_pair in enumerate(patch_freq_pairs):
        feature_ind_freqs = (feature_ind / dimensions[2] ) % dimensions[1]
        feature_ind_times = (feature_ind / dimensions[2]) / dimensions[1]
        patch_use_ind_mask = (feature_ind_freqs >= patch_freq_pair[0]) * (feature_ind_freqs < patch_freq_pair[1]) * (feature_ind_times >= patch_time_pair[0]) * (feature_ind_times < patch_time_pair[1])
        patch_X_indices = np.ascontiguousarray(feature_ind[patch_use_ind_mask])
        patch_rownnz = np.zeros(rownnz.shape,dtype=np.intc)
        patch_rowstartidx = np.zeros(rowstartidx.shape,dtype=np.intc)
        cur_idx =0

        for nnz_id, nnz in enumerate(rownnz):
            patch_rownnz[nnz_id] = patch_use_ind_mask[cur_idx:cur_idx + nnz].sum()
            patch_rowstartidx[nnz_id+1] = patch_rowstartidx[nnz_id] + patch_rownnz[nnz_id]
            cur_idx += patch_rownnz[nnz_id]

        D = (patch_time_pair[1] - patch_time_pair[0]) * (patch_freq_pair[1] - patch_freq_pair[0])

        avgs = np.load('%s/avgs_%sTP_%sFP_%s' % (args.model_prefix,'_'.join([str(k) for k in patch_time_pair]),'_'.join([str(k) for k in patch_freq_pair]), args.model_avgs))


        meta = np.load('%s/meta_%sTP_%sFP_%s' % (args.model_prefix,'_'.join([str(k) for k in patch_time_pair]),'_'.join([str(k) for k in patch_freq_pair]), args.model_avgs)).astype(np.intc)

        avgs_log_inv = np.log(1-avgs)

        model_constants = np.ascontiguousarray(avgs_log_inv.reshape(len(avgs),avgs[0].size).sum(-1))
        model_w = np.ascontiguousarray(np.ones((avgs.shape[0],avgs.shape[1]+1),dtype=float))
        model_w[:,:-1] = np.ascontiguousarray(np.log(avgs) - avgs_log_inv)
        model_w[:,-1] = model_constants

        weights = np.ascontiguousarray(model_w.ravel())
        weights_classes = meta[:,0].copy()
        weights_components = meta[:,1].copy()
        sorted_component_ids = np.argsort(weights_components,kind='mergesort')
        sorted_components = weights_components[sorted_component_ids]
        sorted_weights_classes = weights_classes[sorted_component_ids]
        stable_sorted_weights_classes_ids = np.argsort(sorted_weights_classes,kind='mergesort')
        weights_classes = sorted_weights_classes[stable_sorted_weights_classes_ids]
        weights_components = sorted_components[stable_sorted_weights_classes_ids]

        W = model_w[sorted_component_ids][stable_sorted_weights_classes_ids]

        n_classes = 48
        print "n_classes=%d" % n_classes

        patch_X_indices, patch_rownnz,patch_rowstartidx = add_final_one(patch_X_indices,patch_rownnz,patch_rowstartidx,dim)

        dset = BinaryArrayDataset(
            patch_X_indices, patch_rownnz, patch_rowstartidx,y)

        W_trained2 = np.ascontiguousarray(W.ravel())

        print "number of iterations %d" % 13
        for iter_id in xrange(13):
            W_trained = W_trained2.ravel().copy()
            W_trained2 = multiclass_sgd(W_trained,
                               weights_classes,
                               weights_components, np.intc(n_classes),
                                dset, np.intc(0), 1, np.intc(1),np.intc(1),10000,
                                .05,0,1.0,np.intc(1))

        np.load('%s/w_patches.05_%sTP_%sFP_%s' % (args.model_prefix,'_'.join([str(k) for k in patch_time_pair]),'_'.join([str(k) for k in patch_freq_pair]), args.model_avgs))





import pdb; pdb.set_trace()


