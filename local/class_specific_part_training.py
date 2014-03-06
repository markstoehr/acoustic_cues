import numpy as np
from phoneclassification.phoneclassification.binary_sgd import binary_to_bsparse, add_final_one
from phoneclassification.phoneclassification._fast_EM import EM, e_step, m_step

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
                          ).astype(np.intc)



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
                          ).astype(np.intc)
X_n_rows_test = y_test.shape[0]
y_test39 = np.array([ leehon_dict[phone_id] for phone_id in y_test]).astype(np.intc)
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
        
        max_n_classifiers = len(leehon)*args.ncomponents_per_class
        classifier_id = 0
        for phn_id, phn in enumerate(leehon[:,0]):
            print "Working on phone %s which has id %d" % (phn, phn_id)
            print "classifier_id = %d" % classifier_id
    
            phn_n_data = (y == phn_id).sum()
            phn_rownnz = patch_rownnz[y==phn_id].copy()
            phn_start_idx = np.where(y==phn_id)[0].min()
            phn_end_idx = np.where(y==phn_id)[0].max()+1

            phn_rowstartidx = patch_rowstartidx[phn_start_idx:phn_end_idx+1].copy()
            phn_feature_ind = np.ascontiguousarray(patch_X_indices[phn_rowstartidx[0]:phn_rowstartidx[-1]]).copy()
            phn_rowstartidx -= phn_rowstartidx[0]
    
            converged = False
            cur_ncomponents = args.ncomponents_per_class

            if phn_id == 0:
                avgs = np.zeros((max_n_classifiers,
                                 dim) )
                counts = np.zeros(max_n_classifiers
                              )
                # will keep track of which average belongs to which
                # phone and mixture component--this allows us to
                # drop mixture components if they are potentially
                # not helping
                all_weights = np.zeros(max_n_classifiers,dtype=float)
                meta = np.zeros((max_n_classifiers
                             ,2),dtype=int)

            n_init = 0
            tol = float(args.tol)
            total_iter = np.intc(args.total_iter)
            while n_init < args.total_init:

                A = np.zeros((phn_n_data,cur_ncomponents),dtype=float)
                A[np.arange(phn_n_data),np.random.randint(cur_ncomponents,size=phn_n_data)] = 1
                A = np.ascontiguousarray(A.reshape(A.size))
                P = np.zeros(dim*cur_ncomponents,dtype=float)
                weights = np.zeros(cur_ncomponents,dtype=float)

                # m_step(phn_feature_ind, phn_rownnz, phn_rowstartidx,
                #        P,weights, A, phn_n_data, dim, cur_ncomponents)
                # import pdb; pdb.set_trace() 


                P,weights, A, loglikelihood = EM(phn_feature_ind, phn_rownnz, phn_rowstartidx,
                                                 phn_n_data,np.intc(dim),cur_ncomponents,tol, total_iter,
                                                 A)
                A = A.reshape(phn_n_data,cur_ncomponents)
                P = P.reshape(cur_ncomponents, dim)
                component_counts = A.sum(0)
                good_components = component_counts >= args.min_counts
                n_good = good_components.sum()
                
                while np.any(component_counts < args.min_counts):           
                    good_components = component_counts >= args.min_counts
                    n_good = good_components.sum()
                    P = P[good_components]
                    weights = weights[good_components]
                    A = np.zeros((phn_n_data,n_good),dtype=float)
                    A = A.reshape(A.size)
                    P = P.reshape(P.size)
        
        
                    likelihood = e_step(phn_feature_ind, 
                           phn_rownnz, 
                           phn_rowstartidx,
                           P, 
                           weights, 
                           A, phn_n_data, dim, n_good )
        
                    P,weights, A, loglikelihood = EM(phn_feature_ind, phn_rownnz, phn_rowstartidx,
                                                     
                                                     phn_n_data,dim,n_good,args.tol, args.total_iter,
                                                     A)
                    A = A.reshape(phn_n_data,n_good)
                    P = P.reshape(n_good, dim)
                    component_counts = A.sum(0)

                if n_init == 0:
                    bestP = P.copy()
                    bestweights = weights.copy()
                    best_ll = loglikelihood
                    n_use_components = n_good
                elif loglikelihood > best_ll:
                    print "Updated best loglikelihood to : %g " % loglikelihood
                    bestP = P.copy()
                    bestweights = weights.copy()
                    best_ll = loglikelihood
                    n_use_components = n_good
                    
                n_init += 1
        
            # add the components
            avgs[classifier_id:classifier_id + n_use_components] = bestP[:]
            all_weights[classifier_id:classifier_id + n_use_components] = bestweights[:]
            meta[classifier_id:classifier_id+n_use_components,0] = phn_id
            meta[classifier_id:classifier_id+n_use_components,1] = np.arange(n_use_components)
            
            classifier_id += n_use_components

        print "Total of %d models for time-pair: %s and freq-pair: %s" % (classifier_id,str(patch_time_pair),str(patch_freq_pair))
        np.save('%s/avgs_%sTP_%sFP_%s' % (args.out_prefix,'_'.join([str(k) for k in patch_time_pair]),'_'.join([str(k) for k in patch_freq_pair]), args.out_suffix),
                    avgs[:classifier_id])
        
        np.save('%s/weights_%sTP_%sFP_%s' % (args.out_prefix,'_'.join([str(k) for k in patch_time_pair]),'_'.join([str(k) for k in patch_freq_pair]), args.out_suffix),
                    weights[:classifier_id])
        
        np.save('%s/meta_%sTP_%sFP_%s' % (args.out_prefix,'_'.join([str(k) for k in patch_time_pair]),'_'.join([str(k) for k in patch_freq_pair]), args.out_suffix),
            meta[:classifier_id])

        
    



import pdb; pdb.set_trace()


