from __future__ import division
from phoneclassification.phoneclassification.confusion_matrix import confusion_matrix
import numpy as np
import argparse,collections
from phoneclassification.phoneclassification._fast_EM import EM, e_step, m_step


"""
Extract the set of data associated with a set of phones
and give a label set, also initialize the components using the
basic components from the model
"""

def get_use_phns_row_ids(W_meta,use_phns,phones_dict):
    row_ids = []
    for phn_id, phn in enumerate(use_phns):
        phn_row_ids = np.where(W_meta[:,0]==phones_dict[phn])[0]
        row_ids.extend(phn_row_ids)
        W_meta[phn_row_ids,0] = phn_id

    return W_meta, np.array(row_ids)

def get_reduced_meta(W_meta,leehon_dict):
    W_meta_leehon = W_meta.copy()
    # keep a record of the component id under the
    # reduced set of labels
    component_count = collections.defaultdict(int)
    for w_id, w in enumerate(W_meta):
        W_meta_leehon[w_id,0] = leehon_dict[w[0]]
        W_meta_leehon[w_id,1] = component_count[W_meta_leehon[w_id,0]]
        component_count[W_meta_leehon[w_id,0]] += 1

    return W_meta_leehon

parser = argparse.ArgumentParser("""File to run a basic test of the pegasos multiclass
SVM solver over the scattering features""")
parser.add_argument('--root_dir',default='/home/mark/Research/acoustic_cues',type=str,help='root directory for where to look for things')
parser.add_argument('--in_prefix',type=str,help='prefix for path containing the feature indices and number of non-zero features per datum')
parser.add_argument('--in_suffix',type=str,help='suffix for path containing the data files')
parser.add_argument('--label_in_prefix',type=str,help='prefix for path containing the feature indices and number of non-zero features per datum')
parser.add_argument('--label_in_suffix',type=str,help='suffix for path containing the data files')
parser.add_argument('--label_in_suffix_test',type=str,help='suffix for path containing the data files')
parser.add_argument('--out_prefix',type=str,help='prefix for path to save the output to')
parser.add_argument('--out_suffix',type=str,help='suffix for path to save the output to')
parser.add_argument('--total_iter',type=np.intc,help='Number of iterations to run this for')
parser.add_argument('--total_init',type=np.intc,help='Number of initializations to use in estimating the models')
parser.add_argument('--min_counts',type=np.intc,help='Minimum number of examples for each component')
parser.add_argument('--tol',type=float,help='Convergence criterion')
parser.add_argument('--ncomponents',type=np.intc,help='Maximum number of components per model')
# parser.add_argument('--',type=,help='')
args = parser.parse_args()

rootdir = args.root_dir[:]
confdir='%s/conf'%rootdir



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

use_phns39 = list(phones39[:])
use_phns48 = leehon[:,0]

patch_codes = np.load('%spatch_codes_train_%s' % (args.in_prefix,
                                              args.in_suffix),
                          )
patch_rownnz = np.load('%spatch_rownnz_train_%s' % (args.in_prefix,
                                              args.in_suffix),
                          )
patch_rowstartidx = np.zeros(patch_rownnz.shape[0] + 1,dtype=np.intc)
patch_rowstartidx[1:] = np.cumsum(patch_rownnz)
patch_codes_test = np.load('%spatch_codes_train_%s' % (args.in_prefix,
                                              args.in_suffix),
                          )
patch_rownnz_test = np.load('%spatch_rownnz_train_%s' % (args.in_prefix,
                                              args.in_suffix),
                          )

patch_rowstartidx_test = np.zeros(patch_rownnz.shape[0] + 1,dtype=np.intc)
patch_rowstartidx_test[1:] = np.cumsum(patch_rownnz_test)

n_train_data = patch_rownnz.shape[0]
n_test_data = patch_rownnz_test.shape[0]
dim = np.intc(max(patch_codes.max(),patch_codes_test.max()) + 1)


y = np.load('%sy_%s' % (args.label_in_prefix,
                                              args.label_in_suffix),
                          ).astype(np.int16)

y_test = np.load('%sy_%s' % (args.label_in_prefix,
                                              args.label_in_suffix_test),
                          ).astype(np.int16)


n_test_data = y_test.shape[0]
y_test39 = np.array([ leehon_dict[phone_id] for phone_id in y_test]).astype(np.int16)
test_accuracy = lambda W : np.sum(leehon_dict_array[weights_classes[sparse_dotmm(patch_codes_test,patch_rownnz_test,patch_rowstartidx_test,W.ravel().copy(),n_test_data,W.shape[1],W.shape[0]).argmax(1)]] == y_test39)/float(len(y_test39))


max_n_classifiers = args.ncomponents * 48
classifier_id = 0
for phn_id, phn in enumerate(leehon[:,0]):
    print "Working on phone %s which has id %d" % (phn, phn_id)
    print "classifier_id = %d" % classifier_id
    
    phn_n_data = (y == phn_id).sum()
    phn_rownnz = patch_rownnz[y==phn_id].copy()
    phn_start_idx = np.where(y==phn_id)[0].min()
    phn_end_idx = np.where(y==phn_id)[0].max()+1
    if (phn_end_idx - phn_start_idx) != len(phn_rownnz):
        import pdb; pdb.set_trace()
        
    phn_rowstartidx = patch_rowstartidx[phn_start_idx:phn_end_idx+1].copy()
    phn_feature_ind = patch_codes[phn_rowstartidx[0]:phn_rowstartidx[-1]].copy()

    phn_rowstartidx -= phn_rowstartidx[0]
    
    converged = False
    cur_ncomponents = args.ncomponents
    
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
        A = A.reshape(A.size)
        P = np.zeros(dim*cur_ncomponents,dtype=float)
        weights = np.zeros(cur_ncomponents,dtype=float)

        # m_step(phn_feature_ind, phn_rownnz, phn_rowstartidx,
        #        P,weights, A, phn_n_data, dim, cur_ncomponents)
        # import pdb; pdb.set_trace() 
        P,weights, A, loglikelihood = EM(phn_feature_ind, phn_rownnz, phn_rowstartidx,
           phn_n_data,dim,cur_ncomponents,tol, total_iter,
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

print "Total of %d models" % classifier_id
np.save('%s/avgs_%s' % (args.out_prefix, args.out_suffix),
            avgs[:classifier_id])

np.save('%s/weights_%s' % (args.out_prefix, args.out_suffix),
            weights[:classifier_id])

np.save('%s/meta_%s' % (args.out_prefix, args.out_suffix),
            meta[:classifier_id])


# now we test the model to see what happens
avgs = avgs.reshape(avgs.shape[0],
                    dim)
W = np.zeros((len(avgs),dim+1))
W[:,:-1] = np.log(avgs) - np.log(1-avgs)
W[:,-1] = np.log(1-avgs).sum(-1)
W_meta = meta.astype(np.intc)



# need to construct W_meta39 to use 39 labels
W_meta39 = get_reduced_meta(W_meta,leehon_dict).astype(np.intc)
# now we get the use_phns39 row ids



weights = W.ravel().copy()
weights_classes = W_meta[:,0].copy()
weights_components = W_meta[:,1].copy()
sorted_component_ids = np.argsort(weights_components,kind='mergesort')
sorted_components = weights_components[sorted_component_ids]
sorted_weights_classes = weights_classes[sorted_component_ids]
stable_sorted_weights_classes_ids = np.argsort(sorted_weights_classes,kind='mergesort')
weights_classes = sorted_weights_classes[stable_sorted_weights_classes_ids]
weights_components = sorted_components[stable_sorted_weights_classes_ids]

W = W[sorted_component_ids][stable_sorted_weights_classes_ids]

n_classes = 48
print "n_classes=%d" % n_classes


accuracy = test_accuracy(W)
print "test accuracy = %g" % accuracy
