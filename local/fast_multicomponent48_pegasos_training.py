from __future__ import division
from phoneclassification.phoneclassification.confusion_matrix import confusion_matrix
import numpy as np
import argparse,collections
from phoneclassification.phoneclassification.multicomponent_binary_sgd import BinaryArrayDataset, multiclass_sgd, sparse_dotmm

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

parser.add_argument('--model_avgs',type=str,help='path to where the models are saved that have been initialized')
parser.add_argument('--model_W',type=str,default=None,
                    help='path to where the model log odds are saved, by default this is none and we use the model avgs, if this is not none then this is used instead of the model avgs')
parser.add_argument('--model_meta',type=str,help='path to where the initialized model metadata have been saved')
parser.add_argument('--save_prefix',type=str,help='prefix for path to save the output to')
parser.add_argument('-l',type=float,nargs='+',help='lambda scaling parameter to be using')
parser.add_argument('--time_scaling',type=float,help='time scaling parameter')
parser.add_argument('--use_hinge',type=np.intc,default=1,help='whether to use the hinge loss')
parser.add_argument('--do_projection',action='store_true',help='whether to do the projection')
parser.add_argument('--reuse_previous_iterates',action='store_true',help='whether to build off of a warm-start from previous iterations')
parser.add_argument('--niter',type=np.intc,help='Number of iterations to run this for')
parser.add_argument('--start_t',type=float,default=1.0,help='start time initializer')

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



if args.model_W is None:
    avgs = np.load(args.model_avgs)
    avgs = avgs.reshape(avgs.shape[0],
                    dim)
    W = np.zeros((len(avgs),dim+1))
    W[:,:-1] = np.log(avgs) - np.log(1-avgs)
    W[:,-1] = np.log(1-avgs).sum(-1)
else:
    W = np.load(args.model_W)
W_meta = np.load(args.model_meta).astype(np.intc)


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


dset = BinaryArrayDataset(
                          patch_codes, patch_rownnz, patch_rowstartidx,y)
print y[12]


accuracy = test_accuracy(W)
print "old accuracy = %g" % accuracy
if args.do_projection:
    print "do_projection = True"
else:
    print "do_projection = False"

start_t = args.start_t
for l in args.l:
    print "Using lambda = %g " % l

    W_trained2 = W.ravel().copy()
                            
    print "number of iterations %d" % args.niter
    for iter_id in xrange(args.niter):
        W_trained = W_trained2.ravel().copy()
        W_trained2 = multiclass_sgd(W_trained,
                               weights_classes,
                               weights_components, np.intc(n_classes),
                                dset, np.intc(0), 1, np.intc(1),np.intc(1),start_t,
                                l,np.intc(args.do_projection),args.time_scaling,np.intc(args.use_hinge))

        np.save('%s_%gl_%dniter_W.npy' % (args.save_prefix,l,iter_id), W_trained2)
        print "W_trained2.shape= %s" % (str(W_trained2.shape))
        start_t = start_t + len(y)/2.
        accuracy = test_accuracy(W_trained2)
        print l,iter_id, accuracy
        open('%s_%gl_%dniter_accuracy.txt' % (args.save_prefix,l,iter_id),'w').write(str(accuracy ))
        if args.reuse_previous_iterates:
            W = W_trained2.copy()


accuracy = test_accuracy(W)
print "test accuracy = %g" % accuracy
