import numpy as np
from phoneclassification.phoneclassification.binary_sgd import binary_to_bsparse, add_final_one
from phoneclassification.phoneclassification._fast_EM import EM, e_step, m_step, sparse_dotmm
from acoustic_cues.patch_computations import localized_patch_coding

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
parser.add_argument('--model_prefix',type=str,help='prefix for path to load the model from')
parser.add_argument('--model_avgs',type=str,help='suffix for path to load the model from')
parser.add_argument('--out_prefix',type=str,help='prefix for path to save the output to')
parser.add_argument('--out_suffix',type=str,help='suffix for path to save the output to')
parser.add_argument('--ncomponents_per_class',type=int,default=5,help='number of components per class')
parser.add_argument('--part_size',type=int,nargs='+',help='time then frequency dimension for patch')
parser.add_argument('--noverlap',type=int,default=4,help='number of overlapping pixels in time and frequency')
# parser.add_argument('--min_counts',type=int,default=30,help='minimum number of examples assigned to each class')
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

n_templates = len(patch_time_pairs) * len(patch_freq_pairs)
template_meta = np.zeros((n_templates,6),dtype=np.intc)
template_id = 0
for tpair_id, patch_time_pair in enumerate(patch_time_pairs):
    for fpair_id, patch_freq_pair in enumerate(patch_freq_pairs):
        avgs = np.load('%s/avgs_%sTP_%sFP_%s' % (args.model_prefix,'_'.join([str(k) for k in patch_time_pair]),'_'.join([str(k) for k in patch_freq_pair]), args.model_avgs))

        avgs = avgs.reshape(*(
            (avgs.shape[0],) + tuple(dimensions)))
        avgs = avgs[:,patch_time_pair[0]:patch_time_pair[1],
                    patch_freq_pair[0]: patch_freq_pair[1]]

        meta = np.load('%s/meta_%sTP_%sFP_%s' % (args.model_prefix,'_'.join([str(k) for k in patch_time_pair]),'_'.join([str(k) for k in patch_freq_pair]), args.model_avgs))
        
        avgs_log_inv = np.log(1-avgs)

        model_constants = np.ascontiguousarray(avgs_log_inv.reshape(len(avgs),avgs[0].size).sum(-1))
        model_w = np.ascontiguousarray(np.log(avgs) - avgs_log_inv)
        if tpair_id == 0 and fpair_id == 0:
            template_length = model_w.size
            templates = np.zeros(template_length,dtype=float)
            templates[:] = model_w.ravel()
            template_meta[template_id,0] = patch_time_pair[1] - patch_time_pair[0]
            template_meta[template_id,1] = patch_freq_pair[1] - patch_freq_pair[0]
            template_meta[template_id,2] = avgs.shape[0]
            template_meta[template_id,3] = patch_time_pair[0]
            template_meta[template_id,4] = patch_freq_pair[0] - patch_time_pair[0]
            template_meta[template_id,5] = model_w.size
            template_constants = np.ascontiguousarray(model_constants.copy())
            num_components = len(template_constants)
            template_constants_length = template_constants.shape[0]


        else:
            new_template_length = template_length + model_w.size
            while new_template_length > templates.size:
                new_templates = np.zeros(2*templates.size,dtype=float)
                new_templates[:template_length] =templates[:template_length]
                templates = new_templates

            templates[template_length:new_template_length] = model_w.ravel()
            template_length = new_template_length
            template_meta[template_id,0] = patch_time_pair[1] - patch_time_pair[0]
            template_meta[template_id,1] = patch_freq_pair[1] - patch_freq_pair[0]
            template_meta[template_id,2] = avgs.shape[0]
            template_meta[template_id,3] = patch_time_pair[0]
            template_meta[template_id,4] = patch_freq_pair[0]
            template_meta[template_id,5] = model_w.size
    
            new_num_components = num_components + model_constants.size
            while new_num_components > template_constants.size:
                new_template_constants = np.zeros(2*template_constants.size,dtype=float)
                new_template_constants[:num_components] = template_constants[:num_components]
                template_constants = new_template_constants
                
            template_constants[num_components:new_num_components] = model_constants
            num_components = new_num_components
        
        template_id += 1
            

print "template_id=%d\t n_templates=%d" % (template_id,n_templates)
template_constants = np.ascontiguousarray(template_constants[:num_components])
templates = np.ascontiguousarray(templates[:template_length])
template_meta = np.ascontiguousarray(template_meta)
# best_template_scores = localized_patch_coding(templates,
#                                               template_constants,
#                            template_meta.ravel(),
#                                               n_templates,
#                            2, dimensions[0], dimensions[1],
#                                               dimensions[2],
#                            feature_ind,
#                            rownnz,
#                                               X_n_rows)
# np.save('%s/best_template_scores_%s' % (args.out_prefix,args.out_suffix),best_template_scores)

best_template_scores = np.load('%s/best_template_scores_%s' % (args.out_prefix,args.out_suffix))

max_best_template_scores = best_template_scores.max(-1).max(0)
n_block_entries = max_best_template_scores + 1
# first block starts at 0
block_ends = np.zeros(len(n_block_entries)+1,dtype=np.int32)
block_ends[1:] = np.cumsum(n_block_entries)

patch_codes = np.array([10])
patch_rownnz = np.zeros(X_n_rows,dtype=np.intc)
patch_codes_length = 0
for x_id, x_best_ids in enumerate(best_template_scores):
    patch_rownnz[x_id] = 0
    for patch_id,x_patch_best_ids in enumerate(x_best_ids):
        u = np.unique(x_patch_best_ids)
        while patch_codes_length + len(u) > patch_codes.shape[0]:
            new_patch_codes = np.zeros(patch_codes.size*2,dtype=np.intc)
            new_patch_codes[:patch_codes_length] = patch_codes[:patch_codes_length]
            patch_codes = new_patch_codes

        patch_codes[patch_codes_length:patch_codes_length+len(u)] = u + block_ends[patch_id]
        patch_codes_length += len(u)
        patch_rownnz[x_id] += len(u)

np.save('%s/patch_codes_train_%s' % (args.out_prefix,args.out_suffix),patch_codes)
np.save('%s/patch_rownnz_train_%s' % (args.out_prefix,args.out_suffix),patch_rownnz)


# best_template_scores_test = localized_patch_coding(templates,                                              template_constants,                           template_meta.ravel(),                                              n_templates,                           2, dimensions[0], dimensions[1],                                              dimensions[2],                           feature_ind_test,                           rownnz_test,                                              X_n_rows_test)

# np.save('%s/best_template_scores_test_%s' % (args.out_prefix,args.out_suffix),best_template_scores_test)

best_template_scores_test = np.load('%s/best_template_scores_test_%s' % (args.out_prefix,args.out_suffix))


max_best_template_scores_test = best_template_scores_test.max(-1).max(0)
n_block_entries_test = max_best_template_scores_test + 1
# first block starts at 0
block_ends_test = np.zeros(len(n_block_entries_test)+1,dtype=np.int32)
block_ends_test[1:] = np.cumsum(n_block_entries_test)


patch_codes_test = np.array([10])
patch_rownnz_test = np.zeros(X_n_rows_test,dtype=np.intc)
patch_codes_length_test = 0
for x_id, x_best_ids in enumerate(best_template_scores_test):
    patch_rownnz_test[x_id] = 0
    for patch_id,x_patch_best_ids in enumerate(x_best_ids):
        u = np.unique(x_patch_best_ids)
        while patch_codes_length_test + len(u) > patch_codes_test.shape[0]:
            new_patch_codes_test = np.zeros(patch_codes_test.size*2,dtype=np.intc)
            new_patch_codes_test[:patch_codes_length_test] = patch_codes_test[:patch_codes_length_test]
            patch_codes_test = new_patch_codes_test

        patch_codes_test[patch_codes_length_test:patch_codes_length_test+len(u)] = u + block_ends[patch_id]
        patch_codes_length_test += len(u)
        patch_rownnz_test[x_id] += len(u)

np.save('%s/patch_codes_test_%s' % (args.out_prefix,args.out_suffix),patch_codes_test)
np.save('%s/patch_rownnz_test_%s' % (args.out_prefix,args.out_suffix),patch_rownnz_test)



import pdb; pdb.set_trace()

            
        
#         feature_ind_freqs = (feature_ind / dimensions[2] ) % dimensions[1]
#         feature_ind_times = (feature_ind / dimensions[2]) / dimensions[1]
#         patch_use_ind_mask = (feature_ind_freqs >= patch_freq_pair[0]) * (feature_ind_freqs < patch_freq_pair[1]) * (feature_ind_times >= patch_time_pair[0]) * (feature_ind_times < patch_time_pair[1])
#         patch_X_indices = np.ascontiguousarray(feature_ind[patch_use_ind_mask])
#         patch_rownnz = np.zeros(rownnz.shape,dtype=np.intc)
#         patch_rowstartidx = np.zeros(rowstartidx.shape,dtype=np.intc)
#         cur_idx =0

#         for nnz_id, nnz in enumerate(rownnz):
#             patch_rownnz[nnz_id] = patch_use_ind_mask[cur_idx:cur_idx + nnz].sum()
#             patch_rowstartidx[nnz_id+1] = patch_rowstartidx[nnz_id] + patch_rownnz[nnz_id]
#             cur_idx += patch_rownnz[nnz_id]


#         localized_patch_coding(model_w.ravel(),
#                            model_constants,
#                            np.ndarray[ndim=1,dtype=int,
#                                       mode="c"] template_meta,
#                            int n_templates,
#                            int radius, int n_times, int n_freqs,
#                            int n_features,
#                            np.ndarray[ndim=1,dtype=int,
#                                       mode="c"] feature_ind,
#                            np.ndarray[ndim=1,dtype=int,
#                                       mode="c"] rownnz,
#                            int X_n_rows)

#         test_feature_ind_freqs = (test_feature_ind / dimensions[2] ) % dimensions[1]
#         test_feature_ind_times = (test_feature_ind / dimensions[2]) / dimensions[1]
#         test_patch_use_ind_mask = (test_feature_ind_freqs >= patch_freq_pair[0]) * (test_feature_ind_freqs < patch_freq_pair[1]) * (test_feature_ind_times >= patch_time_pair[0]) * (test_feature_ind_times < patch_time_pair[1])
#         test_patch_X_indices = np.ascontiguousarray(test_feature_ind[patch_use_ind_mask])
#         test_patch_rownnz = np.zeros(test_rownnz.shape,dtype=np.intc)
#         test_patch_rowstartidx = np.zeros(test_rowstartidx.shape,dtype=np.intc)
#         cur_idx =0

#         for nnz_id, nnz in enumerate(rownnz):
#             test_patch_rownnz[nnz_id] = test_patch_use_ind_mask[cur_idx:cur_idx + nnz].sum()
#             test_patch_rowstartidx[nnz_id+1] = test_patch_rowstartidx[nnz_id] + patch_rownnz[nnz_id]
#             cur_idx += test_patch_rownnz[nnz_id]

#         D = (patch_time_pair[1] - patch_time_pair[0]) * (patch_freq_pair[1] - patch_freq_pair[0])
        
#         for phn_id, phn in enumerate(leehon[:,0]):
#             print "Working on phone %s which has id %d" % (phn, phn_id)
#             print "classifier_id = %d" % classifier_id
    
#             phn_n_data = (y == phn_id).sum()
#             phn_rownnz = patch_rownnz[y==phn_id].copy()
#             phn_start_idx = np.where(y==phn_id)[0].min()
#             phn_end_idx = np.where(y==phn_id)[0].max()+1

#             phn_rowstartidx = patch_rowstartidx[phn_start_idx:phn_end_idx+1].copy()
#             phn_feature_ind = np.ascontiguousarray(patch_X_indices[phn_rowstartidx[0]:phn_rowstartidx[-1]]).copy()
#             phn_rowstartidx -= phn_rowstartidx[0]
    
#             converged = False
#             cur_ncomponents = args.ncomponents_per_class

#             if phn_id == 0:
#                 avgs = np.zeros((max_n_classifiers,
#                                  dim) )
#                 counts = np.zeros(max_n_classifiers
#                               )
#                 # will keep track of which average belongs to which
#                 # phone and mixture component--this allows us to
#                 # drop mixture components if they are potentially
#                 # not helping
#                 all_weights = np.zeros(max_n_classifiers,dtype=float)
#                 meta = np.zeros((max_n_classifiers
#                              ,2),dtype=int)

#             n_init = 0
#             tol = float(args.tol)
#             total_iter = np.intc(args.total_iter)
#             while n_init < args.total_init:

#                 A = np.zeros((phn_n_data,cur_ncomponents),dtype=float)
#                 A[np.arange(phn_n_data),np.random.randint(cur_ncomponents,size=phn_n_data)] = 1
#                 A = np.ascontiguousarray(A.reshape(A.size))
#                 P = np.zeros(dim*cur_ncomponents,dtype=float)
#                 weights = np.zeros(cur_ncomponents,dtype=float)

#                 # m_step(phn_feature_ind, phn_rownnz, phn_rowstartidx,
#                 #        P,weights, A, phn_n_data, dim, cur_ncomponents)
#                 # import pdb; pdb.set_trace() 


#                 P,weights, A, loglikelihood = EM(phn_feature_ind, phn_rownnz, phn_rowstartidx,
#                                                  phn_n_data,np.intc(dim),cur_ncomponents,tol, total_iter,
#                                                  A)
#                 A = A.reshape(phn_n_data,cur_ncomponents)
#                 P = P.reshape(cur_ncomponents, dim)
#                 component_counts = A.sum(0)
#                 good_components = component_counts >= args.min_counts
#                 n_good = good_components.sum()
                
#                 while np.any(component_counts < args.min_counts):           
#                     good_components = component_counts >= args.min_counts
#                     n_good = good_components.sum()
#                     P = P[good_components]
#                     weights = weights[good_components]
#                     A = np.zeros((phn_n_data,n_good),dtype=float)
#                     A = A.reshape(A.size)
#                     P = P.reshape(P.size)
        
        
#                     likelihood = e_step(phn_feature_ind, 
#                            phn_rownnz, 
#                            phn_rowstartidx,
#                            P, 
#                            weights, 
#                            A, phn_n_data, dim, n_good )
        
#                     P,weights, A, loglikelihood = EM(phn_feature_ind, phn_rownnz, phn_rowstartidx,
                                                     
#                                                      phn_n_data,dim,n_good,args.tol, args.total_iter,
#                                                      A)
#                     A = A.reshape(phn_n_data,n_good)
#                     P = P.reshape(n_good, dim)
#                     component_counts = A.sum(0)

#                 if n_init == 0:
#                     bestP = P.copy()
#                     bestweights = weights.copy()
#                     best_ll = loglikelihood
#                     n_use_components = n_good
#                 elif loglikelihood > best_ll:
#                     print "Updated best loglikelihood to : %g " % loglikelihood
#                     bestP = P.copy()
#                     bestweights = weights.copy()
#                     best_ll = loglikelihood
#                     n_use_components = n_good
                    
#                 n_init += 1
        
#             # add the components
#             avgs[classifier_id:classifier_id + n_use_components] = bestP[:]
#             all_weights[classifier_id:classifier_id + n_use_components] = bestweights[:]
#             meta[classifier_id:classifier_id+n_use_components,0] = phn_id
#             meta[classifier_id:classifier_id+n_use_components,1] = np.arange(n_use_components)
            
#             classifier_id += n_use_components

#         print "Total of %d models for time-pair: %s and freq-pair: %s" % (classifier_id,str(patch_time_pair),str(patch_freq_pair))
#         np.save('%s/avgs_%sTP_%sFP_%s' % (args.out_prefix,'_'.join([str(k) for k in patch_time_pair]),'_'.join([str(k) for k in patch_freq_pair]), args.out_suffix),
#                     avgs[:classifier_id])
        
#         np.save('%s/weights_%sTP_%sFP_%s' % (args.out_prefix,'_'.join([str(k) for k in patch_time_pair]),'_'.join([str(k) for k in patch_freq_pair]), args.out_suffix),
#                     weights[:classifier_id])
        
#         np.save('%s/meta_%sTP_%sFP_%s' % (args.out_prefix,'_'.join([str(k) for k in patch_time_pair]),'_'.join([str(k) for k in patch_freq_pair]), args.out_suffix),
#             meta[:classifier_id])

        
    



# import pdb; pdb.set_trace()


