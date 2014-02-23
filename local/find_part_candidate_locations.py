import numpy as np
import argparse, itertools

from acoustic_cues.patch_computations import patch_sums, summed_area_table


def nonmaximal_suppression(psums, patch_side_length, nlocs):
    locs = np.zeros((nlocs,3))
    s = psums.copy()
    half_rad = int(patch_side_length/2)
    for cur_loc in xrange(nlocs):
        val = s.max()
        max_loc = np.where(s >= val)
        max_loc_x = max_loc[0][0]
        max_loc_y = max_loc[1][0]
        lo_x = max(max_loc_x -half_rad,0)
        lo_y = max(max_loc_y -half_rad,0)
        hi_x = min(max_loc_x +half_rad,s.shape[0])
        hi_y = min(max_loc_y +half_rad,s.shape[1])
        s[lo_x:hi_x,lo_y:hi_y] = 0
        locs[cur_loc,0] = val
        locs[cur_loc,1] = max_loc_x
        locs[cur_loc,2] = max_loc_y

    return locs

parser = argparse.ArgumentParser("""
Finding candidate locations of parts
""")
parser.add_argument('--W',type=str,help='path to classifier matrix')
parser.add_argument('--meta',type=str,help='path to the metadata for the classifier matrix')
parser.add_argument('--patch_side_length',type=int,default=13,help='side length for the extracted patch') #TODO extend to rectangular patches
parser.add_argument('--npatches_component',type=int,help='number of patches to find')
parser.add_argument('--save_prefix',type=str,help='path to where we save the data')
parser.add_argument('--save_suffix',type=str,help='suffix for the saved patches data--ending indicates parameter settings usually')
parser.add_argument('--dim',type=str,help='path to where the dimensions for the block are saved')
# parser.add_argument('--',type=,help='')

args = parser.parse_args()

W = np.load(args.W)
meta = np.load(args.meta)
dim = tuple(np.load(args.dim).astype(int))
# things to store about patches:
# 1. coordinates - 4 numbers
# 2. rank - simply location within the list
# 3. energy fraction and absolute energy 2 numbers
# 4. metadata -- class id and component id 2 numbers

n_models = W.shape[0]
part_stats = np.zeros((n_models * args.npatches_component, 9))

n_times, n_freqs, n_parts = dim[:]

for component_id, w in enumerate(W):
    # compute integral image

    V = w[:-1].copy().reshape(*(dim))
    v = np.ascontiguousarray((V**2).sum(-1)).reshape(n_times*n_freqs)
    table = summed_area_table(v,n_freqs,n_times)
    psums = patch_sums(table,args.patch_side_length,n_times,n_freqs).reshape(n_times,n_freqs)
    best_patches = nonmaximal_suppression(psums, args.patch_side_length, args.npatches_component)
    for patch_id, (best_patch_t,best_patch_f) in enumerate(itertools.izip(best_patches[:,1],
                                                                          best_patches[:,2])):
        cur_idx = component_id*args.npatches_component + patch_id
        print cur_idx
        part_stats[cur_idx,0] = best_patch_t
        part_stats[cur_idx,1] = best_patch_f
        part_stats[cur_idx,2] = best_patch_t + args.patch_side_length
        part_stats[cur_idx,3] = best_patch_f + args.patch_side_length
        part_stats[cur_idx,4] = psums[best_patch_t,best_patch_f]
        part_stats[cur_idx,5] = psums[best_patch_t,best_patch_f]/ v.sum()
        
        part_stats[cur_idx,6] = meta[component_id,0]
        part_stats[cur_idx,7] = meta[component_id,1]
        part_stats[cur_idx,8] = component_id


np.save('%s/part_stats_%s' % (args.save_prefix,
                              args.save_suffix),part_stats)

    

    
    
    
