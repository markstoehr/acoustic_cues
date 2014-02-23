# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Mark Stoehr
#
#         
#
# Licence: BSD 3 clause


import numpy as np
import sys
from time import time

cimport cython
from libc.limits cimport INT_MAX
from libc.math cimport exp, log, sqrt, pow, fabs
cimport numpy as np

cdef inline double double_min(double a, double b) nogil: return a if a <= b else b
cdef inline double double_max(double a, double b) nogil: return b if a <= b else a
cdef inline int int_max(int a, int b) nogil: return b if a <= b else a

np.import_array()

# Learning rate constants
DEF CONSTANT = 1
DEF OPTIMAL = 2
DEF INVSCALING = 3
DEF PA1 = 4
DEF PA2 = 5

DEF EPSILON = 0.000001
DEF LOGEPSILON = -13.8156

# setup bool types
BOOL = np.uint8
ctypedef np.uint8_t BOOL_t
SHORT = np.int16
ctypedef np.int16_t SHORT_t

cdef void _summed_area_table(double *w, double *table,
                             int ncols, int nrows, int N) nogil:
    cdef int i,j, n
    n = 0
    for i in range(nrows):
        for j in range(ncols):
            table[n] = w[n]
            if i > 0:
                table[n] += table[(i-1)*ncols + j]
            
            if j > 0:
                table[n] += table[i*ncols + (j-1)]
                
            if i> 0 and j>0:
                table[n] -= table[(i-1)*ncols + (j-1)]

            n = n+1

def summed_area_table(np.ndarray[double,ndim=1,mode="c"] w,
                      int ncols, int nrows):
    """
    Assumed to be of a certain length of rows and columns
    """
    cdef:
        int N = w.size
        np.ndarray[double,ndim=1,mode="c"] table = np.zeros(N,dtype=np.float)
        
    _summed_area_table(<double *>w.data,<double *>table.data, ncols, nrows, N)
    return table

def patch_sums(np.ndarray[double,ndim=1,mode="c"] sum_table,
               int patch_side_length, int nrows, int ncols):
    cdef:
        int N = sum_table.size
        np.ndarray[double,ndim=1,mode="c"] psums = np.zeros(N,dtype=np.float)
        
    _patch_sums(<double *>sum_table.data,<double *> psums.data,
                nrows, ncols, patch_side_length)
    return psums
    
cdef void _patch_sums(double *sum_table, double *psums,
                      int nrows, int ncols, int patch_side_length) nogil:
    cdef int i,j,n
    cdef double *psums_ptr = <double *> psums
    n = 0
    for i in range(nrows-patch_side_length+1):
        for j in range(ncols-patch_side_length+1):
            psums_ptr = <double *> &psums[i*ncols + j]
            psums_ptr[0] =  sum_table[(i+patch_side_length-1)*ncols+ (j+patch_side_length-1)]
            if i > 0:
                psums_ptr[0] -= sum_table[(i-1)*ncols + (j+patch_side_length-1)]
            if j > 0:
                psums_ptr[0] -= sum_table[(i+patch_side_length-1)*ncols + j-1]
            
            if i > 0 and j > 0:
                psums_ptr[0] += sum_table[(i-1)*ncols + j-1]


cdef void _count_poffsets_in_X(double * n_poffsets, int n_patch_offsets,
                          int patch_times, int patch_freqs,
                          int X_n_times, int X_n_freqs) nogil:
    cdef int i0,j0,i1,j1,time_offset, freq_offset, idx

    for i0 in range(X_n_times):
        for j0 in range(X_n_freqs):
            for i1 in range(X_n_times):
                for j1 in range(X_n_freqs):
                    time_offset = i0 - i1
                    freq_offset = j0 - j1
                    if time_offset < 0:
                        time_offset = i1 - i0
                        freq_offset = - freq_offset
                    elif (time_offset == 0) and (freq_offset < 0):
                        freq_offset = -freq_offset
                        
                    if time_offset == 0:
                        idx = freq_offset
                    else:
                        idx = time_offset * (2*patch_freqs -1) + freq_offset + patch_freqs -1

                    # check whether the offset belongs in a patch
                    if idx < n_patch_offsets and freq_offset < patch_freqs and freq_offset > -patch_freqs:
                        n_poffsets[idx] += 0.5

    # cleanup step where we divide by 2 for all the offsets other than (0,0) since we have double counted
    n_poffsets[0] *= 2.0

                        

def patch_autocovariance(np.ndarray[int,ndim=1,mode="c"] X_indices,
                         np.ndarray[int,ndim=1,mode="c"] rownnz,
                         np.ndarray[int,ndim=1,mode="c"] rowstartidx,
                         int X_n_rows,
                         int X_n_times,
                         int X_n_freqs,
                         int X_n_features,
                         int patch_times,
                         int patch_freqs):
    cdef:
        int i,t,f,d1,d2
        double n_occurrences
        int N = X_n_times * X_n_freqs * X_n_features
        int P = patch_times * patch_freqs * X_n_features
        np.ndarray[double,ndim=1,mode="c"] feature_counts = np.zeros(X_n_features,dtype=np.float)
        np.ndarray[double,ndim=1,mode="c"] feature_cooccurrences = np.zeros(patch_times*(2*patch_freqs-1)* X_n_features *X_n_features,dtype=np.float)
        int cooccur_d0_stride = X_n_features
        int cooccur_f_stride = X_n_features * cooccur_d0_stride
        int cooccur_t_stride = (2*patch_freqs -1) * cooccur_f_stride
        double n_locs = <double> (X_n_times * X_n_freqs * X_n_rows)

        # number of offsets that can occur within the patch totals to
        # T*F + (T-1)*(F-1) = T*(2F-1) - (F-1)
        # we just ignore that (F-1) and leave those entries empty in the table
        # the entries are:
        # (0,0)   (0,1)   ... (0,F-2)  (0,F-1)  < where the blank F-1 entries are >
        # (1,1-F) (1,2-F) ... (1,-1)   (1,0) (1,1) (1,2) ... (1,F-1)
        # (2,1-F) (2,2-F) ... (2,-1)   (2,0) (2,1) ...       (2,F-1)
        # ...
        # and those are the offsets
        # we constrain the time offset to be nonnegative (by symmetry we may do this)
        # and if the time offset is zero we constrain the frequency offset to be non-negative
        int n_patch_offsets = patch_times * (2* patch_freqs -1)
        # we view this 1-d array as a matrix which has patch_times rows (slow axis) and
        # 2*patch_freqs -1 columns (fast axis) and index accordingly
        np.ndarray[double, ndim=1,mode="c"] n_poffsets_in_X = np.zeros(n_patch_offsets,dtype=np.float)
    # fastest axis is assumed to be features
    # second fastest is frequencies
    # slowest is time

    # first count how many times each of the offsets we care about is observed in
    # the fixed length example X
    _count_poffsets_in_X(< double *> n_poffsets_in_X.data, n_patch_offsets, patch_times, patch_freqs, X_n_times,
                         X_n_freqs)
    
    _patch_autocovariance(<int *> X_indices.data,
                                  <int *> rownnz.data,
                                  <int *> rowstartidx.data,
                                  <double *> feature_counts.data,
                                  <double *> feature_cooccurrences.data,

                                  X_n_rows,
                                  X_n_times,
                                  X_n_freqs,
                                  X_n_features,
                                  patch_times,
                                  patch_freqs)


    for i in range(X_n_features):
        feature_counts[i] /= n_locs
        
    cdef int f_range
    cdef double offsets_count

    for t in range(patch_times):
        if t == 0:
            f_range = patch_freqs
        else:
            f_range = 2*patch_freqs -1
        for f in range(f_range):
            offsets_count = n_poffsets_in_X[
                                              t*(2*patch_freqs -1)
                                              + f] * (<double> X_n_rows)
            
            for d0 in range(X_n_features):
                for d1 in range(X_n_features):
                    feature_cooccurrences[t*cooccur_t_stride
                                          + f*cooccur_f_stride
                                          + d0*cooccur_d0_stride
                                          + d1] /= offsets_count
                    
    #     feature_cooccurrences[i] /= n_patches

    return feature_counts, feature_cooccurrences.reshape(patch_times, 2*patch_freqs-1, X_n_features, X_n_features)
        

cdef double _patch_autocovariance(int * X_indices,
                              int * rownnz,
                              int * rowstartidx,
                              double * feature_counts,
                              double * feature_cooccurrences,
                              int X_n_rows,
                              int X_n_times,
                              int X_n_freqs,
                              int X_n_features,
                              int patch_time,
                              int patch_freq) nogil:
    cdef:
        int i,j0,j1,k, idx0, idx1, max_k, t0,f0,d0,t1,f1,d1
        int offset_count_idx, cooccur_idx
        int time_offset, freq_offset, offset_idx
        int feat_sq = X_n_features*X_n_features
        int time_stride = X_n_freqs * X_n_features
        int *cur_X_ptr = <int *> X_indices
        int *p_X_ptr = <int *> X_indices

    for i in range(X_n_rows):
        for j0 in range(rownnz[i]):
            idx0 = cur_X_ptr[j0]

            t0 = idx0/ time_stride
            f0 = (idx0 / X_n_features) % X_n_freqs
            d0 = idx0 % X_n_features
            feature_counts[d0] += 1.0
            j1 = j0
            time_offset = 0
            while j1 < rownnz[i] and time_offset < patch_time :
                idx1 = cur_X_ptr[j1]
                t1 = idx1/time_stride
                f1 = (idx1/ X_n_features) % X_n_freqs
                d1 = idx1 % X_n_features
                time_offset = t1 - t0
                freq_offset = f1 - f0
                if t1 == t0 and freq_offset < 0:
                    freq_offset = -freq_offset
                
                if time_offset > 0:
                    offset_idx = time_offset *( 2*patch_freq -1) + freq_offset + patch_freq - 1
                else:
                    offset_idx = freq_offset
                    
                if time_offset < patch_time and freq_offset < patch_freq and freq_offset > -patch_freq:
                    cooccur_idx = offset_idx * feat_sq + d0 * X_n_features + d1
                    feature_cooccurrences[cooccur_idx] += 1.0
                    if freq_offset == 0 and time_offset == 0 and d1 != d0:
                        cooccur_idx = offset_idx * feat_sq + d1 * X_n_features + d0
                        feature_cooccurrences[cooccur_idx] += 1.0
                j1 += 1

                
        cur_X_ptr += rownnz[i]
                              
