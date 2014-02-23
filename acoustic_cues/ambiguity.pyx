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



def featurecov_dot_datum(
    np.ndarray[ndim=1,dtype=double,mode="c"] featurecov,
    int n_parts,
    np.ndarray[ndim=1,dtype=int,mode="c"] X_indices,
        int x_nnz, int D):
    """
    D is the total dimensions
    """
    cdef np.ndarray[ndim=1,dtype=double,mode="c"] z = np.zeros(D,dtype=np.float)
    _featurecov_dot_datum(<double *> featurecov.data,
                          n_parts,
                          <int *> X_indices.data,
                          x_nnz,
                          <double *> z.data,
                          D)
    return z
    

cdef void _featurecov_dot_datum(double *featurecov,
                                int n_parts,
                                int * X_indices,
                                int x_nnz,
                                double * z, int D) nogil:
    cdef:
        double * featurecov_ptr = <double *> featurecov
        double * z_ptr = <double *> z
        int * X_ptr = <int *> X_indices
        int i,j, num_use_x_entries, hit_end, x_id
        int z_idx = 0


    x_id = 0
    while z_idx < D and x_id < x_nnz:
        if (X_ptr[0]/n_parts)  == (z_idx/n_parts):
            # two steps:
            # 1) figure out how many more x_indices are in the current z_block (at most n_parts)
            # 2)
            num_use_x_entries = 0
            hit_end = 0 
            while hit_end == 0:
                num_use_x_entries +=1
                if num_use_x_entries == x_nnz - x_id:
                    hit_end = 1
                elif X_ptr[num_use_x_entries]/n_parts > X_ptr[0]/n_parts:
                    hit_end = 1
                    
            # now that the end has been hit we then add things up
            # we reset the location of the pointer to the matrix
            # and we then to sparse addition
            featurecov_ptr = <double *> featurecov
            for i in range(n_parts):
                z_ptr[i] = 0.0
                for j in range(num_use_x_entries):
                    z_ptr[i] += featurecov_ptr[X_ptr[j]-z_idx]
                    
                if i < n_parts-1:
                    featurecov_ptr += n_parts

            j=0
            while x_id < x_nnz and j < num_use_x_entries:
                j+=1
                x_id += 1
                X_ptr += 1

        # now we update the indices
        z_idx += n_parts
        if z_idx < D:
            z_ptr += n_parts

    
def get_top_exemplar_neighbors(
        np.ndarray[ndim=1,dtype=int,mode="c"] X_indices,
        np.ndarray[ndim=1,dtype=int,mode="c"] rownnz,
        np.ndarray[ndim=1,dtype=int,mode="c"] rowstartidx,
        np.ndarray[ndim=1,dtype=double,mode="c"] featurecov,
        int n_parts,
        int D,
        np.ndarray[ndim=1,dtype=double,mode="c"] trans_background,
        int n_neighbors):
    """
    trans_background:
        (l*I + cov^2)^{-1} cov\cdot background basically the fixed
        portion of the LDA classifier

    featurecov:
        transformation matrix normalizing for the variance
    """
    cdef:
        int n_data = rownnz.shape[0]
        np.ndarray[ndim=1,dtype=int,mode="c"] neighbor_indices = np.zeros(n_neighbors * n_data,dtype =np.intc)
        np.ndarray[ndim=1,dtype=double,mode="c"] neighbor_scores = np.zeros(n_neighbors * n_data,dtype =np.float)
        np.ndarray[ndim=1,dtype=double,mode="c"] classifier_vec = np.zeros(D,dtype = np.float)
        np.ndarray[ndim=1,dtype=double,mode="c"] classifier_scores = np.zeros(n_data,dtype = np.float)
        


    _get_top_exemplar_neighbors(<int *> X_indices.data,
                                <int *> rownnz.data,
                                <int *> rowstartidx.data,
                                <double *> featurecov.data,
                                n_parts,
                                D, n_data, n_neighbors,
                                <double *> trans_background.data,
                                <int *> neighbor_indices.data,
                                <double *> neighbor_scores.data,
                                <double *> classifier_vec.data,
                                <double *> classifier_scores.data)
    
    return neighbor_indices.reshape(n_data,n_neighbors), neighbor_scores.reshape(n_data,n_neighbors)
        
cdef void _get_top_exemplar_neighbors(int * X_indices,
                                      int * rownnz,
                                      int * rowstartidx,
                                      double * featurecov,
                                      int n_parts,
                                      int D, int n_data, int n_neighbors,
                                      double * trans_background,
                                      int * neighbor_indices,
                                      double * neighbor_scores,
                                      double * classifier_vec,
                                      double * classifier_scores) nogil:
    cdef:
        int * X_ptr = <int *> X_indices
        int * rownnz_ptr = <int *> rownnz
        int * neighbor_idx_ptr = <int *> neighbor_indices
        double * neighbor_score_ptr = <double *> neighbor_scores
        double * classifier_score_ptr = <double *> classifier_scores
        int i,j,idx,n

    for n in range(n_data):
        for i in range(D):
            classifier_vec[i] = 0.0

        _featurecov_dot_datum(featurecov,
                              n_parts,
                              X_ptr,
                            rownnz_ptr[0],
                              classifier_vec, D)
        for i in range(D):
            classifier_vec[i] -= trans_background[i]
            
        _sparse_dot(X_indices, rownnz,
                    rowstartidx, n_data, classifier_vec, 
                    classifier_scores)

        classifier_score_ptr = <double *> classifier_scores
        for i in range(n_data):
            if i < n_neighbors:
                neighbor_idx_ptr[i] = i
                neighbor_score_ptr[i] = classifier_score_ptr[0]
            else:
                idx = -1
                for j in range(n_neighbors):
                    if classifier_score_ptr[0] > neighbor_score_ptr[j]:
                        if idx > -1:
                            if neighbor_score_ptr[j] < neighbor_score_ptr[idx]:
                                j = idx
                        else:
                            j = idx
                            
                if idx > -1:
                    neighbor_idx_ptr[idx] = i
                    neighbor_score_ptr[idx] = classifier_score_ptr[0]

            classifier_score_ptr += 1
                        
                        
        if n < n_data-1:
            neighbor_idx_ptr += n_neighbors
            neighbor_score_ptr += n_neighbors
            
        
        X_ptr += rownnz_ptr[0]
        rownnz_ptr += 1



cdef void _sparse_dot(int *X_indices_ptr, int *rownnz,
                      int* rowstartidx, int X_n_rows, double *w_ptr, double *z_ptr) nogil:
    """
    """
    cdef int i,j, idx
    cdef int *cur_X_ptr = <int *>X_indices_ptr
    for i in range(X_n_rows):
        z_ptr[i] = 0.0
        for j in range(rownnz[i]):
            idx = cur_X_ptr[j]
            z_ptr[i] += w_ptr[idx]

        cur_X_ptr += rownnz[i]

                                      
