#!python3
#cython: boundscheck=False
#cython: wraparound=False
#cython: optimize.useswitch=True
#cython: cdivision=True

cimport numpy as np
import numpy as np
#np.import_array()

from libc.stdlib cimport rand, RAND_MAX, malloc, free
#from libc.math cimport log
DTYPE = np.int
ctypedef np.int_t DTYPE_t
#DTYPE32 = np.int32
#ctypedef np.int32_t DTYPE32_t

DTYPE2 = np.float
ctypedef np.float_t DTYPE2_t
#from cpython cimport PyObject, Py_INCREF


cdef extern from "nicerdicer.h":
    unsigned short max(unsigned short *rho, int state, int steps)
    double * walker_m(unsigned int xx[3], unsigned long steps, unsigned int offset, float p, float tnl, float tl_1, float tl_2, float l_1, float l_2, float r_12 , float r_21, float spont, int seed)
    void walker_d(unsigned short *walks, double *times, unsigned int steps, unsigned int offset, float p, float tnl, float tl_1, float tl_2, float l_1, float l_2, float r_12 , float r_21, float spont, int seed)

cdef np.ndarray[DTYPE2_t, ndim=3] mkDist(unsigned short *rho, double *times, int steps):
    cdef int i
    print max(rho,0,steps)+1, max(rho,1,steps)+1, max(rho,2,steps)+1
    cdef np.ndarray[DTYPE2_t, ndim=3] RHO_mat = np.zeros((max(rho,0,steps)+1, max(rho,1,steps)+1, max(rho,2,steps)+1), dtype=DTYPE2)
    for i in xrange(steps):
        RHO_mat[rho[i*3+0],rho[i*3+1],rho[i*3+2]] += times[i]/steps 

    return RHO_mat

def moments(unsigned long steps, np.ndarray[DTYPE_t, ndim=1] xx, unsigned  int offset, float p, float tnl, 
            float tl_1, float tl_2, float l_1, float l_2, float r_12 , float r_21, float spont, int seed):
    cdef unsigned int i
    cdef unsigned int x[3] 
    x[0] = xx[0]; x[1] = xx[1]; x[2] = xx[2];
    cdef double weights[8]
    cdef double *charac 
    charac = walker_m(x, steps, offset,  p, tnl, tl_1,  tl_2,  l_1,  l_2,  r_12 ,  r_21,  spont, seed)

    cdef np.ndarray[DTYPE_t, ndim=1] x_np = np.array([x[0], x[1], x[2]])
    cdef np.ndarray[DTYPE2_t, ndim=1] charac_np = np.array([charac[0], charac[1], charac[2], charac[3], charac[4], charac[5]])

    return charac[0], charac[1], charac[2], charac[3], charac[4], charac[5] ,x_np 

def density(unsigned int steps, np.ndarray[DTYPE_t, ndim=1] xx, unsigned  int offset, float p, float tnl, float tl_1, float tl_2, float l_1, float l_2, float r_12 , float r_21, float spont, int seed):
    cdef unsigned short *walks = <unsigned short *>malloc(steps * 3 * sizeof(unsigned short))
    cdef double *times = <double *>malloc(steps * sizeof(double))
    cdef unsigned int i
    cdef np.ndarray[DTYPE_t, ndim=1] x = np.zeros(3, dtype=DTYPE)
    walks[0] = xx[0]; walks[1] = xx[1]; walks[2] = xx[2]; 
    
    walker_d(walks, times, steps, offset,  p, tnl,  tl_1,  tl_2,  l_1,  l_2,  r_12 ,  r_21,  spont, seed)
    cdef np.ndarray[DTYPE2_t, ndim=3] RHO_mat = mkDist(walks, times, steps)#, (max(walks,0,steps)+1 , max(walks,1,steps)+1)
    x[0] = walks[steps*3-3]; x[1] = walks[steps*3-2]; x[2] = walks[steps*3-1];
    free(walks)
    free(times)

    return RHO_mat, x
