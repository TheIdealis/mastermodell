#ifndef TESTLIB_H
#define TESTLIB_H
#include <random>
#include <iostream>

double cython_sum(double y[6]);

unsigned short max(unsigned short *rho, int state, int steps);

double * walker_m(unsigned int x[3], unsigned long steps, unsigned int offset, float p, float tnl,
                  float tl_1, float tl_2, float l_1, float l_2, float r_12 , float r_21, float spont, int seed);

void walker_d(unsigned short *walks, double *times, unsigned int steps, unsigned int offset, float p, float tnl, float tl_1, float tl_2, float l_1, float l_2, float r_12 , float r_21, float spont, int seed);

#endif
