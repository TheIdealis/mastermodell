#include "nicerdicer.h"

std::random_device rd;
std::mt19937_64 mt(rd());
//std::minstd_rand mt(rd());
//std::default_random_engine mt(rd());

double cython_sum(double y[8]){ 
    double x = y[0];
    for (unsigned int i = 1; i<8; i++){
        x += y[i];
    }
    return x;
}

unsigned short max(unsigned short *rho, int state, int steps){
    int max = rho[state];
    for (int i = 0; i<steps; i++){
        if (rho[i*3+state] > max){
            max = rho[i*3+state];
        }
    }
    return max;
}

double * walker_m(unsigned int *xx, const unsigned long steps, const unsigned int offset, 
                  const float p, const float tnl, 
                  const float tl_1, const float tl_2, const float l_1, const float l_2, const float r_12 , const float r_21, const float spont, const int seed){
    double weights[8];
    unsigned int x[3];
    x[0] = xx[0]; x[1] = xx[1];  x[2] = xx[2];
    mt.seed(seed);
    weights[0] = p;
    unsigned int i = 0;
    double num = 0;
    static double charac[6];
    double time = 0;
    double timecum = 0;

    for (i=0; i<offset; i++) {
        weights[1] = tnl*x[0] + weights[0] ;
        weights[2] = tl_1*(x[0]+x[0]*x[1])+ weights[1] ;
        weights[3] = tl_2*(x[0]+x[0]*x[2])+ weights[2] ;
        weights[4] = l_1*x[1] + weights[3] ;
        weights[5] = l_2*x[2] + weights[4] ;
        weights[6] = r_12*x[1]*(x[2] + spont)+ weights[5] ;
        weights[7] = r_21*(x[1]+spont)*x[2]+ weights[6] ;

        std::uniform_real_distribution<double> dist {0, weights[7]};
        num = dist(mt);
        if (num < weights[3]){
            if (num < weights[0]){x[0]++;}
            else if (num < weights[1]){x[0]--;}
            else if (num < weights[2]){x[0]--;x[1]++;}
            else{x[0]--;x[2]++;}
        }
        else{
            if (num < weights[4]){x[1]--;}
            else if (num < weights[5]){x[2]--;}
            else if (num < weights[6]){x[1]--; x[2]++;}
            else                      {x[1]++; x[2]--;}    
        }

    }
    for (i=0; i<steps; i++) {
        weights[1] = tnl*x[0] + weights[0] ;
        weights[2] = tl_1*(x[0]+x[0]*x[1])+ weights[1] ;
        weights[3] = tl_2*(x[0]+x[0]*x[2])+ weights[2] ;
        weights[4] = l_1*x[1] + weights[3] ;
        weights[5] = l_2*x[2] + weights[4] ;
        weights[6] = r_12*x[1]*(x[2] + spont)+ weights[5] ;
        weights[7] = r_21*(x[1]+spont)*x[2]+ weights[6] ;   
 
        time = 1/weights[7];
        timecum += time;

        charac[0] += time*x[0];
        charac[1] += time*x[1];
        charac[2] += time*x[2];
        charac[3] += time*x[1]*x[1];
        charac[4] += time*x[2]*x[2];
        charac[5] += time*x[1]*x[2];

        std::uniform_real_distribution<double> dist {0, weights[7]};
        num = dist(mt);
        if (num < weights[3]){
            if (num < weights[0]){x[0]++;}
            else if (num < weights[1]){x[0]--;}
            else if (num < weights[2]){x[0]--;x[1]++;}
            else{x[0]--;x[2]++;}
        }
        else{
            if (num < weights[4]){x[1]--;}
            else if (num < weights[5]){x[2]--;}
            else if (num < weights[6]){x[1]--; x[2]++;}
            else                      {x[1]++; x[2]--;}    
        }
    }
    charac[0] = charac[0]/timecum;
    charac[1] = charac[1]/timecum;
    charac[2] = charac[2]/timecum;
    charac[3] = charac[3]/timecum;
    charac[4] = charac[4]/timecum;
    charac[5] = charac[5]/timecum;

    xx[0] = x[0]; xx[1] = x[1];  xx[2] = x[2];
    return charac;
}


void walker_d(unsigned short *walks, double *times, unsigned int steps, unsigned int offset, float p, float tnl, 
            float tl_1, float tl_2, float l_1, float l_2, float r_12 , float r_21, float spont, int seed){
    double weights[8];
    unsigned short x[3];
    x[0] = walks[0]; x[1] = walks[1];  x[2] = walks[2];
    mt.seed(seed);
    unsigned int i=0;
    double num = 0;
    weights[0] = p;
    for (i=0; i<offset; i++) {
        weights[1] = tnl*x[0] + weights[0] ;
        weights[2] = tl_1*(x[0]+x[0]*x[1])+ weights[1] ;
        weights[3] = tl_2*(x[0]+x[0]*x[2])+ weights[2] ;
        weights[4] = l_1*x[1] + weights[3] ;
        weights[5] = l_2*x[2] + weights[4] ;
        weights[6] = r_12*x[1]*(x[2] + spont)+ weights[5] ;
        weights[7] = r_21*(x[1]+spont)*x[2]+ weights[6] ;

        std::uniform_real_distribution<double> dist {0, weights[7]};
        num = dist(mt);
        if (num < weights[3]){
            if (num < weights[0]){x[0]++;}
            else if (num < weights[1]){x[0]--;}
            else if (num < weights[2]){x[0]--;x[1]++;}
            else{x[0]--;x[2]++;}
        }
        else{
            if (num < weights[4]){x[1]--;}
            else if (num < weights[5]){x[2]--;}
            else if (num < weights[6]){x[1]--; x[2]++;}
            else {x[1]++; x[2]--; }    
        }

    }
    for (i=0; i<steps; i++) {
        const size_t idx = i * 3;
        weights[1] = tnl*x[0] + weights[0] ;
        weights[2] = tl_1*(x[0]+x[0]*x[1])+ weights[1] ;
        weights[3] = tl_2*(x[0]+x[0]*x[2])+ weights[2] ;
        weights[4] = l_1*x[1] + weights[3] ;
        weights[5] = l_2*x[2] + weights[4] ;
        weights[6] = r_12*x[1]*(x[2] + spont)+ weights[5] ;
        weights[7] = r_21*(x[1]+spont)*x[2]+ weights[6] ;

        times[i] = 1/weights[7];
        walks[idx+0] = x[0]; walks[idx+1] = x[1]; walks[idx+2] = x[2];
        std::uniform_real_distribution<double> dist {0, weights[7]};
        num = dist(mt);
        if (num < weights[3]){
            if (num < weights[0]){x[0]++;}
            else if (num < weights[1]){x[0]--;}
            else if (num < weights[2]){x[0]--;x[1]++;}
            else{x[0]--;x[2]++;}
        }
        else{
            if (num < weights[4]){x[1]--;}
            else if (num < weights[5]){x[2]--;}
            else if (num < weights[6]){x[1]--; x[2]++;}
            else {x[1]++; x[2]--; }    
        }

    }
}


