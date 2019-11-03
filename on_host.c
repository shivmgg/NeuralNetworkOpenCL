#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "on_host.h"

using namespace std;

void multiply(int M, int N, int K, float* A, float* B, float* D, float* C, int flag) {
    // std::cout<<A[0]<<B[0]<<D[0];
    for (int m=0; m<M; m++) {
        for (int n=0; n<N; n++) {
            float acc = 0.0f;
            for (int k=0; k<K; k++) {
            	//std::cout<<B[n*K + k]<<" ";
                acc += A[k*M + m] * B[n*K + k];
            }
            float temp = acc + D[n];

		    if (flag == 1) {
		      if (temp > 1) {
		        C[n*M + m] = 1;
		      }
		      else if (temp < -1) {
		        C[n*M + m] = -1;
		      }
		      else {
		        C[n*M + m] = temp;
		      }      
		    }
        }
    }	
}
