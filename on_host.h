#ifndef __ON_HOST_H__
#define __ON_HOST_H__

#ifdef __cplusplus
extern "C" {
#endif

void multiply(int M, int N, int K, float* A, float* B, float* D, float* C, int flag);

#ifdef __cplusplus
}
#endif

#endif