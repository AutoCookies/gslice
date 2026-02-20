#ifndef GPUSLICE_H
#define GPUSLICE_H
#include <stddef.h>

typedef enum {
  cudaSuccess = 0,
  cudaErrorMemoryAllocation = 2,
  cudaErrorInvalidValue = 11
} cudaError_t;

#endif
