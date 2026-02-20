#ifndef FAKECUDA_H
#define FAKECUDA_H
#include <stddef.h>

typedef enum {
  cudaSuccess = 0,
  cudaErrorMemoryAllocation = 2,
  cudaErrorInvalidValue = 11
} cudaError_t;

cudaError_t cudaMalloc(void **ptr, size_t size);
cudaError_t cudaFree(void *ptr);

#endif
