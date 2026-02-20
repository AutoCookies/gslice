#include "fakecuda.h"
#include <stdlib.h>

cudaError_t cudaMalloc(void **ptr, size_t size) {
  if (ptr == NULL || size == 0) {
    return cudaErrorInvalidValue;
  }
  void *mem = malloc(size);
  if (mem == NULL) {
    *ptr = NULL;
    return cudaErrorMemoryAllocation;
  }
  *ptr = mem;
  return cudaSuccess;
}

cudaError_t cudaFree(void *ptr) {
  if (ptr == NULL) {
    return cudaErrorInvalidValue;
  }
  free(ptr);
  return cudaSuccess;
}
