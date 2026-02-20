#include <stdio.h>
#include <stdlib.h>

#include "../native/fakecuda/fakecuda.h"

int main(int argc, char **argv) {
  int loops = 4;
  size_t chunk = 1024 * 1024;
  if (argc > 1) loops = atoi(argv[1]);
  if (argc > 2) chunk = (size_t)atoll(argv[2]);
  void *ptrs[1024] = {0};
  int success = 0;
  for (int i = 0; i < loops; i++) {
    cudaError_t rc = cudaMalloc(&ptrs[i], chunk);
    if (rc != cudaSuccess) {
      printf("malloc_failed_at=%d code=%d\n", i, rc);
      break;
    }
    success++;
    printf("malloc_ok=%d\n", i);
  }
  for (int i = 0; i < success; i++) {
    if (cudaFree(ptrs[i]) == cudaSuccess) {
      printf("free_ok=%d\n", i);
    }
  }
  return 0;
}
