#include <stdio.h>

#include "../fakecuda/fakecuda.h"

int main(void) {
  void *a = NULL;
  void *b = NULL;

  int rc = cudaMalloc(&a, 64 * 1024 * 1024);
  if (rc != FAKECUDA_SUCCESS) {
    fprintf(stderr, "expected first alloc success got %d\n", rc);
    return 1;
  }

  rc = cudaMalloc(&b, 80 * 1024 * 1024);
  if (rc != FAKECUDA_ERROR_OUT_OF_MEMORY) {
    fprintf(stderr, "expected OOM got %d\n", rc);
    return 1;
  }

  rc = cudaFree(a);
  if (rc != FAKECUDA_SUCCESS) {
    fprintf(stderr, "expected free success got %d\n", rc);
    return 1;
  }

  rc = cudaFree(a);
  if (rc != FAKECUDA_ERROR_INVALID_VALUE) {
    fprintf(stderr, "expected invalid free got %d\n", rc);
    return 1;
  }

  rc = cudaFree((void *)0x1234);
  if (rc != FAKECUDA_ERROR_INVALID_VALUE) {
    fprintf(stderr, "expected invalid free unknown ptr got %d\n", rc);
    return 1;
  }

  printf("FAKECUDA_UNIT_TEST_OK\n");
  return 0;
}
