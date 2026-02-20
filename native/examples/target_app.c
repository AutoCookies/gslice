#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#include "../fakecuda/fakecuda.h"

static long long parse_env_ll(const char *name, long long fallback) {
  const char *raw = getenv(name);
  if (raw == NULL || raw[0] == '\0') {
    return fallback;
  }
  errno = 0;
  char *end = NULL;
  long long value = strtoll(raw, &end, 10);
  if (errno != 0 || end == raw || (end != NULL && *end != '\0') || value <= 0) {
    return fallback;
  }
  return value;
}

int main(void) {
  const long long alloc_bytes_ll = parse_env_ll("ALLOC_BYTES", 64LL * 1024LL * 1024LL);
  const long long iterations_ll = parse_env_ll("ITERATIONS", 10);

  size_t alloc_bytes = (size_t)alloc_bytes_ll;
  int iterations = (int)iterations_ll;
  void **ptrs = (void **)calloc((size_t)iterations, sizeof(void *));
  if (ptrs == NULL) {
    fprintf(stderr, "ERR calloc_failed\n");
    return 1;
  }

  int success = 0;
  for (int i = 0; i < iterations; i++) {
    int rc = cudaMalloc(&ptrs[i], alloc_bytes);
    if (rc != FAKECUDA_SUCCESS) {
      printf("ALLOC_FAIL at iteration %d code=%d\n", i, rc);
      break;
    }
    success++;
    printf("ALLOC_OK iteration %d\n", i);
  }

  for (int i = 0; i < success; i++) {
    int rc = cudaFree(ptrs[i]);
    if (rc == FAKECUDA_SUCCESS) {
      printf("FREE_OK iteration %d\n", i);
    } else {
      printf("FREE_FAIL iteration %d code=%d\n", i, rc);
    }
  }

  free(ptrs);
  return 0;
}
