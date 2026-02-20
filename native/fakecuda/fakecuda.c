#include "fakecuda.h"

#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define FAKECUDA_DEFAULT_TOTAL_MEM (1024ULL * 1024ULL * 1024ULL)

typedef struct allocation {
  void *ptr;
  size_t size;
  struct allocation *next;
} allocation;

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static allocation *g_head = NULL;
static size_t g_total_allocated = 0;
static size_t g_total_limit = FAKECUDA_DEFAULT_TOTAL_MEM;

static void fakecuda_init(void) {
  const char *raw = getenv("FAKECUDA_TOTAL_MEM");
  if (raw == NULL || raw[0] == '\0') {
    return;
  }
  errno = 0;
  char *end = NULL;
  unsigned long long parsed = strtoull(raw, &end, 10);
  if (errno != 0 || end == raw || (end != NULL && *end != '\0')) {
    return;
  }
  if (parsed > (unsigned long long)SIZE_MAX) {
    g_total_limit = SIZE_MAX;
    return;
  }
  g_total_limit = (size_t)parsed;
}

int cudaMalloc(void **ptr, size_t size) {
  if (ptr == NULL || size == 0) {
    return FAKECUDA_ERROR_INVALID_VALUE;
  }
  pthread_once(&g_once, fakecuda_init);

  if (pthread_mutex_lock(&g_lock) != 0) {
    return FAKECUDA_ERROR_INVALID_VALUE;
  }

  if (size > g_total_limit || g_total_allocated > g_total_limit - size) {
    (void)pthread_mutex_unlock(&g_lock);
    *ptr = NULL;
    return FAKECUDA_ERROR_OUT_OF_MEMORY;
  }

  void *mem = malloc(size);
  if (mem == NULL) {
    (void)pthread_mutex_unlock(&g_lock);
    *ptr = NULL;
    return FAKECUDA_ERROR_OUT_OF_MEMORY;
  }

  allocation *node = (allocation *)malloc(sizeof(allocation));
  if (node == NULL) {
    free(mem);
    (void)pthread_mutex_unlock(&g_lock);
    *ptr = NULL;
    return FAKECUDA_ERROR_OUT_OF_MEMORY;
  }

  node->ptr = mem;
  node->size = size;
  node->next = g_head;
  g_head = node;
  g_total_allocated += size;
  *ptr = mem;

  (void)pthread_mutex_unlock(&g_lock);
  return FAKECUDA_SUCCESS;
}

int cudaFree(void *ptr) {
  if (ptr == NULL) {
    return FAKECUDA_ERROR_INVALID_VALUE;
  }
  pthread_once(&g_once, fakecuda_init);

  if (pthread_mutex_lock(&g_lock) != 0) {
    return FAKECUDA_ERROR_INVALID_VALUE;
  }

  allocation *prev = NULL;
  allocation *cur = g_head;
  while (cur != NULL) {
    if (cur->ptr == ptr) {
      if (prev == NULL) {
        g_head = cur->next;
      } else {
        prev->next = cur->next;
      }
      if (cur->size > g_total_allocated) {
        g_total_allocated = 0;
      } else {
        g_total_allocated -= cur->size;
      }
      free(cur->ptr);
      free(cur);
      (void)pthread_mutex_unlock(&g_lock);
      return FAKECUDA_SUCCESS;
    }
    prev = cur;
    cur = cur->next;
  }

  (void)pthread_mutex_unlock(&g_lock);
  return FAKECUDA_ERROR_INVALID_VALUE;
}
