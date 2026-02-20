#define _GNU_SOURCE
#include "gpuslice.h"
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

typedef cudaError_t (*cuda_malloc_fn)(void **, size_t);
typedef cudaError_t (*cuda_free_fn)(void *);

static cuda_malloc_fn real_cudaMalloc = NULL;
static cuda_free_fn real_cudaFree = NULL;
static pthread_once_t init_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t map_mu = PTHREAD_MUTEX_INITIALIZER;

struct alloc_node {
  void *ptr;
  size_t size;
  struct alloc_node *next;
};
static struct alloc_node *alloc_head = NULL;

static const char *session_id = NULL;
static const char *socket_path = NULL;

static void init_real(void) {
  real_cudaMalloc = (cuda_malloc_fn)dlsym(RTLD_NEXT, "cudaMalloc");
  real_cudaFree = (cuda_free_fn)dlsym(RTLD_NEXT, "cudaFree");
  session_id = getenv("GPUSLICE_SESSION");
  socket_path = getenv("GPUSLICE_SOCKET");
}

static int send_ipc(const char *op, size_t bytes, int *allowed) {
  *allowed = 1;
  if (session_id == NULL || socket_path == NULL || session_id[0] == '\0' || socket_path[0] == '\0') {
    return 0;
  }
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    return -1;
  }
  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);
  if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
    close(fd);
    return -1;
  }
  char req[256];
  int n = snprintf(req, sizeof(req), "{\"pid\":%d,\"session_id\":\"%s\",\"op\":\"%s\",\"bytes\":%zu}\n", getpid(), session_id, op, bytes);
  if (n <= 0 || (size_t)n >= sizeof(req)) {
    close(fd);
    return -1;
  }
  if (write(fd, req, (size_t)n) != n) {
    close(fd);
    return -1;
  }
  char buf[256] = {0};
  ssize_t r = read(fd, buf, sizeof(buf) - 1);
  close(fd);
  if (r <= 0) {
    return -1;
  }
  if (strstr(buf, "\"allowed\":true") != NULL) {
    *allowed = 1;
  } else {
    *allowed = 0;
  }
  return 0;
}

static void map_add(void *ptr, size_t size) {
  struct alloc_node *node = (struct alloc_node *)malloc(sizeof(struct alloc_node));
  if (node == NULL) return;
  node->ptr = ptr;
  node->size = size;
  pthread_mutex_lock(&map_mu);
  node->next = alloc_head;
  alloc_head = node;
  pthread_mutex_unlock(&map_mu);
}

static size_t map_remove(void *ptr) {
  size_t out = 0;
  pthread_mutex_lock(&map_mu);
  struct alloc_node *cur = alloc_head;
  struct alloc_node *prev = NULL;
  while (cur != NULL) {
    if (cur->ptr == ptr) {
      out = cur->size;
      if (prev == NULL) alloc_head = cur->next;
      else prev->next = cur->next;
      free(cur);
      break;
    }
    prev = cur;
    cur = cur->next;
  }
  pthread_mutex_unlock(&map_mu);
  return out;
}

cudaError_t cudaMalloc(void **ptr, size_t size) {
  pthread_once(&init_once, init_real);
  if (real_cudaMalloc == NULL || ptr == NULL || size == 0) {
    return cudaErrorInvalidValue;
  }
  int allowed = 1;
  if (send_ipc("reserve", size, &allowed) != 0 || !allowed) {
    *ptr = NULL;
    return cudaErrorMemoryAllocation;
  }
  cudaError_t rc = real_cudaMalloc(ptr, size);
  if (rc != cudaSuccess) {
    int ignored;
    (void)send_ipc("release", size, &ignored);
    return rc;
  }
  map_add(*ptr, size);
  return rc;
}

cudaError_t cudaFree(void *ptr) {
  pthread_once(&init_once, init_real);
  if (real_cudaFree == NULL || ptr == NULL) {
    return cudaErrorInvalidValue;
  }
  size_t size = map_remove(ptr);
  cudaError_t rc = real_cudaFree(ptr);
  if (size > 0) {
    int ignored;
    (void)send_ipc("release", size, &ignored);
  }
  return rc;
}
