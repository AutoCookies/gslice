#define _GNU_SOURCE
#include "gpuslice.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

typedef int (*cuda_malloc_fn)(void **, size_t);
typedef int (*cuda_free_fn)(void *);

#define MAX_MSG_SIZE (64 * 1024)
#define DEFAULT_SOCK_PATH "/tmp/gpusliced.sock"

typedef struct alloc_node {
  void *ptr;
  size_t size;
  struct alloc_node *next;
} alloc_node;

static cuda_malloc_fn g_real_malloc = NULL;
static cuda_free_fn g_real_free = NULL;
static pthread_once_t g_init_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t g_map_mu = PTHREAD_MUTEX_INITIALIZER;
static alloc_node *g_alloc_head = NULL;

static int g_debug = 0;
static int g_warned_release_ipc = 0;

static void debug_log(const char *msg) {
  if (g_debug == 1 && msg != NULL) {
    fprintf(stderr, "[gpuslice] %s\n", msg);
  }
}

static void init_symbols(void) {
  g_real_malloc = (cuda_malloc_fn)dlsym(RTLD_NEXT, "cudaMalloc");
  g_real_free = (cuda_free_fn)dlsym(RTLD_NEXT, "cudaFree");
  const char *dbg = getenv("GPUSLICE_DEBUG");
  if (dbg != NULL && strcmp(dbg, "1") == 0) {
    g_debug = 1;
  }
}

static int valid_session_id(const char *session_id) {
  if (session_id == NULL || session_id[0] == '\0') {
    return 0;
  }
  for (size_t i = 0; session_id[i] != '\0'; i++) {
    unsigned char c = (unsigned char)session_id[i];
    if (!(isalnum(c) || c == '_' || c == '-')) {
      return 0;
    }
  }
  return 1;
}

static int read_full(int fd, void *buf, size_t n) {
  size_t off = 0;
  while (off < n) {
    ssize_t r = read(fd, (char *)buf + off, n - off);
    if (r <= 0) {
      return -1;
    }
    off += (size_t)r;
  }
  return 0;
}

static int write_full(int fd, const void *buf, size_t n) {
  size_t off = 0;
  while (off < n) {
    ssize_t w = write(fd, (const char *)buf + off, n - off);
    if (w <= 0) {
      return -1;
    }
    off += (size_t)w;
  }
  return 0;
}

static int parse_ok(const char *json) {
  if (strstr(json, "\"ok\":true") != NULL) {
    return 1;
  }
  return 0;
}

static int ipc_request(const char *socket_path, const char *session_id, const char *op, size_t bytes, int *ok_out) {
  if (socket_path == NULL || session_id == NULL || op == NULL || ok_out == NULL) {
    return -1;
  }

  char body[MAX_MSG_SIZE];
  int n = snprintf(body, sizeof(body), "{\"v\":1,\"op\":\"%s\",\"session_id\":\"%s\",\"pid\":%d,\"bytes\":%zu}", op, session_id, getpid(), bytes);
  if (n <= 0 || n >= (int)sizeof(body)) {
    return -1;
  }

  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    return -1;
  }

  struct timeval tv;
  tv.tv_sec = 0;
  tv.tv_usec = 200000;
  (void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  (void)setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);

  if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
    close(fd);
    return -1;
  }

  uint32_t len = htonl((uint32_t)n);
  if (write_full(fd, &len, sizeof(len)) != 0 || write_full(fd, body, (size_t)n) != 0) {
    close(fd);
    return -1;
  }

  uint32_t resp_len_be = 0;
  if (read_full(fd, &resp_len_be, sizeof(resp_len_be)) != 0) {
    close(fd);
    return -1;
  }
  uint32_t resp_len = ntohl(resp_len_be);
  if (resp_len == 0 || resp_len >= MAX_MSG_SIZE) {
    close(fd);
    return -1;
  }

  char resp[MAX_MSG_SIZE];
  memset(resp, 0, sizeof(resp));
  if (read_full(fd, resp, resp_len) != 0) {
    close(fd);
    return -1;
  }
  resp[resp_len] = '\0';
  close(fd);

  *ok_out = parse_ok(resp);
  return 0;
}

static void map_add(void *ptr, size_t size) {
  alloc_node *node = (alloc_node *)malloc(sizeof(alloc_node));
  if (node == NULL) {
    return;
  }
  node->ptr = ptr;
  node->size = size;
  pthread_mutex_lock(&g_map_mu);
  node->next = g_alloc_head;
  g_alloc_head = node;
  pthread_mutex_unlock(&g_map_mu);
}

static size_t map_remove(void *ptr, int *found) {
  size_t out = 0;
  *found = 0;
  pthread_mutex_lock(&g_map_mu);
  alloc_node *prev = NULL;
  alloc_node *cur = g_alloc_head;
  while (cur != NULL) {
    if (cur->ptr == ptr) {
      out = cur->size;
      *found = 1;
      if (prev == NULL) {
        g_alloc_head = cur->next;
      } else {
        prev->next = cur->next;
      }
      free(cur);
      break;
    }
    prev = cur;
    cur = cur->next;
  }
  pthread_mutex_unlock(&g_map_mu);
  return out;
}

static void map_reinsert(void *ptr, size_t size) {
  map_add(ptr, size);
}

int cudaMalloc(void **ptr, size_t size) {
  pthread_once(&g_init_once, init_symbols);
  if (g_real_malloc == NULL || ptr == NULL) {
    return GPUSLICE_ERROR_INVALID_VALUE;
  }

  const char *session_id = getenv("GPUSLICE_SESSION");
  const char *sock = getenv("GPUSLICE_IPC_SOCK");
  if (sock == NULL || sock[0] == '\0') {
    sock = DEFAULT_SOCK_PATH;
  }

  int enforce = (session_id != NULL && session_id[0] != '\0');
  if (enforce && !valid_session_id(session_id)) {
    debug_log("invalid session id format; failing closed");
    *ptr = NULL;
    return GPUSLICE_ERROR_OUT_OF_MEMORY;
  }

  if (enforce && size > 0) {
    int ok = 0;
    if (ipc_request(sock, session_id, "reserve", size, &ok) != 0 || !ok) {
      *ptr = NULL;
      return GPUSLICE_ERROR_OUT_OF_MEMORY;
    }
  }

  int rc = g_real_malloc(ptr, size);
  if (rc == GPUSLICE_SUCCESS && ptr != NULL && *ptr != NULL) {
    if (enforce && size > 0) {
      map_add(*ptr, size);
    }
    return rc;
  }

  if (enforce && size > 0) {
    int ok = 0;
    (void)ipc_request(sock, session_id, "release", size, &ok);
  }
  return rc;
}

int cudaFree(void *ptr) {
  pthread_once(&g_init_once, init_symbols);
  if (g_real_free == NULL) {
    return GPUSLICE_ERROR_INVALID_VALUE;
  }

  const char *session_id = getenv("GPUSLICE_SESSION");
  const char *sock = getenv("GPUSLICE_IPC_SOCK");
  if (sock == NULL || sock[0] == '\0') {
    sock = DEFAULT_SOCK_PATH;
  }

  int enforce = (session_id != NULL && session_id[0] != '\0' && valid_session_id(session_id));
  if (!enforce || ptr == NULL) {
    return g_real_free(ptr);
  }

  int found = 0;
  size_t size = map_remove(ptr, &found);
  int rc = g_real_free(ptr);
  if (!found) {
    return rc;
  }
  if (rc != GPUSLICE_SUCCESS) {
    map_reinsert(ptr, size);
    return rc;
  }

  int ok = 0;
  if (ipc_request(sock, session_id, "release", size, &ok) != 0 && g_warned_release_ipc == 0) {
    g_warned_release_ipc = 1;
    fprintf(stderr, "[gpuslice] warning: failed to release quota over IPC\n");
  }
  return rc;
}
