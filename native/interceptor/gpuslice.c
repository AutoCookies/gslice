#define _GNU_SOURCE
#include "gpuslice.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <dlfcn.h>
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
typedef int (*cuda_malloc_managed_fn)(void **, size_t, unsigned int);
typedef int (*cuda_malloc_pitch_fn)(void **, size_t *, size_t, size_t);

#define MAX_MSG_SIZE (64 * 1024)
#define DEFAULT_SOCK_PATH "/tmp/gpusliced.sock"

typedef struct alloc_node { void *ptr; size_t size; struct alloc_node *next; } alloc_node;

static cuda_malloc_fn g_real_malloc = NULL;
static cuda_free_fn g_real_free = NULL;
static cuda_malloc_managed_fn g_real_malloc_managed = NULL;
static cuda_malloc_pitch_fn g_real_malloc_pitch = NULL;
static pthread_once_t g_init_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t g_map_mu = PTHREAD_MUTEX_INITIALIZER;
static alloc_node *g_alloc_head = NULL;
static int g_warned_release_ipc = 0;

static void init_symbols(void) {
  g_real_malloc = (cuda_malloc_fn)dlsym(RTLD_NEXT, "cudaMalloc");
  g_real_free = (cuda_free_fn)dlsym(RTLD_NEXT, "cudaFree");
  g_real_malloc_managed = (cuda_malloc_managed_fn)dlsym(RTLD_NEXT, "cudaMallocManaged");
  g_real_malloc_pitch = (cuda_malloc_pitch_fn)dlsym(RTLD_NEXT, "cudaMallocPitch");
}

static int valid_session_id(const char *session_id) {
  if (session_id == NULL || session_id[0] == '\0') return 0;
  for (size_t i = 0; session_id[i] != '\0'; i++) {
    unsigned char c = (unsigned char)session_id[i];
    if (!(isalnum(c) || c == '_' || c == '-')) return 0;
  }
  return 1;
}

static int rw_full(int fd, void *buf, size_t n, int is_write) {
  size_t off = 0;
  while (off < n) {
    ssize_t k = is_write ? write(fd, (char *)buf + off, n - off) : read(fd, (char *)buf + off, n - off);
    if (k <= 0) return -1;
    off += (size_t)k;
  }
  return 0;
}

static int parse_ok(const char *json) { return strstr(json, "\"ok\":true") != NULL; }

static int ipc_request(const char *sock, const char *session_id, const char *token, const char *op, size_t bytes, const char *ptr_id, int *ok_out) {
  if (sock == NULL || session_id == NULL || op == NULL || ok_out == NULL) return -1;
  char body[MAX_MSG_SIZE];
  const char *tok = token == NULL ? "" : token;
  const char *pidf = ptr_id == NULL ? "" : ptr_id;
  int n = snprintf(body, sizeof(body), "{\"v\":1,\"op\":\"%s\",\"session_id\":\"%s\",\"pid\":%d,\"bytes\":%zu,\"token\":\"%s\",\"ptr_id\":\"%s\"}", op, session_id, getpid(), bytes, tok, pidf);
  if (n <= 0 || n >= (int)sizeof(body)) return -1;
  int fd = socket(AF_UNIX, SOCK_STREAM, 0); if (fd < 0) return -1;
  struct timeval tv = {.tv_sec=0,.tv_usec=200000};
  (void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)); (void)setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
  struct sockaddr_un addr; memset(&addr,0,sizeof(addr)); addr.sun_family = AF_UNIX; strncpy(addr.sun_path,sock,sizeof(addr.sun_path)-1);
  if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) { close(fd); return -1; }
  uint32_t len = htonl((uint32_t)n);
  if (rw_full(fd, &len, sizeof(len), 1) != 0 || rw_full(fd, body, (size_t)n, 1) != 0) { close(fd); return -1; }
  uint32_t rbe = 0; if (rw_full(fd, &rbe, sizeof(rbe), 0) != 0) { close(fd); return -1; }
  uint32_t rlen = ntohl(rbe); if (rlen == 0 || rlen >= MAX_MSG_SIZE) { close(fd); return -1; }
  char resp[MAX_MSG_SIZE]; memset(resp,0,sizeof(resp)); if (rw_full(fd, resp, rlen, 0) != 0) { close(fd); return -1; }
  close(fd); resp[rlen]='\0'; *ok_out = parse_ok(resp); return 0;
}

static void ptr_id_from_ptr(void *ptr, char out[32]) { (void)snprintf(out, 32, "%p", ptr); }

static void map_add(void *ptr, size_t size) { alloc_node *n=(alloc_node*)malloc(sizeof(alloc_node)); if(!n)return; n->ptr=ptr; n->size=size; pthread_mutex_lock(&g_map_mu); n->next=g_alloc_head; g_alloc_head=n; pthread_mutex_unlock(&g_map_mu); }
static size_t map_remove(void *ptr, int *found) { size_t out=0; *found=0; pthread_mutex_lock(&g_map_mu); alloc_node *p=NULL,*c=g_alloc_head; while(c){ if(c->ptr==ptr){ out=c->size; *found=1; if(p)p->next=c->next; else g_alloc_head=c->next; free(c); break;} p=c; c=c->next;} pthread_mutex_unlock(&g_map_mu); return out; }

static int reserve_if_needed(const char *sid, const char *sock, const char *token, size_t size) {
  if (sid == NULL || sid[0] == '\0') return 0;
  if (!valid_session_id(sid)) return -1;
  if (size == 0) return 0;
  int ok = 0;
  if (ipc_request(sock, sid, token, "reserve", size, "", &ok) != 0 || !ok) return -1;
  return 0;
}

static void rollback_reserve(const char *sid, const char *sock, const char *token, size_t size) {
  if (sid == NULL || sid[0] == '\0' || size == 0) return;
  int ok = 0; (void)ipc_request(sock, sid, token, "release", size, "", &ok);
}

static void register_alloc(const char *sid, const char *sock, const char *token, void *ptr, size_t size) {
  if (sid == NULL || sid[0] == '\0' || ptr == NULL || size == 0) return;
  char pid[32]; ptr_id_from_ptr(ptr, pid); int ok = 0; (void)ipc_request(sock, sid, token, "alloc_register", size, pid, &ok);
}

static void unregister_alloc(const char *sid, const char *sock, const char *token, void *ptr, size_t size) {
  (void)size;
  if (sid == NULL || sid[0] == '\0' || ptr == NULL) return;
  char pid[32]; ptr_id_from_ptr(ptr, pid); int ok = 0; if (ipc_request(sock, sid, token, "alloc_unregister", 0, pid, &ok) != 0 && g_warned_release_ipc == 0) { g_warned_release_ipc = 1; fprintf(stderr, "[gpuslice] warning: failed to unregister alloc\n"); }
}

int cudaMalloc(void **ptr, size_t size) {
  pthread_once(&g_init_once, init_symbols);
  if (g_real_malloc == NULL || ptr == NULL) return GPUSLICE_ERROR_INVALID_VALUE;
  const char *sid = getenv("GPUSLICE_SESSION"); const char *sock = getenv("GPUSLICE_IPC_SOCK"); const char *token=getenv("GPUSLICE_IPC_TOKEN");
  if (sock == NULL || sock[0] == '\0') sock = DEFAULT_SOCK_PATH;
  if (reserve_if_needed(sid, sock, token, size) != 0) { *ptr = NULL; return GPUSLICE_ERROR_OUT_OF_MEMORY; }
  int rc = g_real_malloc(ptr, size);
  if (rc == GPUSLICE_SUCCESS && *ptr != NULL) { if (sid && sid[0]) { map_add(*ptr,size); register_alloc(sid,sock,token,*ptr,size);} return rc; }
  rollback_reserve(sid,sock,token,size); return rc;
}

int cudaFree(void *ptr) {
  pthread_once(&g_init_once, init_symbols);
  if (g_real_free == NULL) return GPUSLICE_ERROR_INVALID_VALUE;
  const char *sid = getenv("GPUSLICE_SESSION"); const char *sock = getenv("GPUSLICE_IPC_SOCK"); const char *token=getenv("GPUSLICE_IPC_TOKEN");
  if (sock == NULL || sock[0] == '\0') sock = DEFAULT_SOCK_PATH;
  if (ptr == NULL || sid == NULL || sid[0] == '\0' || !valid_session_id(sid)) return g_real_free(ptr);
  int found = 0; size_t size = map_remove(ptr, &found);
  int rc = g_real_free(ptr);
  if (!found) return rc;
  if (rc != GPUSLICE_SUCCESS) { map_add(ptr,size); return rc; }
  int ok = 0; (void)ipc_request(sock, sid, token, "release", size, "", &ok);
  unregister_alloc(sid,sock,token,ptr,size);
  return rc;
}

int cudaMallocManaged(void **ptr, size_t size, unsigned int flags) {
  (void)flags;
  pthread_once(&g_init_once, init_symbols);
  if (g_real_malloc_managed == NULL) return cudaMalloc(ptr, size);
  const char *sid = getenv("GPUSLICE_SESSION"); const char *sock = getenv("GPUSLICE_IPC_SOCK"); const char *token=getenv("GPUSLICE_IPC_TOKEN");
  if (sock == NULL || sock[0] == '\0') sock = DEFAULT_SOCK_PATH;
  if (reserve_if_needed(sid, sock, token, size) != 0) { *ptr = NULL; return GPUSLICE_ERROR_OUT_OF_MEMORY; }
  int rc = g_real_malloc_managed(ptr, size, flags);
  if (rc == GPUSLICE_SUCCESS && *ptr != NULL) { if (sid && sid[0]) { map_add(*ptr,size); register_alloc(sid,sock,token,*ptr,size);} return rc; }
  rollback_reserve(sid,sock,token,size); return rc;
}

int cudaMallocPitch(void **ptr, size_t *pitch, size_t width, size_t height) {
  pthread_once(&g_init_once, init_symbols);
  if (g_real_malloc_pitch == NULL) return cudaMalloc(ptr, width * height);
  size_t req = width * height;
  const char *sid = getenv("GPUSLICE_SESSION"); const char *sock = getenv("GPUSLICE_IPC_SOCK"); const char *token=getenv("GPUSLICE_IPC_TOKEN");
  if (sock == NULL || sock[0] == '\0') sock = DEFAULT_SOCK_PATH;
  if (reserve_if_needed(sid, sock, token, req) != 0) { if (ptr) *ptr=NULL; return GPUSLICE_ERROR_OUT_OF_MEMORY; }
  int rc = g_real_malloc_pitch(ptr, pitch, width, height);
  if (rc != GPUSLICE_SUCCESS || ptr == NULL || *ptr == NULL) { rollback_reserve(sid,sock,token,req); return rc; }
  size_t actual = req;
  if (pitch != NULL && *pitch > 0 && height > 0) {
    actual = (*pitch) * height;
    if (actual > req) {
      int ok = 0;
      if (ipc_request(sock, sid ? sid : "", token, "reserve", actual-req, "", &ok) != 0 || !ok) {
        (void)g_real_free(*ptr); *ptr = NULL; rollback_reserve(sid,sock,token,req); return GPUSLICE_ERROR_OUT_OF_MEMORY;
      }
    }
  }
  if (sid && sid[0]) { map_add(*ptr, actual); register_alloc(sid,sock,token,*ptr,actual); }
  return rc;
}
