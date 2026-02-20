#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include "../fakecuda/fakecuda.h"

static long long envll(const char *n, long long d){ const char *r=getenv(n); if(!r||!r[0]) return d; errno=0; char *e=NULL; long long v=strtoll(r,&e,10); if(errno||e==r||*e!='\0'||v<=0) return d; return v; }

int main(void){
  int it=(int)envll("ITERATIONS",2); size_t b=(size_t)envll("ALLOC_BYTES",67108864);
  for(int i=0;i<it;i++){ void *p=NULL; int rc=cudaMalloc(&p,b); if(rc!=0){ printf("ALLOC_FAIL at iteration %d code=%d\n",i,rc); return 0; } printf("ALLOC_OK iteration %d\n",i); }
  _Exit(0);
}
