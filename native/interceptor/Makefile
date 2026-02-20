CC ?= cc
CFLAGS ?= -O2 -fPIC -Wall -Wextra -Werror
LDFLAGS ?= -shared -ldl -pthread

all: libgpuslice.so

libgpuslice.so: gpuslice.c gpuslice.h
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ gpuslice.c

clean:
	rm -f libgpuslice.so
