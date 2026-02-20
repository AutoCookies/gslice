CC ?= cc
CFLAGS ?= -O2 -fPIC -Wall -Wextra -Werror
LDFLAGS ?= -shared -pthread

all: libfakecuda.so

libfakecuda.so: fakecuda.c fakecuda.h
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ fakecuda.c

clean:
	rm -f libfakecuda.so
