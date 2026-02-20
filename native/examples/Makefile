CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -Werror
LDFLAGS ?=

all: target_app fakecuda_unit_test crash_alloc

target_app: target_app.c ../fakecuda/fakecuda.h
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ target_app.c -L../fakecuda -lfakecuda -Wl,-rpath,'$$ORIGIN/../fakecuda'

fakecuda_unit_test: fakecuda_unit_test.c ../fakecuda/fakecuda.h
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ fakecuda_unit_test.c -L../fakecuda -lfakecuda -Wl,-rpath,'$$ORIGIN/../fakecuda'

clean:
	rm -f target_app fakecuda_unit_test crash_alloc

crash_alloc: crash_alloc.c ../fakecuda/fakecuda.h
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ crash_alloc.c -L../fakecuda -lfakecuda -Wl,-rpath,'$$ORIGIN/../fakecuda'
