#ifndef FAKECUDA_H
#define FAKECUDA_H

#include <stddef.h>

#define FAKECUDA_SUCCESS 0
#define FAKECUDA_ERROR_OUT_OF_MEMORY 2
#define FAKECUDA_ERROR_INVALID_VALUE 3

int cudaMalloc(void **ptr, size_t size);
int cudaFree(void *ptr);

#endif
