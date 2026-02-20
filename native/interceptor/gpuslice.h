#ifndef GPUSLICE_H
#define GPUSLICE_H

#include <stddef.h>

#define GPUSLICE_SUCCESS 0
#define GPUSLICE_ERROR_OUT_OF_MEMORY 2
#define GPUSLICE_ERROR_INVALID_VALUE 3

int cudaMalloc(void **ptr, size_t size);
int cudaFree(void *ptr);

#endif
