#ifndef GPUSLICE_H
#define GPUSLICE_H

#include <stddef.h>

#define GPUSLICE_SUCCESS 0
#define GPUSLICE_ERROR_OUT_OF_MEMORY 2
#define GPUSLICE_ERROR_INVALID_VALUE 3

int cudaMalloc(void **ptr, size_t size);
int cudaFree(void *ptr);
int cudaMallocManaged(void **ptr, size_t size, unsigned int flags);
int cudaMallocPitch(void **ptr, size_t *pitch, size_t width, size_t height);

#endif
