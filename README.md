# GPU Slice Quota Manager (MVP)

## Build
```bash
make build
```

## Test
```bash
make test
```

## End-to-end demo with interceptor (Phase 4)
```bash
make demo
```
This runs `gpusliced`, allocates a session, then runs `native/examples/target_app` with:
- `LD_PRELOAD=native/interceptor/libgpuslice.so`
- `GPUSLICE_SESSION=<session_id>`
- `GPUSLICE_IPC_SOCK=<uds_path>`

Expected pattern:
- `ALLOC_OK iteration 0`
- `ALLOC_OK iteration 1`
- `ALLOC_FAIL at iteration 2 code=2`

## Fake CUDA-only demo (no interceptor)
```bash
make demo-fake
```

## Native layout
- `native/fakecuda`: `libfakecuda.so`
- `native/interceptor`: `libgpuslice.so`
- `native/examples/target_app`: app linked against `libfakecuda.so`
- `native/examples/fakecuda_unit_test`: C unit-style test binary

## Session ID constraint in interceptor
To keep JSON handling minimal and deterministic in C, interceptor-enforced session IDs accept only:
- `[A-Za-z0-9_-]`

If `GPUSLICE_SESSION` is present but contains other characters, allocations fail closed.

## Server run (control plane)
```bash
make run-server
```
