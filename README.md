# GPU Slice Quota Manager (MVP)

## Build
```bash
make build
```

## Test
```bash
make test
```

## Fake CUDA demo (Phase 3)
```bash
make demo-fake
```

Expected output pattern:
- `ALLOC_OK iteration 0`
- `ALLOC_OK iteration 1`
- `ALLOC_FAIL at iteration 2 code=2`

## Native layout
- `native/fakecuda`: `libfakecuda.so`
- `native/examples/target_app`: app linked against `libfakecuda.so`
- `native/examples/fakecuda_unit_test`: C unit-style test binary

## Server run (control plane)
```bash
make run-server
```
