# GPU Slice Quota Manager (MVP)

Linux-only MVP implementing a clean-architecture control plane in Go and a data plane interceptor in C.

## Architecture
- `internal/domain`: pure quota + session rules.
- `internal/application`: use-cases (`allocate`, `release`, `status`, `run`, `reserve/release bytes`).
- `internal/ports`: interfaces for store/metrics/logger/ipc.
- `internal/adapters`: file-backed store, HTTP API, UDS IPC, Prometheus-text metrics, JSON logging.
- `native/fakecuda`: mock CUDA library (`libfakecuda.so`).
- `native/interceptor`: `LD_PRELOAD` interceptor (`libgpuslice.so`) for `cudaMalloc/cudaFree`.

## Build
```bash
make build
```

## Test
```bash
make test
```

## Run server
```bash
make run-server
```

## API
- `POST /sessions` body: `{"limit_bytes":2097152,"ttl_seconds":120}`
- `GET /sessions`
- `GET /sessions/{id}`
- `DELETE /sessions/{id}`
- `GET /metrics`

## CLI
```bash
GPUSLICE_HTTP_ADDR=:8080 ./bin/gpuslice allocate 2097152 120
GPUSLICE_HTTP_ADDR=:8080 ./bin/gpuslice status <session_id>
GPUSLICE_HTTP_ADDR=:8080 ./bin/gpuslice release <session_id>
./bin/gpuslice run <session_id> $(pwd)/native/interceptor/libgpuslice.so $(pwd)/native/fakecuda ./examples/target_app 4 1048576
```

## Mock mode demo
```bash
make demo
```
Expected behavior: two allocations succeed (2 MiB total), third fails with memory-allocation error.

## Real mode (optional)
On NVIDIA systems, `libgpuslice.so` can resolve real CUDA symbols via `dlsym(RTLD_NEXT)` while keeping the same IPC quota checks.

## IPC protocol
JSON over Unix domain socket (`GPUSLICE_SOCKET`):
- request: `{"pid":123,"session_id":"...","op":"reserve|release","bytes":1048576}` + newline
- response: `{"allowed":true|false,"used_bytes":...,"remaining_bytes":...,"error":"..."}`

## Environment
- `GPUSLICE_HTTP_ADDR` default `:8080`
- `GPUSLICE_SOCKET` default `/tmp/gpuslice.sock`
- `GPUSLICE_DB_PATH` default `./gpuslice.db`
- `GPUSLICE_DEFAULT_TTL` default `10m`
- `GPUSLICE_DEFAULT_VRAM` default `67108864`
