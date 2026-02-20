# GPU Slice Quota Manager v1.0.0

## Build and test
```bash
make build
make test
```

## Run server
```bash
GPUSLICE_IPC_TOKEN=change-me GPUSLICE_IPC_TOKEN_REQUIRED=1 make run-server
```

## End-to-end demo (LD_PRELOAD + quota enforcement + metrics)
```bash
make demo
```

## CLI production run helper
```bash
# gpuslice run <limit_bytes> <ttl_seconds> <preload_lib> <ld_library_path> <command...>
GPUSLICE_IPC_TOKEN=change-me ./bin/gpuslice run 134217728 120 \
  $(pwd)/native/interceptor/libgpuslice.so \
  $(pwd)/native/fakecuda \
  ./native/examples/target_app
```
This allocates a session, runs the command under `LD_PRELOAD`, and releases the session on exit (best effort).

## Security and auth
- IPC token is passed as `GPUSLICE_IPC_TOKEN` and validated server-side.
- Set `GPUSLICE_IPC_TOKEN_REQUIRED=1` in production.

## Recovery hardening
- Session TTL default: 15m (`GPUSLICE_DEFAULT_TTL`).
- Background recovery loop (`GPUSLICE_RECOVERY_INTERVAL`, default 2s):
  - expires stale sessions and zeros used bytes,
  - reconciles orphaned allocations for dead PIDs using `/proc/<pid>`.

## Interceptor constraints
- `GPUSLICE_SESSION` must match `[A-Za-z0-9_-]` for enforcement mode.
- Missing session -> interceptor passes through to underlying CUDA calls.

## Metrics
`/metrics` includes:
- `gpuslice_sessions_active`
- `gpuslice_used_bytes_total`
- `gpuslice_alloc_events_total`
- `gpuslice_denied_alloc_total`
- `gpuslice_recovered_bytes_total`

Enable per-session debug metric labels only with `GPUSLICE_METRICS_DEBUG=1`.

## Packaging
- `make install PREFIX=/usr/local`
- Example systemd unit: `packaging/gpusliced.service`
