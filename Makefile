GO ?= go

.PHONY: build build-go build-native build-example build-interceptor test test-native test-integration run-server demo demo-fake clean lint

build: build-go build-native

build-go:
	$(GO) mod tidy
	$(GO) build -o bin/gpusliced ./cmd/gpusliced
	$(GO) build -o bin/gpuslice ./cmd/gpuslice

build-native: build-interceptor
	$(MAKE) -C native/fakecuda
	$(MAKE) -C native/examples

build-example:
	$(MAKE) -C native/examples target_app

build-interceptor:
	$(MAKE) -C native/interceptor

test-native: build-native
	FAKECUDA_TOTAL_MEM=134217728 ./native/examples/fakecuda_unit_test

test-integration:
	$(GO) test ./internal/tests -run TestInterceptor -count=1

test: test-native
	$(GO) test ./...

lint:
	gofmt -w $(shell rg --files -g '*.go')
	$(GO) vet ./...

run-server: build-go
	GPUSLICE_DB_PATH=./var/gpuslice.db GPUSLICE_IPC_SOCK=/tmp/gpusliced.sock ./bin/gpusliced

demo: build
	@set -e; \
	DB=./var/demo.db; SOCK=/tmp/gpusliced-demo.sock; \
	mkdir -p ./var; rm -f $$SOCK $$DB; \
	GPUSLICE_DB_PATH=$$DB GPUSLICE_IPC_SOCK=$$SOCK GPUSLICE_HTTP_ADDR=:18080 ./bin/gpusliced > ./var/demo.log 2>&1 & \
	PID=$$!; trap 'kill $$PID >/dev/null 2>&1 || true' EXIT; \
	for i in 1 2 3 4 5 6 7 8 9 10; do [ -S $$SOCK ] && break; done; \
	SESSION=$$(GPUSLICE_HTTP_ADDR=:18080 ./bin/gpuslice allocate 134217728 120 | python -c "import sys,json; print(json.load(sys.stdin)['session_id'])"); \
	echo "session=$$SESSION"; \
	FAKECUDA_TOTAL_MEM=1073741824 ALLOC_BYTES=67108864 ITERATIONS=5 GPUSLICE_SESSION=$$SESSION GPUSLICE_IPC_SOCK=$$SOCK LD_PRELOAD=$$(pwd)/native/interceptor/libgpuslice.so ./native/examples/target_app

demo-fake: build-native
	@FAKECUDA_TOTAL_MEM=134217728 ALLOC_BYTES=67108864 ITERATIONS=5 ./native/examples/target_app

clean:
	rm -rf bin var gpuslice.db
	$(MAKE) -C native/fakecuda clean
	$(MAKE) -C native/examples clean
	$(MAKE) -C native/interceptor clean
