GO ?= go
CC ?= cc

.PHONY: build build-go build-native test run-server demo clean lint

build: build-go build-native

build-go:
	$(GO) mod tidy
	$(GO) build -o bin/gpusliced ./cmd/gpusliced
	$(GO) build -o bin/gpuslice ./cmd/gpuslice

build-native:
	$(MAKE) -C native/fakecuda
	$(MAKE) -C native/interceptor
	$(CC) -O2 -Wall -Wextra -Werror -o examples/target_app examples/target_app.c -Lnative/fakecuda -lfakecuda

test:
	$(GO) test ./...

lint:
	gofmt -w $(shell rg --files -g '*.go')
	$(GO) vet ./...

run-server: build-go
	GPUSLICE_DB_PATH=./var/gpuslice.db GPUSLICE_IPC_SOCK=/tmp/gpusliced.sock ./bin/gpusliced

demo: build
	@set -e; \
	DB=./var/demo.db; SOCK=/tmp/gpuslice-demo.sock; LOG=./var/demo-server.log; \
	mkdir -p ./var; rm -f $$SOCK $$DB; \
	GPUSLICE_DB_PATH=$$DB GPUSLICE_IPC_SOCK=$$SOCK GPUSLICE_HTTP_ADDR=:18080 ./bin/gpusliced >$$LOG 2>&1 & \
	PID=$$!; trap 'kill $$PID >/dev/null 2>&1 || true' EXIT; sleep 1; \
	SESSION=$$(GPUSLICE_HTTP_ADDR=:18080 ./bin/gpuslice allocate 2097152 120 | python -c "import sys,json; print(json.load(sys.stdin)['session_id'])"); \
	echo "session=$$SESSION"; \
	LD_PRELOAD=$$(pwd)/native/interceptor/libgpuslice.so LD_LIBRARY_PATH=$$(pwd)/native/fakecuda GPUSLICE_SESSION=$$SESSION GPUSLICE_IPC_SOCK=$$SOCK ./examples/target_app 4 1048576; \
	GPUSLICE_HTTP_ADDR=:18080 ./bin/gpuslice status $$SESSION; \
	GPUSLICE_HTTP_ADDR=:18080 ./bin/gpuslice release $$SESSION; \
	echo "metrics: curl http://127.0.0.1:18080/metrics | rg gpuslice";

clean:
	rm -rf bin var gpuslice.db
	$(MAKE) -C native/fakecuda clean
	$(MAKE) -C native/interceptor clean
	rm -f examples/target_app
