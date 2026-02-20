GO ?= go

.PHONY: build build-go build-native build-example test test-native run-server demo-fake clean lint

build: build-go build-native

build-go:
	$(GO) mod tidy
	$(GO) build -o bin/gpusliced ./cmd/gpusliced
	$(GO) build -o bin/gpuslice ./cmd/gpuslice

build-native:
	$(MAKE) -C native/fakecuda
	$(MAKE) -C native/examples

build-example:
	$(MAKE) -C native/examples target_app

test-native: build-native
	FAKECUDA_TOTAL_MEM=134217728 ./native/examples/fakecuda_unit_test

test: test-native
	$(GO) test ./...

lint:
	gofmt -w $(shell rg --files -g '*.go')
	$(GO) vet ./...

run-server: build-go
	GPUSLICE_DB_PATH=./var/gpuslice.db GPUSLICE_IPC_SOCK=/tmp/gpusliced.sock ./bin/gpusliced

demo-fake: build-native
	@FAKECUDA_TOTAL_MEM=134217728 ALLOC_BYTES=67108864 ITERATIONS=5 ./native/examples/target_app

clean:
	rm -rf bin var gpuslice.db
	$(MAKE) -C native/fakecuda clean
	$(MAKE) -C native/examples clean
