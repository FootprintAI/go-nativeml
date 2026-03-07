# Makefile — build prebuilt static libraries for go-nativeml
#
# For consumers:
#   go get github.com/footprintai/go-nativeml
#   go build -tags llamacpp ./...     # just works — prebuilt .a files are in the module
#   go build -tags whispercpp ./...   # whisper.cpp bindings
#
# For maintainers (rebuild .a files from source):
#   make build-libs              # Build all libraries for current platform
#   make build-libs-llama        # Build llama.cpp only
#   make build-libs-whisper      # Build whisper.cpp only
#   make build-libs-linux        # Build linux-amd64 .a files via Docker
#   make build-libs-all          # Build native + linux-amd64
#   make clean                   # Remove temp build dirs (keeps prebuilt .a + headers)

# Version sync via Go toolchain (single source of truth: version.go)
LLAMA_VERSION := $(shell go run ./cmd/versioncmd llama.cpp)
WHISPER_VERSION := $(shell go run ./cmd/versioncmd whisper.cpp)

# Platform detection
PLATFORM := $(shell go env GOOS)-$(shell go env GOARCH)

# Parallel build cores
NPROC := $(shell if which nproc > /dev/null 2>&1; then nproc; elif [ "$$(uname)" = "Darwin" ]; then sysctl -n hw.ncpu; else echo 4; fi)

# Paths — assets live inside the Go package directories
LLAMA_THIRD_PARTY := ggml/llamacpp/third_party
LLAMA_SRC := $(LLAMA_THIRD_PARTY)/src
LLAMA_PREBUILT := $(LLAMA_THIRD_PARTY)/prebuilt/$(PLATFORM)

WHISPER_THIRD_PARTY := ggml/whispercpp/third_party
WHISPER_SRC := $(WHISPER_THIRD_PARTY)/src
WHISPER_PREBUILT := $(WHISPER_THIRD_PARTY)/prebuilt/$(PLATFORM)

.PHONY: build-libs build-libs-llama build-libs-whisper build-libs-linux build-libs-all clean verify

build-libs: build-libs-llama build-libs-whisper

# Build both native platform and linux-amd64 (via Docker)
build-libs-all: build-libs build-libs-linux

# ============================================================================
# llama.cpp
# ============================================================================
build-libs-llama: $(LLAMA_PREBUILT)

$(LLAMA_PREBUILT): $(LLAMA_SRC)
	@echo "==> Building llama.cpp $(LLAMA_VERSION) for $(PLATFORM)..."
	cd $(LLAMA_SRC) && cmake -B build -DBUILD_SHARED_LIBS=OFF && \
		cmake --build build --config Release -j$(NPROC)
	@mkdir -p $(LLAMA_PREBUILT)
	find $(LLAMA_SRC)/build -name "*.a" -exec cp {} $(LLAMA_PREBUILT)/ \;
	@echo "==> Copying llama.cpp headers..."
	@mkdir -p $(LLAMA_THIRD_PARTY)/include
	cp $(LLAMA_SRC)/include/*.h $(LLAMA_THIRD_PARTY)/include/
	@mkdir -p $(LLAMA_THIRD_PARTY)/ggml/include
	cp $(LLAMA_SRC)/ggml/include/*.h $(LLAMA_THIRD_PARTY)/ggml/include/
	@mkdir -p $(LLAMA_THIRD_PARTY)/common
	cp $(LLAMA_SRC)/common/common.h $(LLAMA_THIRD_PARTY)/common/
	cp $(LLAMA_SRC)/common/sampling.h $(LLAMA_THIRD_PARTY)/common/
	@echo "==> llama.cpp $(LLAMA_VERSION) ready: $(LLAMA_PREBUILT)/"

$(LLAMA_SRC):
	@echo "==> Downloading llama.cpp $(LLAMA_VERSION)..."
	wget -qO llama.cpp.tar.gz https://github.com/ggerganov/llama.cpp/archive/refs/tags/$(LLAMA_VERSION).tar.gz
	mkdir -p $(LLAMA_SRC)
	tar xzf llama.cpp.tar.gz --strip-components=1 -C $(LLAMA_SRC)
	rm llama.cpp.tar.gz

# ============================================================================
# whisper.cpp
# ============================================================================
build-libs-whisper: $(WHISPER_PREBUILT)

$(WHISPER_PREBUILT): $(WHISPER_SRC)
	@echo "==> Building whisper.cpp $(WHISPER_VERSION) for $(PLATFORM)..."
	cd $(WHISPER_SRC) && cmake -B build -DBUILD_SHARED_LIBS=OFF && \
		cmake --build build --config Release -j$(NPROC)
	@mkdir -p $(WHISPER_PREBUILT)
	find $(WHISPER_SRC)/build -name "*.a" -exec cp {} $(WHISPER_PREBUILT)/ \;
	@echo "==> Copying whisper.cpp headers..."
	@mkdir -p $(WHISPER_THIRD_PARTY)/include
	cp $(WHISPER_SRC)/include/*.h $(WHISPER_THIRD_PARTY)/include/
	@mkdir -p $(WHISPER_THIRD_PARTY)/ggml/include
	cp $(WHISPER_SRC)/ggml/include/*.h $(WHISPER_THIRD_PARTY)/ggml/include/
	@echo "==> whisper.cpp $(WHISPER_VERSION) ready: $(WHISPER_PREBUILT)/"

$(WHISPER_SRC):
	@echo "==> Downloading whisper.cpp $(WHISPER_VERSION)..."
	wget -qO whisper.cpp.tar.gz https://github.com/ggerganov/whisper.cpp/archive/refs/tags/$(WHISPER_VERSION).tar.gz
	mkdir -p $(WHISPER_SRC)
	tar xzf whisper.cpp.tar.gz --strip-components=1 -C $(WHISPER_SRC)
	rm whisper.cpp.tar.gz

# ============================================================================
# Docker build for linux-amd64 (cross-compile from macOS)
# ============================================================================
build-libs-linux:
	@echo "==> Building linux-amd64 static libraries via Docker..."
	docker build -f Dockerfile.libs -o ./out .
	@# llama.cpp
	@mkdir -p $(LLAMA_THIRD_PARTY)/prebuilt/linux-amd64
	cp out/llama.cpp/linux-amd64/*.a $(LLAMA_THIRD_PARTY)/prebuilt/linux-amd64/
	@mkdir -p $(LLAMA_THIRD_PARTY)/include $(LLAMA_THIRD_PARTY)/ggml/include $(LLAMA_THIRD_PARTY)/common
	cp out/llama.cpp/include/*.h $(LLAMA_THIRD_PARTY)/include/
	cp out/llama.cpp/ggml/include/*.h $(LLAMA_THIRD_PARTY)/ggml/include/
	cp out/llama.cpp/common/common.h $(LLAMA_THIRD_PARTY)/common/
	cp out/llama.cpp/common/sampling.h $(LLAMA_THIRD_PARTY)/common/
	@# whisper.cpp
	@mkdir -p $(WHISPER_THIRD_PARTY)/prebuilt/linux-amd64
	cp out/whisper.cpp/linux-amd64/*.a $(WHISPER_THIRD_PARTY)/prebuilt/linux-amd64/
	@mkdir -p $(WHISPER_THIRD_PARTY)/include $(WHISPER_THIRD_PARTY)/ggml/include
	cp out/whisper.cpp/include/*.h $(WHISPER_THIRD_PARTY)/include/
	cp out/whisper.cpp/ggml/include/*.h $(WHISPER_THIRD_PARTY)/ggml/include/
	rm -rf out
	@echo "==> linux-amd64 libraries ready"

# ============================================================================
# Verification
# ============================================================================
verify:
	@echo "==> Verifying stub builds (no tags)..."
	go build ./ggml/llamacpp/...
	go build ./ggml/whispercpp/...
	@echo "==> Verifying CGO builds (with tags)..."
	CGO_ENABLED=1 go build -tags llamacpp ./ggml/llamacpp/...
	CGO_ENABLED=1 go build -tags whispercpp ./ggml/whispercpp/...
	@echo "==> Running stub tests..."
	go test ./ggml/llamacpp/...
	go test ./ggml/whispercpp/...
	@echo "==> All checks passed"

# ============================================================================
# Cleanup
# ============================================================================
clean:
	rm -rf $(LLAMA_SRC) $(WHISPER_SRC) out
