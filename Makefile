# Makefile — build prebuilt static libraries for go-nativeml
#
# For consumers:
#   go get github.com/footprintai/go-nativeml
#   go build -tags llamacpp ./...    # just works — prebuilt .a files are in the module
#
# For maintainers (rebuild .a files from source):
#   make build-libs              # Build all libraries for current platform
#   make build-libs-llama        # Build llama.cpp only
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

# Paths
THIRD_PARTY := third_party
LLAMA_DIR := $(THIRD_PARTY)/llama.cpp
LLAMA_SRC := $(LLAMA_DIR)/src
LLAMA_PREBUILT := $(LLAMA_DIR)/prebuilt/$(PLATFORM)

.PHONY: build-libs build-libs-llama build-libs-linux build-libs-all clean verify

build-libs: build-libs-llama

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
	@mkdir -p $(LLAMA_DIR)/include
	cp $(LLAMA_SRC)/include/*.h $(LLAMA_DIR)/include/
	@mkdir -p $(LLAMA_DIR)/ggml/include
	cp $(LLAMA_SRC)/ggml/include/*.h $(LLAMA_DIR)/ggml/include/
	@mkdir -p $(LLAMA_DIR)/common
	cp $(LLAMA_SRC)/common/common.h $(LLAMA_DIR)/common/
	cp $(LLAMA_SRC)/common/sampling.h $(LLAMA_DIR)/common/
	@echo "==> llama.cpp $(LLAMA_VERSION) ready: $(LLAMA_PREBUILT)/"

$(LLAMA_SRC):
	@echo "==> Downloading llama.cpp $(LLAMA_VERSION)..."
	wget -qO llama.cpp.tar.gz https://github.com/ggerganov/llama.cpp/archive/refs/tags/$(LLAMA_VERSION).tar.gz
	mkdir -p $(LLAMA_SRC)
	tar xzf llama.cpp.tar.gz --strip-components=1 -C $(LLAMA_SRC)
	rm llama.cpp.tar.gz

# ============================================================================
# Docker build for linux-amd64 (cross-compile from macOS)
# ============================================================================
build-libs-linux:
	@echo "==> Building linux-amd64 static libraries via Docker..."
	docker build -f Dockerfile.libs -o ./out .
	@mkdir -p $(LLAMA_DIR)/prebuilt/linux-amd64
	cp out/llama.cpp/linux-amd64/*.a $(LLAMA_DIR)/prebuilt/linux-amd64/
	@# Copy headers if not already present
	@mkdir -p $(LLAMA_DIR)/include $(LLAMA_DIR)/ggml/include $(LLAMA_DIR)/common
	cp out/llama.cpp/include/*.h $(LLAMA_DIR)/include/
	cp out/llama.cpp/ggml/include/*.h $(LLAMA_DIR)/ggml/include/
	cp out/llama.cpp/common/common.h $(LLAMA_DIR)/common/
	cp out/llama.cpp/common/sampling.h $(LLAMA_DIR)/common/
	rm -rf out
	@echo "==> linux-amd64 libraries ready"

# ============================================================================
# Verification
# ============================================================================
verify:
	@echo "==> Verifying stub build (no tag)..."
	go build ./ggml/llamacpp/...
	@echo "==> Verifying CGO build (with tag)..."
	CGO_ENABLED=1 go build -tags llamacpp ./ggml/llamacpp/...
	@echo "==> Running stub tests..."
	go test ./ggml/llamacpp/...
	@echo "==> All checks passed"

# ============================================================================
# Cleanup
# ============================================================================
clean:
	rm -rf $(LLAMA_SRC) out
