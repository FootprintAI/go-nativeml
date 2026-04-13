# Makefile — build prebuilt static libraries for go-nativeml
#
# For consumers:
#   go get github.com/footprintai/go-nativeml
#   go build -tags llamacpp ./...     # just works — prebuilt .a files are in the module
#   go build -tags whispercpp ./...   # whisper.cpp bindings
#
# For maintainers (rebuild .a files from source):
#   make build-libs                   # Build all libraries for current platform
#   make build-libs-llama             # Build llama.cpp only
#   make build-libs-whisper           # Build whisper.cpp only
#   make build-libs-linux             # Build all linux-amd64 variants (cpu, cuda, vulkan)
#   make build-libs-linux-cpu         # Build linux-amd64 CPU only
#   make build-libs-linux-cuda        # Build linux-amd64 CUDA only
#   make build-libs-linux-vulkan      # Build linux-amd64 Vulkan only
#   make build-libs-android           # Build android-arm64 via NDK
#   make build-libs-all               # Build native + all linux + android
#   make clean                        # Remove temp build dirs (keeps prebuilt .a + headers)

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

.PHONY: build-libs build-libs-llama build-libs-whisper \
       build-libs-linux build-libs-linux-cpu build-libs-linux-cuda build-libs-linux-vulkan \
       build-libs-android build-libs-all clean verify

build-libs: build-libs-llama build-libs-whisper

# Build native + all linux variants + android
build-libs-all: build-libs build-libs-linux build-libs-android

# Build all linux-amd64 variants (cpu, cuda, vulkan)
build-libs-linux: build-libs-linux-cpu build-libs-linux-cuda build-libs-linux-vulkan

# ============================================================================
# llama.cpp
# ============================================================================
build-libs-llama: $(LLAMA_PREBUILT)

$(LLAMA_PREBUILT): $(LLAMA_SRC)
	@echo "==> Building llama.cpp $(LLAMA_VERSION) for $(PLATFORM)..."
	cd $(LLAMA_SRC) && cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF && \
		cmake --build build --config Release -j$(NPROC)
	@mkdir -p $(LLAMA_PREBUILT)
	find $(LLAMA_SRC)/build -name "*.a" -exec cp {} $(LLAMA_PREBUILT)/ \;
	find $(LLAMA_PREBUILT) -name "*.a" -exec strip -S {} \; 2>/dev/null || true
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
	cd $(WHISPER_SRC) && cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF && \
		cmake --build build --config Release -j$(NPROC)
	@mkdir -p $(WHISPER_PREBUILT)
	find $(WHISPER_SRC)/build -name "*.a" -exec cp {} $(WHISPER_PREBUILT)/ \;
	find $(WHISPER_PREBUILT) -name "*.a" -exec strip -S {} \; 2>/dev/null || true
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
# Docker build for linux-amd64 variants (cross-compile from macOS)
# ============================================================================

# Helper: build a linux-amd64 variant via Dockerfile.libs
#   $(1) = GPU_BACKEND (cpu, cuda, vulkan)
#   $(2) = prebuilt directory suffix ("" for cpu, "-cuda", "-vulkan")
define build-linux-variant
	@echo "==> Building linux-amd64$(2) static libraries via Docker ($(1))..."
	docker build -f Dockerfile.libs --build-arg GPU_BACKEND=$(1) -o ./out .
	@mkdir -p $(LLAMA_THIRD_PARTY)/prebuilt/linux-amd64$(2)
	cp out/llama.cpp/linux-amd64$(2)/*.a $(LLAMA_THIRD_PARTY)/prebuilt/linux-amd64$(2)/
	@mkdir -p $(LLAMA_THIRD_PARTY)/include $(LLAMA_THIRD_PARTY)/ggml/include $(LLAMA_THIRD_PARTY)/common
	cp out/llama.cpp/include/*.h $(LLAMA_THIRD_PARTY)/include/
	cp out/llama.cpp/ggml/include/*.h $(LLAMA_THIRD_PARTY)/ggml/include/
	cp out/llama.cpp/common/common.h $(LLAMA_THIRD_PARTY)/common/
	cp out/llama.cpp/common/sampling.h $(LLAMA_THIRD_PARTY)/common/
	@mkdir -p $(WHISPER_THIRD_PARTY)/prebuilt/linux-amd64$(2)
	cp out/whisper.cpp/linux-amd64$(2)/*.a $(WHISPER_THIRD_PARTY)/prebuilt/linux-amd64$(2)/
	@mkdir -p $(WHISPER_THIRD_PARTY)/include $(WHISPER_THIRD_PARTY)/ggml/include
	cp out/whisper.cpp/include/*.h $(WHISPER_THIRD_PARTY)/include/
	cp out/whisper.cpp/ggml/include/*.h $(WHISPER_THIRD_PARTY)/ggml/include/
	rm -rf out
	@echo "==> linux-amd64$(2) libraries ready"
endef

build-libs-linux-cpu:
	$(call build-linux-variant,cpu,)

build-libs-linux-cuda:
	$(call build-linux-variant,cuda,-cuda)

build-libs-linux-vulkan:
	$(call build-linux-variant,vulkan,-vulkan)

# ============================================================================
# Docker build for android-arm64 (cross-compile via Android NDK)
# ============================================================================
build-libs-android:
	@echo "==> Building android-arm64 static libraries via Docker (NDK)..."
	docker build -f Dockerfile.android -o ./out .
	@# llama.cpp
	@mkdir -p $(LLAMA_THIRD_PARTY)/prebuilt/android-arm64
	cp out/llama.cpp/android-arm64/*.a $(LLAMA_THIRD_PARTY)/prebuilt/android-arm64/
	@mkdir -p $(LLAMA_THIRD_PARTY)/include $(LLAMA_THIRD_PARTY)/ggml/include $(LLAMA_THIRD_PARTY)/common
	cp out/llama.cpp/include/*.h $(LLAMA_THIRD_PARTY)/include/
	cp out/llama.cpp/ggml/include/*.h $(LLAMA_THIRD_PARTY)/ggml/include/
	cp out/llama.cpp/common/common.h $(LLAMA_THIRD_PARTY)/common/
	cp out/llama.cpp/common/sampling.h $(LLAMA_THIRD_PARTY)/common/
	@# whisper.cpp
	@mkdir -p $(WHISPER_THIRD_PARTY)/prebuilt/android-arm64
	cp out/whisper.cpp/android-arm64/*.a $(WHISPER_THIRD_PARTY)/prebuilt/android-arm64/
	@mkdir -p $(WHISPER_THIRD_PARTY)/include $(WHISPER_THIRD_PARTY)/ggml/include
	cp out/whisper.cpp/include/*.h $(WHISPER_THIRD_PARTY)/include/
	cp out/whisper.cpp/ggml/include/*.h $(WHISPER_THIRD_PARTY)/ggml/include/
	rm -rf out
	@echo "==> android-arm64 libraries ready"

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
