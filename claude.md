Principal: protobuf contract first, typed safer approach, and no backward compatibility.

This project provides CGO wrappers for C++ inference frameworks for Go.

## Structure

- `ggml/llamacpp/` — Go bindings for llama.cpp (build tag: `llamacpp`)
- `ggml/whispercpp/` — (future) Go bindings for whisper.cpp
- `third_party/llama.cpp/` — Upstream headers + prebuilt static libraries (keep upstream layout untouched)

## Build Tags

- Default (no tag): stub implementations that return errors
- `llamacpp`: enables CGO bindings to prebuilt llama.cpp libraries

## Adding New Platforms

1. Build llama.cpp static libraries for the target platform
2. Place `.a` files in `third_party/llama.cpp/prebuilt/<os>-<arch>/`
3. Add CGO LDFLAGS directive in `llamacpp.go`
