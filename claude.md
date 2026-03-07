Principal: protobuf contract first, typed safer approach, and no backward compatibility.

This project provides CGO wrappers for C++ inference frameworks for Go.

## Structure

- `ggml/llamacpp/` — Go bindings for llama.cpp (build tag: `llamacpp`)
  - `third_party/` — Upstream headers + prebuilt static libraries
- `ggml/whispercpp/` — Go bindings for whisper.cpp (build tag: `whispercpp`)
  - `third_party/` — Upstream headers + prebuilt static libraries
- `embed.go` files use `//go:embed` to ensure `go mod vendor` includes headers and `.a` files

## Build Tags

- Default (no tag): stub implementations that return errors
- `llamacpp`: enables CGO bindings to prebuilt llama.cpp libraries
- `whispercpp`: enables CGO bindings to prebuilt whisper.cpp libraries

## Adding New Platforms

1. Build static libraries for the target platform
2. Place `.a` files in `ggml/<pkg>/third_party/prebuilt/<os>-<arch>/`
3. Add CGO LDFLAGS directive in the corresponding `.go` file
