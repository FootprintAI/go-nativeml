# go-nativeml

Go bindings for C++ inference frameworks via CGO, with prebuilt static libraries for zero-dependency builds.

## Supported Frameworks

| Framework | Version | Package | Build Tag | Capabilities | Status |
|-----------|---------|---------|-----------|--------------|--------|
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | `b8220` | `ggml/llamacpp` | `llamacpp` | Text generation, embeddings, tokenization | Available |
| [whisper.cpp](https://github.com/ggerganov/whisper.cpp) | `v1.8.3` | `ggml/whispercpp` | `whispercpp` | Speech-to-text | Planned |

## Quick Start

```bash
go get github.com/footprintai/go-nativeml
```

```go
import "github.com/footprintai/go-nativeml/ggml/llamacpp"

llamacpp.Init()
defer llamacpp.Shutdown()

model, _ := llamacpp.LoadModel("model.gguf", llamacpp.WithGPULayers(999))
defer model.Close()

ctx, _ := model.NewContext(llamacpp.WithContextSize(2048), llamacpp.WithThreads(4))
defer ctx.Close()

// Streaming generation
ctx.GenerateStream("Hello, world", func(token string) bool {
    fmt.Print(token)
    return true // return false to stop early
}, llamacpp.WithMaxTokens(256), llamacpp.WithTemperature(0.8))
```

## Build Tags

| Tag | Behavior |
|-----|----------|
| _(none)_ | Stub implementations that return errors. Allows `go build` without CGO. |
| `llamacpp` | Enables CGO bindings to prebuilt llama.cpp static libraries. |

```bash
# Stub build (no CGO required)
go build ./...

# CGO build with llama.cpp
CGO_ENABLED=1 go build -tags llamacpp ./...
```

## API

### Lifecycle

```go
llamacpp.Init()          // initialize backend
llamacpp.Shutdown()      // cleanup
```

### Model

```go
model, err := llamacpp.LoadModel(path,
    llamacpp.WithGPULayers(n),  // layers to offload to GPU
)
model.Close()
model.EmbeddingSize()    // returns embedding dimension
```

### Context

```go
ctx, err := model.NewContext(
    llamacpp.WithContextSize(2048),
    llamacpp.WithThreads(4),
    llamacpp.WithEmbeddings(),     // enable embedding mode
)
ctx.Close()
```

### Generation

```go
// Blocking
text, err := ctx.Generate(prompt,
    llamacpp.WithMaxTokens(256),
    llamacpp.WithTemperature(0.8),
    llamacpp.WithTopP(0.95),
    llamacpp.WithTopK(40),
    llamacpp.WithMinP(0.05),
    llamacpp.WithRepeatPenalty(1.1),
    llamacpp.WithSeed(42),
)

// Streaming
err := ctx.GenerateStream(prompt, func(token string) bool {
    fmt.Print(token)
    return true // return false to cancel
}, llamacpp.WithMaxTokens(256))
```

### Embeddings

```go
ctx, _ := model.NewContext(llamacpp.WithContextSize(512), llamacpp.WithEmbeddings())
embeddings, err := ctx.GetEmbeddings("some text")  // []float32
```

### Tokenization

```go
tokens, err := ctx.Tokenize("some text")  // []int
```

## Examples

```bash
# Text generation
CGO_ENABLED=1 go run -tags llamacpp ./examples/generate \
    -model /path/to/model.gguf \
    -prompt "Hello, world" \
    -max-tokens 256 \
    -temperature 0.8

# Embeddings
CGO_ENABLED=1 go run -tags llamacpp ./examples/embeddings \
    -model /path/to/model.gguf \
    -text "Hello, world"
```

## Supported Platforms

| Platform | Status |
|----------|--------|
| darwin-amd64 (macOS Intel) | Prebuilt libraries included |
| darwin-arm64 (macOS Apple Silicon) | Prebuilt libraries not yet available |
| linux-amd64 | Prebuilt libraries included |

## Building Libraries from Source

For maintainers who need to rebuild the static libraries:

```bash
make build-libs          # Build for current platform
make build-libs-linux    # Build linux-amd64 via Docker
make build-libs-all      # Build native + linux-amd64
make verify              # Run stub + CGO build checks
make clean               # Remove temp build dirs
```

## Adding New Platforms

1. Build llama.cpp static libraries for the target platform
2. Place `.a` files in `third_party/llama.cpp/prebuilt/<os>-<arch>/`
3. Add a `#cgo <os>,<arch> LDFLAGS` directive in `ggml/llamacpp/llamacpp.go`

## Project Structure

```
ggml/llamacpp/         Go bindings for llama.cpp
  llamacpp.go          CGO implementation (build tag: llamacpp)
  llamacpp_stub.go     Stub implementation (default)
  options.go           Option builders for model, context, generation
  wrapper.h/.cpp       C++ bridge to llama.cpp APIs
  bridge.c             CGO callback adapter
third_party/llama.cpp/ Upstream headers + prebuilt static libraries
examples/              Usage examples (generate, embeddings)
```

## License

Apache-2.0
