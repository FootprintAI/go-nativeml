# go-nativeml

Go bindings for C++ inference frameworks via CGO, with prebuilt static libraries for zero-dependency builds.

## Why go-nativeml?

| Approach | Build complexity | Runtime dependency | `go mod vendor` | GPU support |
|----------|------------------|--------------------|-----------------|-------------|
| **go-nativeml (this project)** | `go build -tags llamacpp` — just works | None — static linking | Works via `go:embed` | Metal, CPU |
| HTTP/subprocess wrapper (e.g. ollama server) | Separate process to manage | Running server required | N/A | Depends on server |
| Dynamic linking (shared `.so`/`.dylib`) | Must install libs on every machine | Shared libs must exist at runtime | Cannot vendor native libs | Depends on build |
| Build from source at `go get` | Requires C++ toolchain + cmake on every machine | None | Fragile — source download at build | Depends on build |
| Pure Go reimplementation | Simple | None | Works | Limited/none |

**Key advantages:**

- **Zero build-time setup** — prebuilt `.a` files ship with the Go module. No cmake, no C++ toolchain, no downloads.
- **Vendoring works** — `embed.go` files use `//go:embed` to ensure `go mod vendor` captures headers and static libraries. Standard Go tooling just works.
- **No runtime dependencies** — everything is statically linked. No shared libraries to install, no server to run.
- **Stub fallback** — without build tags, all packages compile to stubs returning errors. CI, linters, and `go build ./...` work everywhere without CGO.
- **Type-safe Go API** — idiomatic option pattern, proper error handling, streaming callbacks. No shell-outs or HTTP round-trips.

## Supported Frameworks

| Framework | Version | Package | Build Tag | Capabilities | Status |
|-----------|---------|---------|-----------|--------------|--------|
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | `b8220` | `ggml/llamacpp` | `llamacpp` | Text generation, embeddings, tokenization | Available |
| [whisper.cpp](https://github.com/ggerganov/whisper.cpp) | `v1.8.3` | `ggml/whispercpp` | `whispercpp` | Speech-to-text transcription | Available |

## Quick Start

```bash
go get github.com/footprintai/go-nativeml
```

### llama.cpp — Text Generation

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

### whisper.cpp — Speech-to-Text

```go
import "github.com/footprintai/go-nativeml/ggml/whispercpp"

model, _ := whispercpp.LoadModel("ggml-base.bin", whispercpp.WithGPU(true))
defer model.Close()

// pcmData: 16kHz mono float32 samples
segments, _ := model.Transcribe(pcmData,
    whispercpp.WithLanguage("en"),
    whispercpp.WithThreads(4),
)
for _, seg := range segments {
    fmt.Printf("[%s -> %s] %s\n", seg.Start, seg.End, seg.Text)
}
```

## Build Tags

| Tag | Behavior |
|-----|----------|
| _(none)_ | Stub implementations that return errors. Allows `go build` without CGO. |
| `llamacpp` | Enables CGO bindings to prebuilt llama.cpp static libraries. |
| `whispercpp` | Enables CGO bindings to prebuilt whisper.cpp static libraries. |

```bash
# Stub build (no CGO required)
go build ./...

# CGO build with llama.cpp
CGO_ENABLED=1 go build -tags llamacpp ./...

# CGO build with whisper.cpp
CGO_ENABLED=1 go build -tags whispercpp ./...

# Both
CGO_ENABLED=1 go build -tags "llamacpp whispercpp" ./...
```

## API

### llamacpp

```go
// Lifecycle
llamacpp.Init()
llamacpp.Shutdown()

// Model
model, err := llamacpp.LoadModel(path, llamacpp.WithGPULayers(n))
model.Close()
model.EmbeddingSize()

// Context
ctx, err := model.NewContext(
    llamacpp.WithContextSize(2048),
    llamacpp.WithThreads(4),
    llamacpp.WithEmbeddings(),
)
ctx.Close()

// Generation (blocking)
text, err := ctx.Generate(prompt,
    llamacpp.WithMaxTokens(256),
    llamacpp.WithTemperature(0.8),
    llamacpp.WithTopP(0.95),
    llamacpp.WithTopK(40),
    llamacpp.WithMinP(0.05),
    llamacpp.WithRepeatPenalty(1.1),
    llamacpp.WithSeed(42),
)

// Generation (streaming)
err := ctx.GenerateStream(prompt, func(token string) bool {
    fmt.Print(token)
    return true // return false to cancel
}, llamacpp.WithMaxTokens(256))

// Embeddings
embeddings, err := ctx.GetEmbeddings("some text") // []float32

// Tokenization
tokens, err := ctx.Tokenize("some text") // []int
```

### whispercpp

```go
// Model
model, err := whispercpp.LoadModel(path,
    whispercpp.WithGPU(true),
    whispercpp.WithFlashAttention(true),
)
model.Close()
model.IsMultilingual()

// Transcription (pcmData: 16kHz mono float32)
segments, err := model.Transcribe(pcmData,
    whispercpp.WithThreads(4),
    whispercpp.WithLanguage("en"),
    whispercpp.WithTranslate(false),
    whispercpp.WithTimestamps(true),
    whispercpp.WithTokenTimestamps(false),
    whispercpp.WithSingleSegment(false),
    whispercpp.WithTemperature(0.0),
    whispercpp.WithMaxTokens(0),
    whispercpp.WithPrompt(""),
)

// Utilities
id := whispercpp.LangID("en") // language string -> ID
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

1. Build static libraries for the target platform
2. Place `.a` files in `ggml/<pkg>/third_party/prebuilt/<os>-<arch>/`
3. Add a `#cgo <os>,<arch> LDFLAGS` directive in the corresponding `.go` file

## Project Structure

```
ggml/
  llamacpp/              Go bindings for llama.cpp
    llamacpp.go          CGO implementation (build tag: llamacpp)
    llamacpp_stub.go     Stub implementation (default)
    options.go           Option builders
    wrapper.h/.cpp       C++ bridge to llama.cpp APIs
    bridge.c             CGO callback adapter
    embed.go             go:embed for vendoring support
    third_party/         Upstream headers + prebuilt .a files
  whispercpp/            Go bindings for whisper.cpp
    whispercpp.go        CGO implementation (build tag: whispercpp)
    whispercpp_stub.go   Stub implementation (default)
    options.go           Option builders
    embed.go             go:embed for vendoring support
    third_party/         Upstream headers + prebuilt .a files
examples/                Usage examples (generate, embeddings)
```

## License

Apache-2.0
