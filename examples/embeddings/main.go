//go:build llamacpp

// Example: generate embeddings using go-nativeml's llamacpp package.
//
// Usage:
//
//	CGO_ENABLED=1 go run -tags llamacpp ./examples/embeddings -model /path/to/model.gguf -text "Hello, world"
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/footprintai/go-nativeml/ggml/llamacpp"
)

func main() {
	modelPath := flag.String("model", "", "path to GGUF model file")
	text := flag.String("text", "Hello, world", "text to embed")
	gpuLayers := flag.Int("gpu-layers", 999, "number of layers to offload to GPU")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "error: -model is required")
		flag.Usage()
		os.Exit(1)
	}

	llamacpp.Init()
	defer llamacpp.Shutdown()

	model, err := llamacpp.LoadModel(*modelPath,
		llamacpp.WithGPULayers(*gpuLayers),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error loading model: %v\n", err)
		os.Exit(1)
	}
	defer model.Close()

	ctx, err := model.NewContext(
		llamacpp.WithContextSize(512),
		llamacpp.WithEmbeddings(),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error creating context: %v\n", err)
		os.Exit(1)
	}
	defer ctx.Close()

	embeddings, err := ctx.GetEmbeddings(*text)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error getting embeddings: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Text: %s\n", *text)
	fmt.Printf("Embedding dim: %d\n", len(embeddings))
	fmt.Printf("First 5 values: %v\n", embeddings[:min(5, len(embeddings))])
}
