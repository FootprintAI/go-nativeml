//go:build llamacpp

// Example: basic text generation using go-nativeml's llamacpp package.
//
// Usage:
//
//	CGO_ENABLED=1 go run -tags llamacpp ./examples/generate -model /path/to/model.gguf -prompt "Hello, world"
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/footprintai/go-nativeml/ggml/llamacpp"
)

func main() {
	modelPath := flag.String("model", "", "path to GGUF model file")
	prompt := flag.String("prompt", "Hello", "prompt text")
	maxTokens := flag.Int("max-tokens", 256, "maximum tokens to generate")
	temp := flag.Float64("temperature", 0.8, "sampling temperature")
	gpuLayers := flag.Int("gpu-layers", 999, "number of layers to offload to GPU")
	threads := flag.Int("threads", 4, "number of CPU threads")
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
		llamacpp.WithContextSize(2048),
		llamacpp.WithThreads(*threads),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error creating context: %v\n", err)
		os.Exit(1)
	}
	defer ctx.Close()

	fmt.Printf("Prompt: %s\n\n", *prompt)

	err = ctx.GenerateStream(*prompt, func(token string) bool {
		fmt.Print(token)
		return true
	},
		llamacpp.WithMaxTokens(*maxTokens),
		llamacpp.WithTemperature(float32(*temp)),
	)
	fmt.Println()

	if err != nil {
		fmt.Fprintf(os.Stderr, "\nerror: %v\n", err)
		os.Exit(1)
	}
}
