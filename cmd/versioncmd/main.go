package main

import (
	"fmt"
	"os"

	gonativeml "github.com/footprintai/go-nativeml"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "usage: %s <llama.cpp|whisper.cpp>\n", os.Args[0])
		os.Exit(1)
	}
	switch os.Args[1] {
	case "llama.cpp":
		fmt.Print(gonativeml.LlamaCppVersion)
	case "whisper.cpp":
		fmt.Print(gonativeml.WhisperCppVersion)
	default:
		fmt.Fprintf(os.Stderr, "unknown library: %s\n", os.Args[1])
		os.Exit(1)
	}
}
