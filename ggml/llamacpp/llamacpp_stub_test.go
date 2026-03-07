//go:build !llamacpp

// Copyright 2025 FootprintAI
// SPDX-License-Identifier: Apache-2.0

package llamacpp

import (
	"testing"
)

func TestStubLoadModel(t *testing.T) {
	_, err := LoadModel("/nonexistent.gguf")
	if err == nil {
		t.Fatal("expected error from stub LoadModel")
	}
	if err.Error() != "llamacpp: not available, build with -tags llamacpp" {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestStubContext(t *testing.T) {
	var ctx Context

	_, err := ctx.Tokenize("hello")
	if err == nil {
		t.Fatal("expected error from stub Tokenize")
	}

	_, err = ctx.Generate("hello")
	if err == nil {
		t.Fatal("expected error from stub Generate")
	}

	err = ctx.GenerateStream("hello", func(token string) bool { return true })
	if err == nil {
		t.Fatal("expected error from stub GenerateStream")
	}

	_, err = ctx.GetEmbeddings("hello")
	if err == nil {
		t.Fatal("expected error from stub GetEmbeddings")
	}

	// Close should be a no-op
	ctx.Close()
}

func TestStubModel(t *testing.T) {
	var m Model
	if m.EmbeddingSize() != 0 {
		t.Fatal("expected 0 from stub EmbeddingSize")
	}
	m.Close()
}

func TestStubOptions(t *testing.T) {
	// Options should be constructable without error
	_ = WithGPULayers(10)
	_ = WithMMap(true)
	_ = WithMLock(false)
	_ = WithContextSize(4096)
	_ = WithBatchSize(512)
	_ = WithThreads(4)
	_ = WithEmbeddings()
	_ = WithMaxTokens(100)
	_ = WithTemperature(0.7)
	_ = WithTopK(40)
	_ = WithTopP(0.9)
	_ = WithMinP(0.05)
	_ = WithRepeatPenalty(1.1)
	_ = WithFreqPenalty(0.1)
	_ = WithPresencePenalty(0.1)
	_ = WithSeed(42)
}
