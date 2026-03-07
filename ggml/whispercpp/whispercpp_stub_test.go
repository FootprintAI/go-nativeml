//go:build !whispercpp

// Copyright 2025 FootprintAI
// SPDX-License-Identifier: Apache-2.0

package whispercpp

import (
	"testing"
)

func TestStubLoadModel(t *testing.T) {
	_, err := LoadModel("/nonexistent.bin")
	if err == nil {
		t.Fatal("expected error from stub LoadModel")
	}
	if err.Error() != "whispercpp: not available, build with -tags whispercpp" {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestStubTranscribe(t *testing.T) {
	var m Model

	_, err := m.Transcribe([]float32{0.1, 0.2, 0.3})
	if err == nil {
		t.Fatal("expected error from stub Transcribe")
	}

	if m.IsMultilingual() {
		t.Fatal("expected false from stub IsMultilingual")
	}

	// Close should be a no-op
	m.Close()
}

func TestStubLangID(t *testing.T) {
	if LangID("en") != -1 {
		t.Fatal("expected -1 from stub LangID")
	}
}

func TestStubOptions(t *testing.T) {
	// Options should be constructable without error
	_ = WithGPU(true)
	_ = WithFlashAttention(false)
	_ = WithThreads(4)
	_ = WithLanguage("en")
	_ = WithTranslate(true)
	_ = WithTimestamps(true)
	_ = WithTokenTimestamps(false)
	_ = WithSingleSegment(false)
	_ = WithTemperature(0.0)
	_ = WithMaxTokens(100)
	_ = WithPrompt("test")
}
