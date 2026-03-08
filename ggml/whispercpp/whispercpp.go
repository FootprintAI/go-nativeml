//go:build whispercpp

// Copyright 2025 FootprintAI
// SPDX-License-Identifier: Apache-2.0

// Package whispercpp provides Go bindings for whisper.cpp via CGO.
// Build with -tags whispercpp to enable; without the tag, stub implementations are used.
package whispercpp

/*
#cgo CFLAGS: -I${SRCDIR}/third_party/include -I${SRCDIR}/third_party/ggml/include
#cgo CXXFLAGS: -std=c++17 -I${SRCDIR}/third_party/include -I${SRCDIR}/third_party/ggml/include
#cgo darwin,arm64 LDFLAGS: -L${SRCDIR}/third_party/prebuilt/darwin-arm64
#cgo darwin,amd64 LDFLAGS: -L${SRCDIR}/third_party/prebuilt/darwin-amd64
#cgo darwin LDFLAGS: -lwhisper -lcommon -lggml-cpu -lggml-base -lggml -lggml-blas -lggml-metal -L/usr/local/opt/libomp/lib -L/opt/homebrew/opt/libomp/lib -lomp -framework Accelerate -framework Metal -framework Foundation -lstdc++ -lm
#include <stdlib.h>
#include "whisper.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"sync"
	"time"
	"unsafe"
)

// Segment represents a transcribed audio segment with timing information.
type Segment struct {
	Start time.Duration
	End   time.Duration
	Text  string
}

// Model wraps a loaded whisper.cpp model context.
type Model struct {
	c  *C.struct_whisper_context
	mu sync.Mutex
}

// LoadModel loads a whisper GGML model from path.
func LoadModel(path string, opts ...ModelOption) (*Model, error) {
	cfg := defaultModelConfig()
	for _, o := range opts {
		o(&cfg)
	}

	params := C.whisper_context_default_params()
	params.use_gpu = C.bool(cfg.useGPU)
	params.flash_attn = C.bool(cfg.flashAttn)

	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	ctx := C.whisper_init_from_file_with_params(cpath, params)
	if ctx == nil {
		return nil, fmt.Errorf("failed to load whisper model: %s", path)
	}
	return &Model{c: ctx}, nil
}

// Close frees the model resources.
func (m *Model) Close() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.c != nil {
		C.whisper_free(m.c)
		m.c = nil
	}
}

// IsMultilingual returns true if the model supports multiple languages.
func (m *Model) IsMultilingual() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.c == nil {
		return false
	}
	return C.whisper_is_multilingual(m.c) != 0
}

// Transcribe runs the full whisper pipeline on PCM audio data.
// pcmData must be 16kHz mono float32 samples.
func (m *Model) Transcribe(pcmData []float32, opts ...TranscribeOption) ([]Segment, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.c == nil {
		return nil, errors.New("model is closed")
	}
	if len(pcmData) == 0 {
		return nil, errors.New("empty audio data")
	}

	cfg := defaultTranscribeConfig()
	for _, o := range opts {
		o(&cfg)
	}

	params := C.whisper_full_default_params(C.WHISPER_SAMPLING_GREEDY)
	params.n_threads = C.int(cfg.threads)
	params.translate = C.bool(cfg.translate)
	params.no_timestamps = C.bool(!cfg.timestamps)
	params.single_segment = C.bool(cfg.singleSegment)
	params.print_special = C.bool(false)
	params.print_progress = C.bool(false)
	params.print_realtime = C.bool(false)
	params.print_timestamps = C.bool(false)
	params.token_timestamps = C.bool(cfg.tokenTimestamps)
	params.temperature = C.float(cfg.temperature)
	params.max_tokens = C.int(cfg.maxTokens)

	if cfg.language != "" {
		clang := C.CString(cfg.language)
		defer C.free(unsafe.Pointer(clang))
		params.language = clang
	}

	if cfg.prompt != "" {
		cprompt := C.CString(cfg.prompt)
		defer C.free(unsafe.Pointer(cprompt))
		params.initial_prompt = cprompt
	}

	rc := C.whisper_full(
		m.c,
		params,
		(*C.float)(unsafe.Pointer(&pcmData[0])),
		C.int(len(pcmData)),
	)
	if rc != 0 {
		return nil, fmt.Errorf("whisper_full failed with code %d", rc)
	}

	nSegments := int(C.whisper_full_n_segments(m.c))
	segments := make([]Segment, 0, nSegments)
	for i := 0; i < nSegments; i++ {
		t0 := int64(C.whisper_full_get_segment_t0(m.c, C.int(i)))
		t1 := int64(C.whisper_full_get_segment_t1(m.c, C.int(i)))
		text := C.GoString(C.whisper_full_get_segment_text(m.c, C.int(i)))
		segments = append(segments, Segment{
			Start: time.Duration(t0) * 10 * time.Millisecond,
			End:   time.Duration(t1) * 10 * time.Millisecond,
			Text:  text,
		})
	}

	return segments, nil
}

// LangID returns the language ID for the given language string.
// Returns -1 if not found.
func LangID(lang string) int {
	clang := C.CString(lang)
	defer C.free(unsafe.Pointer(clang))
	return int(C.whisper_lang_id(clang))
}
