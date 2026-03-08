//go:build llamacpp

// Copyright 2025 FootprintAI
// SPDX-License-Identifier: Apache-2.0

// Package llamacpp provides Go bindings for llama.cpp via CGO.
// Build with -tags llamacpp to enable; without the tag, stub implementations are used.
package llamacpp

/*
#cgo CFLAGS: -I${SRCDIR}/third_party/include -I${SRCDIR}/third_party/ggml/include
#cgo CXXFLAGS: -std=c++17 -I${SRCDIR}/third_party/include -I${SRCDIR}/third_party/ggml/include -I${SRCDIR}/third_party/common
#cgo darwin,arm64 LDFLAGS: -L${SRCDIR}/third_party/prebuilt/darwin-arm64
#cgo darwin,amd64 LDFLAGS: -L${SRCDIR}/third_party/prebuilt/darwin-amd64
#cgo darwin LDFLAGS: -lcommon -lllama -lggml-cpu -lggml-base -lggml -lggml-blas -lggml-metal -L/usr/local/opt/libomp/lib -L/opt/homebrew/opt/libomp/lib -lomp -framework Accelerate -framework Metal -framework Foundation -lstdc++ -lm
#include <stdlib.h>
#include <stdbool.h>
#include "wrapper.h"
#include "llama.h"

// Implemented in bridge.c — wraps the Go-exported goTokenCallback.
extern bool goTokenCallbackBridge(const char* token, int len, void* user_data);
*/
import "C"

import (
	"errors"
	"fmt"
	"strings"
	"sync"
	"unsafe"
)

// Init initializes the llama.cpp backend. Call once at program start.
func Init() {
	C.llama_backend_init()
}

// Shutdown frees the llama.cpp backend. Call once at program exit.
func Shutdown() {
	C.llama_backend_free()
}

// Model wraps a loaded llama.cpp model.
type Model struct {
	c *C.struct_llama_model
}

// LoadModel loads a GGUF model from path.
func LoadModel(path string, opts ...ModelOption) (*Model, error) {
	cfg := defaultModelConfig()
	for _, o := range opts {
		o(&cfg)
	}

	params := C.llama_model_default_params()
	params.n_gpu_layers = C.int32_t(cfg.gpuLayers)
	params.use_mmap = C.bool(cfg.mmap)
	params.use_mlock = C.bool(cfg.mlock)

	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	m := C.llama_model_load_from_file(cpath, params)
	if m == nil {
		return nil, fmt.Errorf("failed to load model: %s", path)
	}
	return &Model{c: m}, nil
}

// NewContext creates an inference context from this model.
func (m *Model) NewContext(opts ...ContextOption) (*Context, error) {
	if m.c == nil {
		return nil, errors.New("model is closed")
	}

	cfg := defaultContextConfig()
	for _, o := range opts {
		o(&cfg)
	}

	params := C.llama_context_default_params()
	if cfg.contextSize > 0 {
		params.n_ctx = C.uint32_t(cfg.contextSize)
	}
	if cfg.batchSize > 0 {
		params.n_batch = C.uint32_t(cfg.batchSize)
	}
	if cfg.threads > 0 {
		params.n_threads = C.int32_t(cfg.threads)
		params.n_threads_batch = C.int32_t(cfg.threads)
	}
	params.embeddings = C.bool(cfg.embeddings)

	ctx := C.llama_init_from_model(m.c, params)
	if ctx == nil {
		return nil, errors.New("failed to create context")
	}
	return &Context{c: ctx, model: m}, nil
}

// Close frees the model resources.
func (m *Model) Close() {
	if m.c != nil {
		C.llama_model_free(m.c)
		m.c = nil
	}
}

// EmbeddingSize returns the model's embedding dimension.
func (m *Model) EmbeddingSize() int {
	if m.c == nil {
		return 0
	}
	return int(C.llama_model_n_embd(m.c))
}

// Context wraps a llama.cpp inference context.
type Context struct {
	c     *C.struct_llama_context
	model *Model
	mu    sync.Mutex
}

// Tokenize converts text to token IDs.
func (c *Context) Tokenize(text string) ([]int32, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.c == nil || c.model.c == nil {
		return nil, errors.New("context or model is closed")
	}

	vocab := C.llama_model_get_vocab(c.model.c)
	ctext := C.CString(text)
	defer C.free(unsafe.Pointer(ctext))

	// First call to get required size.
	n := C.llama_tokenize(vocab, ctext, C.int32_t(len(text)), nil, 0, true, false)
	if n == 0 {
		return []int32{}, nil
	}
	// n is negative when buffer is too small — the absolute value is the required size.
	bufSize := n
	if bufSize < 0 {
		bufSize = -bufSize
	}

	tokens := make([]C.int32_t, bufSize)
	n = C.llama_tokenize(vocab, ctext, C.int32_t(len(text)),
		&tokens[0], C.int32_t(bufSize), true, false)
	if n < 0 {
		return nil, fmt.Errorf("tokenization failed: %d", n)
	}

	result := make([]int32, n)
	for i := C.int32_t(0); i < n; i++ {
		result[i] = int32(tokens[i])
	}
	return result, nil
}

// Generate runs text generation and returns the full output.
func (c *Context) Generate(prompt string, opts ...GenerateOption) (string, error) {
	var sb strings.Builder
	err := c.GenerateStream(prompt, func(token string) bool {
		sb.WriteString(token)
		return true
	}, opts...)
	if err != nil {
		return "", err
	}
	return sb.String(), nil
}

// streamState holds the callback state passed through CGO.
type streamState struct {
	cb      func(string) bool
	stopped bool
}

//export goTokenCallback
func goTokenCallback(token *C.char, length C.int, userData unsafe.Pointer) C.int {
	state := (*streamState)(userData)
	goToken := C.GoStringN(token, length)
	if !state.cb(goToken) {
		state.stopped = true
		return 0
	}
	return 1
}

// GenerateStream runs text generation, streaming tokens via the callback.
// Return false from cb to stop generation.
func (c *Context) GenerateStream(prompt string, cb func(token string) bool, opts ...GenerateOption) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.c == nil || c.model.c == nil {
		return errors.New("context or model is closed")
	}

	cfg := defaultGenerateConfig()
	for _, o := range opts {
		o(&cfg)
	}

	cprompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cprompt))

	params := C.go_llama_default_generate_params()
	params.max_tokens = C.int(cfg.maxTokens)
	params.temperature = C.float(cfg.temperature)
	params.top_k = C.int(cfg.topK)
	params.top_p = C.float(cfg.topP)
	params.min_p = C.float(cfg.minP)
	params.repeat_penalty = C.float(cfg.repeatPenalty)
	params.freq_penalty = C.float(cfg.freqPenalty)
	params.presence_penalty = C.float(cfg.presencePenalty)
	params.seed = C.int(cfg.seed)
	params.penalty_last_n = C.int(cfg.penaltyLastN)

	state := &streamState{cb: cb}

	rc := C.go_llama_generate(
		unsafe.Pointer(c.c),
		unsafe.Pointer(c.model.c),
		cprompt,
		params,
		C.go_llama_token_callback(C.goTokenCallbackBridge),
		unsafe.Pointer(state),
	)

	if rc != 0 {
		errMsg := C.GoString(C.go_llama_last_error())
		return fmt.Errorf("generate failed: %s", errMsg)
	}
	return nil
}

// GetEmbeddings returns the embedding vector for the given text.
func (c *Context) GetEmbeddings(text string) ([]float32, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.c == nil || c.model.c == nil {
		return nil, errors.New("context or model is closed")
	}

	nEmbd := int(C.llama_model_n_embd(c.model.c))
	if nEmbd <= 0 {
		return nil, errors.New("model does not support embeddings")
	}

	out := make([]float32, nEmbd)
	ctext := C.CString(text)
	defer C.free(unsafe.Pointer(ctext))

	n := C.go_llama_embeddings(
		unsafe.Pointer(c.c),
		unsafe.Pointer(c.model.c),
		ctext,
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(nEmbd),
	)
	if n < 0 {
		errMsg := C.GoString(C.go_llama_last_error())
		return nil, fmt.Errorf("embeddings failed: %s", errMsg)
	}
	return out[:n], nil
}

// Close frees the context resources.
func (c *Context) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.c != nil {
		C.llama_free(c.c)
		c.c = nil
	}
}
