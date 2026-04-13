//go:build !llamacpp

// Copyright 2025 FootprintAI
// SPDX-License-Identifier: Apache-2.0

// Package llamacpp provides stub implementations when llama.cpp is not available.
// Build with -tags llamacpp to enable the real implementation.
package llamacpp

import "errors"

var errNotAvailable = errors.New("llamacpp: not available, build with -tags llamacpp")

// Init is a no-op without the llamacpp build tag.
func Init() {}

// Shutdown is a no-op without the llamacpp build tag.
func Shutdown() {}

// Model is a stub.
type Model struct{}

// LoadModel returns an error without the llamacpp build tag.
func LoadModel(path string, opts ...ModelOption) (*Model, error) {
	return nil, errNotAvailable
}

// NewContext returns an error without the llamacpp build tag.
func (m *Model) NewContext(opts ...ContextOption) (*Context, error) {
	return nil, errNotAvailable
}

// Close is a no-op.
func (m *Model) Close() {}

// EmbeddingSize returns 0.
func (m *Model) EmbeddingSize() int { return 0 }

// Context is a stub.
type Context struct{}

// Tokenize returns an error without the llamacpp build tag.
func (c *Context) Tokenize(text string) ([]int32, error) {
	return nil, errNotAvailable
}

// Generate returns an error without the llamacpp build tag.
func (c *Context) Generate(prompt string, opts ...GenerateOption) (string, error) {
	return "", errNotAvailable
}

// GenerateStream returns an error without the llamacpp build tag.
func (c *Context) GenerateStream(prompt string, cb func(token string) bool, opts ...GenerateOption) error {
	return errNotAvailable
}

// GetEmbeddings returns an error without the llamacpp build tag.
func (c *Context) GetEmbeddings(text string) ([]float32, error) {
	return nil, errNotAvailable
}

// Close is a no-op.
func (c *Context) Close() {}

// ModelOption configures model loading (stub).
type ModelOption func(*modelConfig)
type modelConfig struct{}

func WithGPULayers(n int) ModelOption   { return func(*modelConfig) {} }
func WithMMap(enabled bool) ModelOption { return func(*modelConfig) {} }
func WithMLock(enabled bool) ModelOption { return func(*modelConfig) {} }

// ContextOption configures context creation (stub).
type ContextOption func(*contextConfig)
type contextConfig struct{}

func WithContextSize(n int) ContextOption   { return func(*contextConfig) {} }
func WithBatchSize(n int) ContextOption     { return func(*contextConfig) {} }
func WithThreads(n int) ContextOption       { return func(*contextConfig) {} }
func WithEmbeddings() ContextOption         { return func(*contextConfig) {} }
func WithTypeK(t GGMLType) ContextOption    { return func(*contextConfig) {} }
func WithTypeV(t GGMLType) ContextOption    { return func(*contextConfig) {} }

// GenerateOption configures text generation (stub).
type GenerateOption func(*generateConfig)
type generateConfig struct{}

func WithMaxTokens(n int) GenerateOption         { return func(*generateConfig) {} }
func WithTemperature(t float32) GenerateOption    { return func(*generateConfig) {} }
func WithTopK(k int) GenerateOption               { return func(*generateConfig) {} }
func WithTopP(p float32) GenerateOption           { return func(*generateConfig) {} }
func WithMinP(p float32) GenerateOption           { return func(*generateConfig) {} }
func WithRepeatPenalty(p float32) GenerateOption   { return func(*generateConfig) {} }
func WithFreqPenalty(p float32) GenerateOption     { return func(*generateConfig) {} }
func WithPresencePenalty(p float32) GenerateOption { return func(*generateConfig) {} }
func WithSeed(s int) GenerateOption                { return func(*generateConfig) {} }
