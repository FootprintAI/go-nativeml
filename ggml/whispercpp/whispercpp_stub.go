//go:build !whispercpp

// Copyright 2025 FootprintAI
// SPDX-License-Identifier: Apache-2.0

// Package whispercpp provides stub implementations when whisper.cpp is not available.
// Build with -tags whispercpp to enable the real implementation.
package whispercpp

import (
	"errors"
	"time"
)

var errNotAvailable = errors.New("whispercpp: not available, build with -tags whispercpp")

// Segment represents a transcribed audio segment with timing information.
type Segment struct {
	Start time.Duration
	End   time.Duration
	Text  string
}

// Model is a stub.
type Model struct{}

// LoadModel returns an error without the whispercpp build tag.
func LoadModel(path string, opts ...ModelOption) (*Model, error) {
	return nil, errNotAvailable
}

// Close is a no-op.
func (m *Model) Close() {}

// IsMultilingual returns false.
func (m *Model) IsMultilingual() bool { return false }

// Transcribe returns an error without the whispercpp build tag.
func (m *Model) Transcribe(pcmData []float32, opts ...TranscribeOption) ([]Segment, error) {
	return nil, errNotAvailable
}

// LangID returns -1.
func LangID(lang string) int { return -1 }

// ModelOption configures model loading (stub).
type ModelOption func(*modelConfig)
type modelConfig struct{}

func WithGPU(enabled bool) ModelOption            { return func(*modelConfig) {} }
func WithFlashAttention(enabled bool) ModelOption { return func(*modelConfig) {} }

// TranscribeOption configures transcription (stub).
type TranscribeOption func(*transcribeConfig)
type transcribeConfig struct{}

func WithThreads(n int) TranscribeOption             { return func(*transcribeConfig) {} }
func WithLanguage(lang string) TranscribeOption       { return func(*transcribeConfig) {} }
func WithTranslate(enabled bool) TranscribeOption     { return func(*transcribeConfig) {} }
func WithTimestamps(enabled bool) TranscribeOption    { return func(*transcribeConfig) {} }
func WithTokenTimestamps(enabled bool) TranscribeOption { return func(*transcribeConfig) {} }
func WithSingleSegment(enabled bool) TranscribeOption { return func(*transcribeConfig) {} }
func WithTemperature(t float32) TranscribeOption      { return func(*transcribeConfig) {} }
func WithMaxTokens(n int) TranscribeOption            { return func(*transcribeConfig) {} }
func WithPrompt(prompt string) TranscribeOption       { return func(*transcribeConfig) {} }
