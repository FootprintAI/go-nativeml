//go:build whispercpp

// Copyright 2025 FootprintAI
// SPDX-License-Identifier: Apache-2.0

package whispercpp

// ModelOption configures model loading.
type ModelOption func(*modelConfig)

type modelConfig struct {
	useGPU    bool
	flashAttn bool
}

func defaultModelConfig() modelConfig {
	return modelConfig{
		useGPU:    true,
		flashAttn: false,
	}
}

// WithGPU enables or disables GPU acceleration.
func WithGPU(enabled bool) ModelOption {
	return func(c *modelConfig) { c.useGPU = enabled }
}

// WithFlashAttention enables or disables flash attention.
func WithFlashAttention(enabled bool) ModelOption {
	return func(c *modelConfig) { c.flashAttn = enabled }
}

// TranscribeOption configures transcription.
type TranscribeOption func(*transcribeConfig)

type transcribeConfig struct {
	threads         int
	language        string
	translate       bool
	timestamps      bool
	tokenTimestamps bool
	singleSegment   bool
	temperature     float32
	maxTokens       int
	prompt          string
}

func defaultTranscribeConfig() transcribeConfig {
	return transcribeConfig{
		threads:         4,
		language:        "auto",
		translate:       false,
		timestamps:      true,
		tokenTimestamps: false,
		singleSegment:   false,
		temperature:     0.0,
		maxTokens:       0,
		prompt:          "",
	}
}

// WithThreads sets the number of CPU threads for inference.
func WithThreads(n int) TranscribeOption {
	return func(c *transcribeConfig) { c.threads = n }
}

// WithLanguage sets the language for transcription (e.g. "en", "de", "auto").
func WithLanguage(lang string) TranscribeOption {
	return func(c *transcribeConfig) { c.language = lang }
}

// WithTranslate enables translation to English.
func WithTranslate(enabled bool) TranscribeOption {
	return func(c *transcribeConfig) { c.translate = enabled }
}

// WithTimestamps enables or disables timestamps in output.
func WithTimestamps(enabled bool) TranscribeOption {
	return func(c *transcribeConfig) { c.timestamps = enabled }
}

// WithTokenTimestamps enables token-level timestamps.
func WithTokenTimestamps(enabled bool) TranscribeOption {
	return func(c *transcribeConfig) { c.tokenTimestamps = enabled }
}

// WithSingleSegment forces output into a single segment.
func WithSingleSegment(enabled bool) TranscribeOption {
	return func(c *transcribeConfig) { c.singleSegment = enabled }
}

// WithTemperature sets the sampling temperature.
func WithTemperature(t float32) TranscribeOption {
	return func(c *transcribeConfig) { c.temperature = t }
}

// WithMaxTokens sets the maximum tokens per segment (0 = no limit).
func WithMaxTokens(n int) TranscribeOption {
	return func(c *transcribeConfig) { c.maxTokens = n }
}

// WithPrompt sets the initial prompt for the decoder.
func WithPrompt(prompt string) TranscribeOption {
	return func(c *transcribeConfig) { c.prompt = prompt }
}
