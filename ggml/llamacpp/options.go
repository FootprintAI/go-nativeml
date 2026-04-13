//go:build llamacpp

// Copyright 2025 FootprintAI
// SPDX-License-Identifier: Apache-2.0

package llamacpp

// ModelOption configures model loading.
type ModelOption func(*modelConfig)

type modelConfig struct {
	gpuLayers int
	mmap      bool
	mlock     bool
}

func defaultModelConfig() modelConfig {
	return modelConfig{
		gpuLayers: 999, // offload all layers by default
		mmap:      true,
		mlock:     false,
	}
}

// WithGPULayers sets the number of layers to offload to GPU.
func WithGPULayers(n int) ModelOption {
	return func(c *modelConfig) { c.gpuLayers = n }
}

// WithMMap enables or disables memory-mapped file loading.
func WithMMap(enabled bool) ModelOption {
	return func(c *modelConfig) { c.mmap = enabled }
}

// WithMLock enables or disables memory locking.
func WithMLock(enabled bool) ModelOption {
	return func(c *modelConfig) { c.mlock = enabled }
}

// ContextOption configures context creation.
type ContextOption func(*contextConfig)

type contextConfig struct {
	contextSize int
	batchSize   int
	threads     int
	embeddings  bool
	typeK       GGMLType
	typeV       GGMLType
}

func defaultContextConfig() contextConfig {
	return contextConfig{
		contextSize: 0, // 0 = use model default
		batchSize:   0, // 0 = use default
		threads:     0, // 0 = use default
		embeddings:  false,
		typeK:       GGMLTypeF16, // llama.cpp default
		typeV:       GGMLTypeF16, // llama.cpp default
	}
}

// WithContextSize sets the context window size.
func WithContextSize(n int) ContextOption {
	return func(c *contextConfig) { c.contextSize = n }
}

// WithBatchSize sets the batch size for prompt processing.
func WithBatchSize(n int) ContextOption {
	return func(c *contextConfig) { c.batchSize = n }
}

// WithThreads sets the number of CPU threads for inference.
func WithThreads(n int) ContextOption {
	return func(c *contextConfig) { c.threads = n }
}

// WithEmbeddings enables embedding output mode.
func WithEmbeddings() ContextOption {
	return func(c *contextConfig) { c.embeddings = true }
}

// WithTypeK sets the data type for the KV cache K values (EXPERIMENTAL).
// Use lower-precision types (e.g. GGMLTypeQ8_0, GGMLTypeQ4_0) to reduce memory usage.
func WithTypeK(t GGMLType) ContextOption {
	return func(c *contextConfig) { c.typeK = t }
}

// WithTypeV sets the data type for the KV cache V values (EXPERIMENTAL).
// Use lower-precision types (e.g. GGMLTypeQ8_0, GGMLTypeQ4_0) to reduce memory usage.
func WithTypeV(t GGMLType) ContextOption {
	return func(c *contextConfig) { c.typeV = t }
}

// GenerateOption configures text generation.
type GenerateOption func(*generateConfig)

type generateConfig struct {
	maxTokens       int
	temperature     float32
	topK            int
	topP            float32
	minP            float32
	repeatPenalty   float32
	freqPenalty     float32
	presencePenalty float32
	seed            int
	penaltyLastN    int
}

func defaultGenerateConfig() generateConfig {
	return generateConfig{
		maxTokens:       512,
		temperature:     0.8,
		topK:            40,
		topP:            0.95,
		minP:            0.05,
		repeatPenalty:   1.0,
		freqPenalty:     0.0,
		presencePenalty: 0.0,
		seed:            0xFFFFFFFF, // LLAMA_DEFAULT_SEED
		penaltyLastN:    64,
	}
}

// WithMaxTokens sets the maximum number of tokens to generate.
func WithMaxTokens(n int) GenerateOption {
	return func(c *generateConfig) { c.maxTokens = n }
}

// WithTemperature sets the sampling temperature.
func WithTemperature(t float32) GenerateOption {
	return func(c *generateConfig) { c.temperature = t }
}

// WithTopK sets the top-k sampling parameter.
func WithTopK(k int) GenerateOption {
	return func(c *generateConfig) { c.topK = k }
}

// WithTopP sets the top-p (nucleus) sampling parameter.
func WithTopP(p float32) GenerateOption {
	return func(c *generateConfig) { c.topP = p }
}

// WithMinP sets the min-p sampling parameter.
func WithMinP(p float32) GenerateOption {
	return func(c *generateConfig) { c.minP = p }
}

// WithRepeatPenalty sets the repetition penalty.
func WithRepeatPenalty(p float32) GenerateOption {
	return func(c *generateConfig) { c.repeatPenalty = p }
}

// WithFreqPenalty sets the frequency penalty.
func WithFreqPenalty(p float32) GenerateOption {
	return func(c *generateConfig) { c.freqPenalty = p }
}

// WithPresencePenalty sets the presence penalty.
func WithPresencePenalty(p float32) GenerateOption {
	return func(c *generateConfig) { c.presencePenalty = p }
}

// WithSeed sets the random seed for sampling.
func WithSeed(s int) GenerateOption {
	return func(c *generateConfig) { c.seed = s }
}
