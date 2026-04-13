// Copyright 2025 FootprintAI
// SPDX-License-Identifier: Apache-2.0

package llamacpp

// GGMLType represents ggml tensor data types, used for KV cache quantization.
// Values match the ggml_type enum in ggml.h.
type GGMLType int

const (
	GGMLTypeF32    GGMLType = 0
	GGMLTypeF16    GGMLType = 1
	GGMLTypeQ4_0   GGMLType = 2
	GGMLTypeQ4_1   GGMLType = 3
	GGMLTypeQ5_0   GGMLType = 6
	GGMLTypeQ5_1   GGMLType = 7
	GGMLTypeQ8_0   GGMLType = 8
	GGMLTypeQ8_1   GGMLType = 9
	GGMLTypeQ2_K   GGMLType = 10
	GGMLTypeQ3_K   GGMLType = 11
	GGMLTypeQ4_K   GGMLType = 12
	GGMLTypeQ5_K   GGMLType = 13
	GGMLTypeQ6_K   GGMLType = 14
	GGMLTypeQ8_K   GGMLType = 15
	GGMLTypeBF16   GGMLType = 30
	GGMLTypeTQ1_0  GGMLType = 34
	GGMLTypeTQ2_0  GGMLType = 35
)
