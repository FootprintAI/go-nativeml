// Copyright 2025 FootprintAI
// SPDX-License-Identifier: Apache-2.0
//
// Thin C bridge for llama.cpp C++ APIs (common_sampler_*, etc.)
// Pure C API functions from llama.h are called directly from CGO.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

// Callback for streaming tokens — return false to stop generation.
typedef bool (*go_llama_token_callback)(const char* token, int len, void* user_data);

// Sampling parameters (C-friendly flat struct).
typedef struct {
    int   max_tokens;
    float temperature;
    int   top_k;
    float top_p;
    float min_p;
    float repeat_penalty;
    float freq_penalty;
    float presence_penalty;
    int   seed;
    int   penalty_last_n;
} go_llama_generate_params;

// Returns sensible defaults.
go_llama_generate_params go_llama_default_generate_params(void);

// Generation: tokenize prompt, run decode loop with sampler chain, stream tokens via callback.
// Returns 0 on success, -1 on error (call go_llama_last_error for message).
int go_llama_generate(
    void* ctx_ptr,
    void* model_ptr,
    const char* prompt,
    go_llama_generate_params params,
    go_llama_token_callback callback,
    void* user_data);

// Embeddings: tokenize text, decode, extract embeddings.
// Returns number of floats written, -1 on error.
int go_llama_embeddings(
    void* ctx_ptr,
    void* model_ptr,
    const char* text,
    float* out,
    int max_floats);

// Last error message (thread-local).
const char* go_llama_last_error(void);

// Image data for multimodal generation.
typedef struct {
    const unsigned char* data;
    int size;
} go_llama_image;

// Generation with images: tokenize prompt + images via mtmd, evaluate, then generate.
// mmproj_path: path to the multimodal projector GGUF file.
// Returns 0 on success, -1 on error.
int go_llama_generate_with_images(
    void* ctx_ptr,
    void* model_ptr,
    const char* mmproj_path,
    const char* prompt,
    go_llama_image* images,
    int n_images,
    go_llama_generate_params params,
    go_llama_token_callback callback,
    void* user_data);

// Free cached multimodal context (call on shutdown).
void go_llama_mtmd_free(void);

#ifdef __cplusplus
}
#endif
