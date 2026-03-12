//go:build llamacpp

// Copyright 2025 FootprintAI
// SPDX-License-Identifier: Apache-2.0
//
// Thin C++ bridge wrapping common_sampler_* (C++ API) into C-callable functions.
// Pure C API functions from llama.h are called directly from CGO — not wrapped here.

#include "wrapper.h"
#include "llama.h"
#include "common.h"
#include "sampling.h"

#include <string>
#include <vector>
#include <cstring>

// Thread-local error string.
static thread_local std::string g_last_error;

static void set_error(const std::string& msg) {
    g_last_error = msg;
}

extern "C" {

go_llama_generate_params go_llama_default_generate_params(void) {
    go_llama_generate_params p;
    p.max_tokens      = 512;
    p.temperature     = 0.8f;
    p.top_k           = 40;
    p.top_p           = 0.95f;
    p.min_p           = 0.05f;
    p.repeat_penalty  = 1.0f;
    p.freq_penalty    = 0.0f;
    p.presence_penalty = 0.0f;
    p.seed            = LLAMA_DEFAULT_SEED;
    p.penalty_last_n  = 64;
    return p;
}

int go_llama_generate(
    void* ctx_ptr,
    void* model_ptr,
    const char* prompt,
    go_llama_generate_params params,
    go_llama_token_callback callback,
    void* user_data)
{
    auto* ctx   = static_cast<llama_context*>(ctx_ptr);
    auto* model = static_cast<llama_model*>(model_ptr);

    if (!ctx || !model || !prompt) {
        set_error("null context, model, or prompt");
        return -1;
    }

    // Clear KV cache from any previous generation so positions don't collide.
    // Use data=false to only clear metadata (positions/sequences), not the underlying buffers.
    llama_memory_clear(llama_get_memory(ctx), false);

    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (!vocab) {
        set_error("failed to get vocab from model");
        return -1;
    }

    // Tokenize prompt.
    const int n_prompt_max = llama_n_ctx(ctx);
    std::vector<llama_token> tokens(n_prompt_max);
    int n_tokens = llama_tokenize(vocab, prompt, (int)strlen(prompt),
                                  tokens.data(), n_prompt_max,
                                  /*add_special=*/true, /*parse_special=*/true);
    if (n_tokens < 0) {
        set_error("tokenization failed (prompt too long?)");
        return -1;
    }
    tokens.resize(n_tokens);

    if (n_tokens == 0) {
        return 0; // empty prompt, nothing to generate
    }

    // Build sampling parameters.
    common_params_sampling sparams;
    sparams.seed           = (uint32_t)params.seed;
    sparams.temp           = params.temperature;
    sparams.top_k          = params.top_k;
    sparams.top_p          = params.top_p;
    sparams.min_p          = params.min_p;
    sparams.penalty_repeat = params.repeat_penalty;
    sparams.penalty_freq   = params.freq_penalty;
    sparams.penalty_present = params.presence_penalty;
    sparams.penalty_last_n = params.penalty_last_n;

    // Create sampler chain.
    common_sampler* smpl = common_sampler_init(model, sparams);
    if (!smpl) {
        set_error("failed to initialize sampler");
        return -1;
    }

    // Process prompt tokens in a single batch.
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i]    = tokens[i];
        batch.pos[i]      = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = (i == n_tokens - 1) ? 1 : 0; // only compute logits for last token
    }
    batch.n_tokens = n_tokens;

    int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        common_sampler_free(smpl);
        set_error("llama_decode failed on prompt");
        return -1;
    }

    // Generation loop.
    char piece_buf[128];
    int n_cur = n_tokens;
    const int n_ctx = (int)llama_n_ctx(ctx);
    const int max_tokens = params.max_tokens > 0 ? params.max_tokens : 512;
    int generated = 0;

    for (int i = 0; i < max_tokens; i++) {
        // Sample next token.
        llama_token new_token = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, new_token, /*accept_grammar=*/true);

        // Check for end-of-generation.
        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }

        generated++;

        // Convert token to text.
        int n_piece = llama_token_to_piece(vocab, new_token,
                                           piece_buf, sizeof(piece_buf) - 1,
                                           /*lstrip=*/0, /*special=*/false);
        if (n_piece < 0) {
            continue;
        }
        piece_buf[n_piece] = '\0';

        // Stream token to callback.
        if (callback) {
            bool cb_ok = callback(piece_buf, n_piece, user_data);
            if (!cb_ok) {
                break;
            }
        }

        // Prepare next batch (single token).
        llama_batch next_batch = llama_batch_init(1, 0, 1);
        next_batch.token[0]    = new_token;
        next_batch.pos[0]      = n_cur;
        next_batch.n_seq_id[0] = 1;
        next_batch.seq_id[0][0] = 0;
        next_batch.logits[0]   = 1;
        next_batch.n_tokens    = 1;
        n_cur++;

        rc = llama_decode(ctx, next_batch);
        llama_batch_free(next_batch);
        if (rc != 0) {
            common_sampler_free(smpl);
            set_error("llama_decode failed during generation");
            return -1;
        }

        // Check context window.
        if (n_cur >= n_ctx) {
            break;
        }
    }

    common_sampler_free(smpl);
    return 0;
}

int go_llama_embeddings(
    void* ctx_ptr,
    void* model_ptr,
    const char* text,
    float* out,
    int max_floats)
{
    auto* ctx   = static_cast<llama_context*>(ctx_ptr);
    auto* model = static_cast<llama_model*>(model_ptr);

    if (!ctx || !model || !text || !out) {
        set_error("null argument to go_llama_embeddings");
        return -1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (!vocab) {
        set_error("failed to get vocab from model");
        return -1;
    }

    // Tokenize.
    const int n_max = llama_n_ctx(ctx);
    std::vector<llama_token> tokens(n_max);
    int n_tokens = llama_tokenize(vocab, text, (int)strlen(text),
                                  tokens.data(), n_max,
                                  /*add_special=*/true, /*parse_special=*/false);
    if (n_tokens < 0) {
        set_error("tokenization failed for embeddings");
        return -1;
    }
    tokens.resize(n_tokens);

    if (n_tokens == 0) {
        set_error("empty text for embeddings");
        return -1;
    }

    // Enable embeddings mode.
    llama_set_embeddings(ctx, true);

    // Create batch.
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i]    = tokens[i];
        batch.pos[i]      = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = 1; // request output for all tokens (needed for embeddings)
    }
    batch.n_tokens = n_tokens;

    int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        set_error("llama_decode failed for embeddings");
        return -1;
    }

    // Get embeddings (use sequence-level if available, else last token).
    const int n_embd = llama_model_n_embd(model);
    if (n_embd > max_floats) {
        set_error("output buffer too small for embeddings");
        return -1;
    }

    // Try sequence embeddings first (pooling models).
    float* emb = llama_get_embeddings_seq(ctx, 0);
    if (!emb) {
        // Fall back to last token embeddings.
        emb = llama_get_embeddings_ith(ctx, -1);
    }
    if (!emb) {
        set_error("failed to get embeddings");
        return -1;
    }

    memcpy(out, emb, n_embd * sizeof(float));

    // Restore non-embedding mode.
    llama_set_embeddings(ctx, false);

    return n_embd;
}

const char* go_llama_last_error(void) {
    return g_last_error.c_str();
}

} // extern "C"
