// Copyright 2025 FootprintAI
// SPDX-License-Identifier: Apache-2.0
//
// C bridge that wraps the CGO-exported Go callback into a standard C function pointer.
// This file is compiled by CGO and has access to _cgo_export.h.

#include <stdbool.h>
#include "_cgo_export.h"

// goTokenCallback returns GoInt32 (0 = stop, 1 = continue).
// This bridge converts it to the bool expected by go_llama_token_callback.
bool goTokenCallbackBridge(const char* token, int len, void* user_data) {
    return goTokenCallback((char*)token, len, user_data) != 0;
}
