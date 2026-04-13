//go:build llamacpp && android && arm64

// Copyright 2025 FootprintAI
// SPDX-License-Identifier: Apache-2.0

package llamacpp

/*
#cgo android,arm64 LDFLAGS: -L${SRCDIR}/third_party/prebuilt/android-arm64
#cgo android LDFLAGS: -Wl,--start-group -lcommon -lllama -lggml-cpu -lggml-base -lggml -Wl,--end-group -lstdc++ -lm -ldl -llog
*/
import "C"
