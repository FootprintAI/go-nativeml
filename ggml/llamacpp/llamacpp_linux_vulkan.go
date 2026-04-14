//go:build llamacpp && linux && vulkan

// Copyright 2025 FootprintAI
// SPDX-License-Identifier: Apache-2.0

package llamacpp

/*
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/third_party/prebuilt/linux-amd64-vulkan
#cgo linux LDFLAGS: -Wl,--start-group -lmtmd -lcommon -lllama -lggml-cpu -lggml-base -lggml -lggml-vulkan -Wl,--end-group -lvulkan -lstdc++ -lm -lpthread -ldl -lrt -lgomp
*/
import "C"
