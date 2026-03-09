//go:build whispercpp && linux && !cuda && !vulkan

// Copyright 2025 FootprintAI
// SPDX-License-Identifier: Apache-2.0

package whispercpp

/*
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/third_party/prebuilt/linux-amd64
#cgo linux,arm64 LDFLAGS: -L${SRCDIR}/third_party/prebuilt/linux-arm64
#cgo linux LDFLAGS: -Wl,--start-group -lwhisper -lcommon -lggml-cpu -lggml-base -lggml -Wl,--end-group -lstdc++ -lm -lpthread -ldl -lrt -lgomp
*/
import "C"
