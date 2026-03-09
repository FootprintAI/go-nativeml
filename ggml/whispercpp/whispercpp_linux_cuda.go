//go:build whispercpp && linux && cuda

// Copyright 2025 FootprintAI
// SPDX-License-Identifier: Apache-2.0

package whispercpp

/*
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/third_party/prebuilt/linux-amd64-cuda
#cgo linux LDFLAGS: -Wl,--start-group -lwhisper -lcommon -lggml-cpu -lggml-base -lggml -lggml-cuda -Wl,--end-group -lcuda -lcudart -lcublas -lcublasLt -lstdc++ -lm -lpthread -ldl -lrt -lgomp
*/
import "C"
