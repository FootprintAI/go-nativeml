//go:build cuda && vulkan

// Copyright 2025 FootprintAI
// SPDX-License-Identifier: Apache-2.0

package whispercpp

// cuda and vulkan build tags are mutually exclusive.
// Setting both will produce a compile error.
var _ int = "cuda and vulkan are mutually exclusive"
