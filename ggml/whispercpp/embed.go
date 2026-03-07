// Copyright 2025 FootprintAI
// SPDX-License-Identifier: Apache-2.0

package whispercpp

import "embed"

// Embed directives ensure go mod vendor includes headers and prebuilt libraries.
// The embedded filesystem is not used at runtime — CGO links directly via ${SRCDIR} paths.

//go:embed third_party/include/*.h
//go:embed third_party/ggml/include/*.h
//go:embed third_party/prebuilt/darwin-amd64/*.a
//go:embed third_party/prebuilt/linux-amd64/*.a
var _ embed.FS
