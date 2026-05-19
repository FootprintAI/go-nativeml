# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.6] - 2026-05-19

### Added
- `linux-arm64` prebuilt static libraries (CPU) via `Dockerfile.libs-arm64`
  using the aarch64-linux-gnu cross-compile toolchain.
- `build-libs-linux-arm64` Makefile target.
- `ARCH_SUFFIX` build-arg on the CPU stage of `Dockerfile.libs` so the same
  Dockerfile can output `linux-amd64` or `linux-arm64` paths.

### Changed
- Bumped llama.cpp from `b8772` to `b9222`.
- CGO LDFLAGS now link `-lllama-common -lllama-common-base` instead of
  `-lcommon` across all platforms — upstream split the legacy
  `libcommon.a` into `libllama-common.a` + `libllama-common-base.a`.
  Consumers that build with `-tags llamacpp` (or the cuda/vulkan/whispercpp
  variants) only need to rebuild; no source change required.
- CUDA + Vulkan stages of `Dockerfile.libs` switched to `gcc-12` — gcc-13
  hits a reproducible internal-compiler-error on `fattn-mma-f16` templates
  (CUDA) and `arg.cpp` (Vulkan) at b9222.
- Vulkan stage now installs `spirv-headers` (new b9222 dep —
  `ggml-vulkan` calls `find_package(SPIRV-Headers)`).

### Fixed
- `Dockerfile.libs` and `Dockerfile.android` now copy mtmd headers from
  `tools/mtmd/` (`mtmd.h`, `mtmd-helper.h`). Previously the headers were
  committed manually because the Dockerfiles didn't track them.

### Rebuilt
All seven prebuilt platforms refreshed against b9222:
`linux-amd64`, `linux-amd64-cuda`, `linux-amd64-vulkan`, `linux-arm64`,
`android-arm64`, `darwin-amd64`, `darwin-arm64`.

[Unreleased]: https://github.com/FootprintAI/go-nativeml/compare/v0.1.6...HEAD
[0.1.6]: https://github.com/FootprintAI/go-nativeml/compare/v0.1.5...v0.1.6
