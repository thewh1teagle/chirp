# Building

Build the native C/C++ runtime first:

```console
cmake -S chirp-c -B chirp-c/build
cmake --build chirp-c/build -j
```

Or use the packaging script:

```console
uv run python scripts/build-libs.py --backend cpu
uv run python scripts/package-libs.py --backend cpu --platform linux-arm64 --archive
```

Release builds use accelerated backends by default: `metal` on macOS and
`vulkan` on Linux/Windows.

Prepare local GGUF models under the ignored `models/` directory:

```console
uv run hf download Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --local-dir models/Qwen3-TTS-12Hz-0.6B-Base

uv run --project chirp-c/scripts python chirp-c/scripts/convert_model_to_gguf.py \
  --input models/Qwen3-TTS-12Hz-0.6B-Base \
  --output models/qwen3-tts-0.6b-q4_k.gguf \
  --type q4_k

uv run --project chirp-c/scripts python chirp-c/scripts/convert_codec_to_gguf.py \
  --input models/Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer \
  --output models/qwen3-tts-tokenizer-f16.gguf \
  --type f16
```

Or download packaged GGUF model bundles from a model release:

```console
gh release download chirp-models-v0.1.3 \
  --pattern 'chirp-models-q5_0.tar.gz' \
  --dir dist

mkdir -p models
tar -xzf dist/chirp-models-q5_0.tar.gz -C models
```

Build the Go runner:

```console
cd chirp-runner
go build ./cmd/chirp
```

The Go runner links to local `chirp-c/build` during development. It also
checks `chirp-runner/third_party/chirp-c` first, which is where prebuilt release
libraries are downloaded.

Download prebuilt native libraries from a GitHub release:

```console
uv run python scripts/download-libs.py --tag chirp-c-v0.3.0 --backend vulkan
```

Native library releases use `chirp-c-v*` tags. The release workflow packages
Linux, macOS, and Windows `chirp-c` archives and uploads them as release
assets.

Runner releases use `chirp-runner-v*` tags. The runner release workflow
downloads a pinned `chirp-c` release, builds the Go `cmd/chirp` binary with
cgo enabled, and uploads platform archives:

```console
git tag chirp-runner-v0.1.0
git push origin chirp-runner-v0.1.0
```

Manual runner releases can choose the native library version:

```console
gh workflow run release-chirp-runner.yml \
  --ref main \
  -f version=chirp-runner-v0.1.0 \
  -f chirp_c_tag=chirp-c-v0.3.0
```

Run checks:

```console
uv run --project chirp-c/scripts python -c "import gguf, numpy, torch, safetensors, tqdm"

cd chirp-runner
go test ./...
```
