# Building

Build the native C/C++ runtime first:

```bash
cmake -S chirp-c -B chirp-c/build
cmake --build chirp-c/build -j
```

Or use the packaging script:

```bash
uv run python scripts/build-libs.py --backend cpu
uv run python scripts/package-libs.py --backend cpu --platform linux-arm64 --archive
```

Build the Go runner:

```bash
cd runner
go build ./cmd/chirp
```

The Go runner links to local `chirp-c/build` during development. It also
checks `runner/third_party/chirp-c` first, which is where prebuilt release
libraries are downloaded.

Download prebuilt native libraries from a GitHub release:

```bash
uv run python scripts/download-libs.py --tag chirp-c-v0.1.0 --backend cpu
```

Native library releases use `chirp-c-v*` tags. The release workflow packages
Linux, macOS, and Windows `chirp-c` archives and uploads them as release
assets.

Run checks:

```bash
uv run --project chirp-c/scripts python -c "import gguf, numpy, torch, safetensors, tqdm"

cd runner
go test ./...
```
