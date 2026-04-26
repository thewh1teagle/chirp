# Building

Build the native C/C++ runtime first:

```bash
cmake -S chirp/chirp-c -B chirp/chirp-c/build
cmake --build chirp/chirp-c/build -j
```

Build the Go runner:

```bash
cd chirp/runner
go build ./cmd/chirp
```

Run checks:

```bash
uv run --project chirp/chirp-c/scripts python -c "import gguf, numpy, torch, safetensors, tqdm"

cd chirp/runner
go test ./...
```
