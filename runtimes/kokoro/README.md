# Kokoro Runtime

Small C++ ONNX Runtime integration for Kokoro.

This runtime is intentionally focused:

- macOS-first CMake build
- statically links Pyke's CPU ONNX Runtime archive
- statically builds espeak-ng for C++ phonemization
- converts eSpeak IPA output to the Misaki-style phoneme alphabet Kokoro was trained on
- loads Kokoro voice styles from `voices-v1.0.bin` with `miniz` and `libnpy`
- exposes a small C API in `src/chirp_kokoro.h`
- chunks text over `?`, `!`, `.`, and `,`

Download local test assets:

```console
mkdir -p /tmp/runtimes/kokoro-assets
cd /tmp/runtimes/kokoro-assets
curl -L -o kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

Build and run:

```console
cmake -S runtimes/kokoro -B runtimes/kokoro/build-static
cmake --build runtimes/kokoro/build-static -j
ESPEAK_DATA_PATH="$PWD/runtimes/kokoro/build-static/_deps/espeak_ng-build" \
runtimes/kokoro/build-static/kokoro-basic \
  --model /tmp/runtimes/kokoro-assets/kokoro-v1.0.onnx \
  --voices /tmp/runtimes/kokoro-assets/voices-v1.0.bin \
  --voice af_heart \
  --text "Hello world! Chirp Kokoro is running locally." \
  --output /tmp/runtimes/kokoro-assets/kokoro-basic.wav
```

The Pyke archive is raw LZMA2, so CMake extracts it with inline Python using
`lzma.FORMAT_RAW` and dictionary size `1 << 26`.
